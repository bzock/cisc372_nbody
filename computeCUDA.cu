
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>


//16 is best number of threads per block for system
#define BLOCK 16
//values to retain between function calls
static vector3* d_pos = 0;
static vector3* d_vel = 0;
static double*  d_mass = 0;
static vector3* d_accels = 0;
static int allocated_n = 0;

//kernel 1 function will calculate acceleration of objects
__global__ void accelK(vector3* pos, double* mass, vector3* accels, int n) {

    __shared__ vector3 shPos[BLOCK];
    __shared__ double  shMass[BLOCK];
    //thread id
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    //accel starts at 0
    double myX = 0.0, myY = 0.0, myZ = 0.0;
    if (i < n) {
        myX = pos[i][0];
        myY = pos[i][1];
        myZ = pos[i][2];
    }

    //shared memory
    if (threadIdx.y == 0) {
        if (j < n) {
            shPos[threadIdx.x][0] = pos[j][0];
            shPos[threadIdx.x][1] = pos[j][1];
            shPos[threadIdx.x][2] = pos[j][2];
            shMass[threadIdx.x]  = mass[j];
        } else {
            shPos[threadIdx.x][0] = 0.0;
            shPos[threadIdx.x][1] = 0.0;
            shPos[threadIdx.x][2] = 0.0;
            shMass[threadIdx.x]   = 0.0;
        }
    }
    //wait for all threads to finsih
    __syncthreads();

    //accel computation for all vectors
    if (i < n && j < n) {
        if (i == j) {
            FILL_VECTOR(accels[i*n+j], 0, 0, 0);
        } else {
            double dx = myX - shPos[threadIdx.x][0];
			double dy = myY - shPos[threadIdx.x][1];
			double dz = myZ - shPos[threadIdx.x][2];

            double magsq = dx * dx + dy * dy + dz * dz + 1e-12;
            double mag = sqrt(magsq);
            double accel = -1 * GRAV_CONSTANT * shMass[threadIdx.x] / magsq;

            accels[i * n + j][0] = accel * dx / mag;
            accels[i * n + j][1] = accel * dy / mag;
            accels[i * n + j][2] = accel * dz / mag;
        }
    }
} 

//kernel 2 function adds all the acceleration in accelersation matrix, updates velocity of objects and updates position of objects
__global__ void reduceK(vector3* pos, vector3* vel, vector3* accels, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    //cut threads beyond object
    if (i >= n){
        return;
    }
    //objects start at 0 accel
    vector3 accel_sum = {0, 0, 0};

    //sum cells
    for (int j = 0; j < n; j++) {
        for (int k = 0; k < 3; k++)
            accel_sum[k] += accels[i * n + j][k];
    }
   
    //times interval from config
    for (int k = 0; k < 3; k++) {
        vel[i][k] += accel_sum[k] * INTERVAL;
        pos[i][k] += vel[i][k] * INTERVAL;
    }
}

extern "C" void compute() {
    //number of objects in sim
    int n = NUMENTITIES;

    //if sim size changes free up the old memory
    if (allocated_n != n) {
        if (d_pos) {
            cudaFree(d_pos);
            cudaFree(d_vel);
            cudaFree(d_mass);
            cudaFree(d_accels);
        }

	//allocates memory on gpu
        cudaMalloc(&d_pos, sizeof(vector3) * n);
        cudaMalloc(&d_vel, sizeof(vector3) * n);
        cudaMalloc(&d_mass, sizeof(double)  * n);
        cudaMalloc(&d_accels, sizeof(vector3) * n * n);

        allocated_n = n;
    }

    //copy host data to device
    cudaMemcpy(d_pos,  hPos, sizeof(vector3) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel,  hVel, sizeof(vector3) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, sizeof(double)  * n, cudaMemcpyHostToDevice);

    //kernel 1 
    dim3 blockSize(16, 16);
    dim3 gridSize((n + 15) / 16, (n + 15) / 16);
    accelK<<<gridSize, blockSize>>>(d_pos, d_mass, d_accels, n);
    cudaDeviceSynchronize();

    //kernel 2
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    reduceK<<<blocks, threads>>>(d_pos, d_vel, d_accels, n);
    
    //synchronize data after kernel 2 finsihed
    cudaDeviceSynchronize();

    //update new position and velocity to host
    cudaMemcpy(hPos, d_pos, sizeof(vector3) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_vel, sizeof(vector3) * n, cudaMemcpyDeviceToHost);
}
