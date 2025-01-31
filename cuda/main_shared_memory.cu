/*
    * File path: 
    * .../N-body-simulation/cuda
    * 
    * To compile:
    * nvcc main_shared_memory.cu -o main
    * 
    * To run:
    * ./main N 
    * where N is number of bodies
    * 
    * To (run) profile:
    * nvprof ./main N
    * 
    * or
    * nsys profile --stats=true --output=report ./main N
*/


#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <sys/time.h>


//---------- Constants ----------//

#define G 6.67430e-11
#define DELTA_TIME 0.01 // time step in simulation time (in seconds)
#define T_END 10000 // how many seconds (in real time) the simsulation will run
// #define N 10 // number of bodies

#define BLOCK_SIZE 256 //128, 256, 512, 1024 are common block sizes


// CUDA error checking
#define CUDACHECK(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)
inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}


//---------- Structs ----------//
struct Body {
    double3 position;
    double3 velocity;
    double3 force;
    float mass;
};



//---------- Functions ----------//

__device__ double dot_product(double3 a) {
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

//replace nan value with 0
__device__ void check_and_replace_nan(double* value) {
    if (isnan(*value)) {
        *value = 0.0f; 
    }
}


// this function calculate initial position of the N bodies in the our empty universum
void initBodies(Body *bodies, int n) {
    float destination_parameter = 1.0e+3;
    float velocity_parameter = 1.0e+0;
    float mass_parameter = 1.0e+24;


    for (int i = 0; i < n; i++) {
        bodies[i].position.x = ((rand() % 1000) - 500)*destination_parameter;
        bodies[i].position.y = ((rand() % 1000) - 500)*destination_parameter;
        bodies[i].position.z = ((rand() % 1000) - 500)*destination_parameter;
        
        bodies[i].velocity.x = ((rand() % 1000) - 500)*velocity_parameter;
        bodies[i].velocity.y = ((rand() % 1000) - 500)*velocity_parameter;
        bodies[i].velocity.z = ((rand() % 1000) - 500)*velocity_parameter;
        
        bodies[i].mass = (rand() % 1000 + 1) * mass_parameter;                                       
    }
}

//CUDA kernel that calculates the gravitational forces acting on each body
__global__ void calculate_parameters(Body *bodies, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= n) return;

    __shared__ Body shared_bodies[BLOCK_SIZE];
    Body self_body = bodies[i];

    double3 f;
    f.x =0.0f;
    f.y =0.0f;
    f.z =0.0f;
    

    for (int tile = 0; tile < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        int idx = tile * BLOCK_SIZE + threadIdx.x;
        if (idx < n) {
            shared_bodies[threadIdx.x] = bodies[idx];
        }
        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; ++j) {
            int global_j = tile * BLOCK_SIZE + j;
            if (global_j >= n || i == global_j) continue;

            double3 diff;
            diff.x = shared_bodies[j].position.x - self_body.position.x;
            diff.y = shared_bodies[j].position.y - self_body.position.y;
            diff.z = shared_bodies[j].position.z - self_body.position.z;

            double dist = sqrt(dot_product(diff));
            double forceMagnitude = G * self_body.mass * shared_bodies[j].mass / (dist * dist + 1e-10);

            f.x += forceMagnitude * diff.x / dist;
            f.y += forceMagnitude * diff.y / dist;
            f.z += forceMagnitude * diff.z / dist;
        }
        __syncthreads();
    }

    bodies[i].force = f;
}

//CUDA kernel that updates the positions and velocities of each body based on the forces calculated.
__global__ void updateBodies(Body *bodies, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        bodies[i].velocity.x += bodies[i].force.x / bodies[i].mass * DELTA_TIME;
        bodies[i].velocity.y += bodies[i].force.y / bodies[i].mass * DELTA_TIME;
        bodies[i].velocity.z += bodies[i].force.z / bodies[i].mass * DELTA_TIME;

        bodies[i].position.x += bodies[i].velocity.x * DELTA_TIME / 2.0f;
        bodies[i].position.y += bodies[i].velocity.y * DELTA_TIME / 2.0f;
        bodies[i].position.z += bodies[i].velocity.z * DELTA_TIME / 2.0f;


        check_and_replace_nan(&bodies[i].force.x);
        check_and_replace_nan(&bodies[i].force.y);
        check_and_replace_nan(&bodies[i].force.z);

        check_and_replace_nan(&bodies[i].position.x);
        check_and_replace_nan(&bodies[i].position.y);
        check_and_replace_nan(&bodies[i].position.z);
    }
}

void save_results(Body *bodies, int n){ 

    FILE *file;
    file = fopen("results.txt", "a");


    //check if the file was opened
    if(file == NULL){ 
        printf("Error opening file\n");
        exit(1);
    }

    // format of the result: body_number mass position_x position_y position_z
    for(int i = 0; i < n; i++){
        fprintf(file, "%d %f %f %f %f\n",i, bodies[i].mass, bodies[i].position.x, bodies[i].position.y, bodies[i].position.z);
    }
    fclose(file);

    // printf("Results saved to results.txt\n");
}


int main(int argc, char **argv) {
    //remove the file if it already exists
    remove("results.txt");

    srand(time(NULL));

    if (argc != 2){
        printf("Error: wrong number of arguments\n");
        exit(1);
    }

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <number of bodies>\n", argv[0]);
        exit(1);
    }

    int n = atoi(argv[1]); // Convert command-line argument to integer
    
    if (n <= 0) {
        fprintf(stderr, "Number of bodies must be a positive integer\n");
        exit(1);
    }


    // Get the number of CUDA devices
    int deviceCount;

    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices available\n");
        return 1;
    }

    // Query CUDA device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);                                      // Assuming using the first device
    // printf("Device Name: %s\n", prop.name);                                 //is an ASCII string identifying the device
    // printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);    // is the number of multiprocessors on the device
    // printf("Total global memory: %zu bytes\n", prop.totalGlobalMem);        //is the total amount of global memory available on the device in bytes
    // printf("Maximum number of threads per block: %d bytes\n", prop.maxThreadsPerBlock);   // is the maximum number of threads per block;
    // printf("maximum size of each dimension of a block: %d bytes\n", prop.maxThreadsDim[3]);      // contains the maximum size of each dimension of a block;
    // printf("maximum size of each dimension of a grid: %d bytes\n", prop.maxGridSize[3]);        // contains the maximum size of each dimension of a grid;
    // printf("clock frequency in kilohertz: %d bytes\n", prop.clockRate);             //is the clock frequency in kilohertz;
    

    // Define CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start recording
    cudaEventRecord(start);
    double start_time = clock();

    Body *h_bodies = (Body*)malloc(n * sizeof(Body));  // Host bodies - CPU
    Body *d_bodies;  // Device bodies - GPU


    int blockSize=BLOCK_SIZE;
    int blocks = (n + blockSize - 1) / blockSize;



    initBodies(h_bodies, n);

    CUDACHECK(cudaMalloc(&d_bodies, n * sizeof(Body)));
    CUDACHECK(cudaMemcpy(d_bodies, h_bodies, n * sizeof(Body), cudaMemcpyHostToDevice));


    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);


    for (int iter = 0; iter < (T_END); iter++) {
        calculate_parameters<<<blocks, blockSize, sizeof(Body) * BLOCK_SIZE,stream1>>>(d_bodies, n);
        updateBodies<<<blocks, blockSize, sizeof(Body) * BLOCK_SIZE, stream2>>>(d_bodies, n);

        //to save
        // CUDACHECK(cudaMemcpy(h_bodies, d_bodies, n * sizeof(Body), cudaMemcpyDeviceToHost));
        // save_results(h_bodies, n);
    }
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);


    double end_time = clock();

    // Stop recording

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate and print the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken for execution: %f milliseconds\n", milliseconds);

    printf("Time taken for execution: %f seconds\n", (end_time - start_time) / CLOCKS_PER_SEC);
    
    
    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    cudaFree(d_bodies);
    free(h_bodies);

    return 0;
}