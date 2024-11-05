/*

This is a simple example of OpenMP in C

File path: 
.../N-body-simulation/openmp

To compile:
gcc -fopenmp main_single_core.cpp -o main

or

g++ -fopenmp main.cpp -o main

To run:
./main

*/


/*
TO DO:
check if the number of threads is a positive integer
check if the number of threads is less than or equal to the number of cores
make function to print error message and stop


create function to free the memory
create function to calculate the time of the simulation with OpenMP
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include <time.h>


#define G 6.67430e-11
#define DELTA_TIME 1 // time step in simulation time (in seconds)
#define T_END 1000000// how many seconds (in real time) the simulation will run
#define N 2 // number of bodies

struct float3 {
    float x, y, z;
};

struct Body {
    float3 position;
    float3 velocity;
    float3 acceleration;
    float mass;
};


double addVectors(float3 a, float3 b){
    return (a.x*b.x + a.y*b.y + a.z*b.z);
}

double scaleVector(double scale, float3 a){
    return (scale*a.x, scale*a.y, scale*a.z);
}

double substractVector(float3 a, float3 b){
    return (a.x - b.x, a.y - b.y, a.z - b.z);
}

double mod(float3 a){
    return sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
}


void init_bodies(Body *bodies, int n){
/* this function calculate initial position of the N bodies in the our empty universum*/
    float destination_parameter = 1.0e+4;
    float mass_parameter = 1.0e+24;

    for(int i = 0; i < n; i++){
        //random parameters

        bodies[i].position.x = ((rand() % 1000) - 500)*destination_parameter;
        bodies[i].position.y = ((rand() % 1000) - 500)*destination_parameter;
        bodies[i].position.z = ((rand() % 1000) - 500)*destination_parameter;
        
        bodies[i].velocity.x = ((rand() % 1000) - 500)*destination_parameter;
        bodies[i].velocity.y = ((rand() % 1000) - 500)*destination_parameter;
        bodies[i].velocity.z = ((rand() % 1000) - 500)*destination_parameter;
        

    
        // bodies[i].acceleration.x = (rand()%20 -10);
        // bodies[i].acceleration.y = (rand()%20 -10);
        // bodies[i].acceleration.z = (rand()%20 -10);

        bodies[i].mass = (rand() % 1000 + 1) * mass_parameter;

    }
};

// void resolve_colisions(Body *bodies, int n){
//     for(int i = 0; i < n; i++){
//         for(int j = i+1; j < n; j++){
//             if(bodies[i].position.x == bodies[j].position.x && bodies[i].position.y == bodies[j].position.y && bodies[i].position.z == bodies[j].position.z){
// 				// int a=0
                
//                 double temp_vx, temp_vy, temp_vz;

//                 temp_vx = bodies[i].velocity.x;
//                 temp_vy = bodies[i].velocity.y;
//                 temp_vz = bodies[i].velocity.z;

//                 bodies[i].velocity.x = bodies[j].velocity.x;
//                 bodies[i].velocity.y = bodies[j].velocity.y;
//                 bodies[i].velocity.z = bodies[j].velocity.z;

//                 bodies[j].velocity.x = temp_vx;
//                 bodies[j].velocity.y = temp_vy;
//                 bodies[j].velocity.z = temp_vz;
//             }
//         }
//     }
// };


void calculate_parameters(Body *bodies,int n){
    for (int i = 0; i < n; i++){
        float3 acceleration = {0, 0, 0};
        float epsilon = 1e-10;

        for (int j = 0; j < n; j++){
            if (i != j){

                float force_X = (G * bodies[i].mass * bodies[j].mass) *((bodies[i].position.x - bodies[j].position.x)+epsilon) / (pow(abs(bodies[i].position.x - bodies[j].position.x),3.0)+epsilon);
                float force_Y = (G * bodies[i].mass * bodies[j].mass) *((bodies[i].position.y - bodies[j].position.y)+epsilon) / (pow(abs(bodies[i].position.y - bodies[j].position.y),3.0)+epsilon);
                float force_Z = (G * bodies[i].mass * bodies[j].mass) *((bodies[i].position.z - bodies[j].position.z)+epsilon) / (pow(abs(bodies[i].position.z - bodies[j].position.z),3.0)+epsilon);
                
                acceleration.x += force_X/bodies[i].mass;
                acceleration.y += force_Y/bodies[i].mass;
                acceleration.z += force_Z/bodies[i].mass;
                
            }
        }
        bodies[i].acceleration = acceleration;
    }
};

void check_and_replace_nan(float* value) {
    if (isnan(*value)) {
        *value = 0.0f; // Zamiana NaN na 0
    }
}

void update_velocity_and_position(Body *bodies, int n){
    for(int i = 0; i < n; i++){

        // Aktualizacja pozycji z uwzględnieniem współczynnika 1/2 * dt^2
        bodies[i].position.x += bodies[i].velocity.x * DELTA_TIME + 0.5 * bodies[i].acceleration.x * DELTA_TIME * DELTA_TIME;
        bodies[i].position.y += bodies[i].velocity.y * DELTA_TIME + 0.5 * bodies[i].acceleration.y * DELTA_TIME * DELTA_TIME;
        bodies[i].position.z += bodies[i].velocity.z * DELTA_TIME + 0.5 * bodies[i].acceleration.z * DELTA_TIME * DELTA_TIME;

        // Nowa prędkość po pełnym kroku czasowym z nową przyspieszeniem
        bodies[i].velocity.x += bodies[i].acceleration.x * DELTA_TIME;
        bodies[i].velocity.y += bodies[i].acceleration.y * DELTA_TIME;
        bodies[i].velocity.z += bodies[i].acceleration.z * DELTA_TIME;

        check_and_replace_nan(&bodies[i].position.x);
        check_and_replace_nan(&bodies[i].position.y);
        check_and_replace_nan(&bodies[i].position.z);
    }
};




void save_results(Body *bodies, int n, char filename[100]){


    FILE *file;
    file = fopen(filename, "a");
    //check if the file was opened
    if(file == NULL){ 
        printf("Error opening file\n");
        exit(1);
    }
    //result format
    for(int i = 0; i < n; i++){
        /* format of the result: 
        body_number mass position_x position_y position_z */

        fprintf(file, "%d %f %f %f %f\n",i, bodies[i].mass, bodies[i].position.x, bodies[i].position.y, bodies[i].position.z);
    }

    fclose(file);

    // printf("Results saved to results.txt\n");
}


//print initial conditions
void print_bodies(Body *bodies, int n){
    for(int i = 0; i < n; i++){
        printf("Body %d\n", i);
        printf("Position: %f %f %f\n", bodies[i].position.x, bodies[i].position.y, bodies[i].position.z);
        printf("Velocity: %f %f %f\n", bodies[i].velocity.x, bodies[i].velocity.y, bodies[i].velocity.z);
        printf("Mass: %f\n", bodies[i].mass);
    }
};




int main(){
    // int simulation_time_end = T_END/DELTA_TIME;
    int threads_number = 6;

    //set number of threads
    printf("All of threads: %d\n", omp_get_max_threads());
    
    if (threads_number > omp_get_max_threads() || threads_number < 1){
        printf("Error: wrong number of threads\n");
        exit(1);
    }
    omp_set_num_threads(threads_number);
    printf("Set threads: %d\n", threads_number);
    

    //init and print bodies
    Body bodies[N];
    init_bodies(bodies, N);
    // print_bodies(bodies, N);



    
    //time measurement
    clock_t start_time, end_time;
    double total_time;


    //filename == actual date and time
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    char filename[100];
    sprintf(filename, "results_%d-%d-%d_%d:%d:%d.txt", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec); //

    //simulation loop
    start_time = clock();

    for (int t=0; t < T_END; t++){
        calculate_parameters(bodies, N);
        update_velocity_and_position(bodies, N);
        // resolve_colisions(bodies, N);
        save_results(bodies, N, filename);
    }
    end_time = clock();
    total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Total time: %f\n", total_time);



    return 0;
};


