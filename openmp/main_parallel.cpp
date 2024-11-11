/*
    * This is a simple example of OpenMP in C
    * 
    * File path: 
    * .../N-body-simulation/openmp
    * 
    * To compile:
    * gcc -fopenmp main_parallel.cpp -o main
    * 
    * or
    * 
    * g++ -fopenmp main_parallel.cpp -o main
    * 
    * To run: (N - number of bodies, X - number of threads)
    * ./main N X 
    * 
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include <time.h>


//-----------------structures-----------------
#define G 6.67430e-11
#define DELTA_TIME 0.01 // time step in simulation time (in seconds)
#define T_END 100000 // how many seconds (in real time) the simulation will run

#define N 10 // number of bodies

struct double3 {
    double x, y, z;
};

struct Body {
    double3 position;
    double3 velocity;
    double3 force;
    float mass;
};

//-----------------functions-----------------

float dot_product(double3 a) {
    return a.x * a.x + a.y * a.y + a.z * a.z;
}

//replace nan value with 0
void check_and_replace_nan(double* value) {
    if (isnan(*value)) {
        *value = 0.0f; 
    }
}

/* this function calculate initial position of the N bodies in the our empty universum*/
void init_bodies(Body *bodies, int n){
    float destination_parameter = 1.0e+2;
    float velocity_parameter = 1.0e+0;
    float mass_parameter = 1.0e+24;

    for(int i = 0; i < n; i++){
        //
        
        // //custom parameters for 3-body simimilatiom just uncomment the code below, and set numbers of bodies to 3 (#define N 3)
        // float MASS = 1.0e+15;
        // if (i==0) {
        //     bodies[i].position=(float3){100.0,0.0, 0.0};
        //     bodies[0].velocity = (float3){0.0, 10.0, 0.0};
        //     bodies[0].mass = MASS;
        // }
        
        // if (i=1) {
        //     bodies[i].position=(float3){-100.0, 0.0, 0.0};
        //     bodies[1].velocity = (float3){0.0, -20.0, 0.0};
        //     bodies[1].mass = MASS;
        // }

        // if (i=2) {
        //     bodies[i].position=(float3){0.0, 100.0, 0.0 };
        //     bodies[2].velocity = (float3){-20.0, 0.0, 0.0};
        //     bodies[2].mass = MASS;
        // }

        // random start parameters
        bodies[i].position.x = ((rand() % 1000) - 500)*destination_parameter;
        bodies[i].position.y = ((rand() % 1000) - 500)*destination_parameter;
        bodies[i].position.z = ((rand() % 1000) - 500)*destination_parameter;
        
        bodies[i].velocity.x = ((rand() % 1000) - 500)*velocity_parameter;
        bodies[i].velocity.y = ((rand() % 1000) - 500)*velocity_parameter;
        bodies[i].velocity.z = ((rand() % 1000) - 500)*velocity_parameter;
        
        bodies[i].mass = (rand() % 1000 + 1) * mass_parameter;

    }
}


void calculate_parameters(Body *bodies,int n){
    float epsilon = 1e-10;
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++){
        double3 f;
        f.x = 0.0;
        f.y = 0.0;
        f.z = 0.0;

        for (int j = 0; j < n; j++){
            if (i != j){

            double3 diff;
            diff.x = bodies[j].position.x - bodies[i].position.x;
            diff.y = bodies[j].position.y - bodies[i].position.y;
            diff.z = bodies[j].position.z - bodies[i].position.z;

            double dist = sqrtf(dot_product(diff)); 
            double forceMagnitude = G * bodies[i].mass * bodies[j].mass / (dist * dist + 1e-10f);  //+ 1e-10f -> prevention of division by zero
            
            f.x += forceMagnitude * diff.x / dist;
            f.y += forceMagnitude * diff.y / dist;
            f.z += forceMagnitude * diff.z / dist;
            
            }
        }      
        
        bodies[i].force = f;
    }
}

void update_velocity_and_position(Body *bodies, int n){
    #pragma omp parallel for
    for(int i = 0; i < n; i++){
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

void save_results(Body *bodies, int n, char filename[100]){

    #pragma omp critical
    {
        FILE *file;
        file = fopen("results.txt", "a");
        //check if the file was opened
        if(file == NULL){ 
            printf("Error opening file\n");
            exit(1);
        }
        //result format
        for(int i = 0; i < n; i++){
            // format of the result: body_number mass position_x position_y position_z
            fprintf(file, "%d %f %f %f %f\n",i, bodies[i].mass, bodies[i].position.x, bodies[i].position.y, bodies[i].position.z);
        }
        fclose(file);
    }
    // printf("Results saved to results.txt\n");
}

int main(int argc, char *argv[]){
    remove("results.txt");

    if (argc != 3){
        printf("Error: wrong number of arguments\n");
        exit(1);
    }
    int n = atoi(argv[1]);
    int threads_number = atoi(argv[2]);

    //set number of threads
    printf("All of threads: %d\n", omp_get_max_threads());
    
    if (threads_number > omp_get_max_threads() || threads_number < 1){
        printf("Error: wrong number of threads\n");
        exit(1);
    }
    omp_set_num_threads(threads_number);
    printf("Set threads: %d\n", threads_number);
    
//---------------------------------------------- 
    //init and print bodies
    // Body bodies[n];

    Body *bodies = (Body *)malloc(N * sizeof(Body));
    init_bodies(bodies, n);

    //time measurement
    double start_time, end_time;
    double total_time;


    //filename == actual date and time
    time_t t = time(NULL);
    struct tm tm = *localtime(&t);
    char filename[100];
    sprintf(filename, "results_%d-%d-%d_%d:%d:%d.txt", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec); //

    //simulation loop
    start_time = omp_get_wtime();

    #pragma omp parallel for num_threads(threads_number)
    for (int t=0; t < T_END; t++){
        calculate_parameters(bodies, n);
        update_velocity_and_position(bodies, n);

        save_results(bodies, n, filename);
    }
    end_time = omp_get_wtime();
    total_time = end_time - start_time;
    printf("Total time: %f seconds\n", total_time);

    free(bodies);

    return 0;
}
