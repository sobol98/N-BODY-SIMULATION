#!/bin/bash



#Settings

N=1000
START_CORE_NUMBERS=1
END_CORE_NUMBERS=1
REPEATS=10
OUTPUT_FILE="output_results_cuda_stream.txt"


> $OUTPUT_FILE



#main loop
for ((core_id = START_CORE_NUMBERS; core_id <= END_CORE_NUMBERS; core_id++)); do
        echo "Running for Number of bodys=$N, core numbers=$core_id " >> $OUTPUT_FILE
        echo "Running for Number of bodys=$N, core numbers=$core_id "

        for ((i=1;i<=REPEATS;i++)); do
                #COMMAND="./openmp/main_p $N $core_id"
                #COMMAND="./openmp/main_s $N"
                COMMAND="./cuda/main_shared $N"
                echo "Running: $COMMAND (iteration $i)" >> $OUTPUT_FILE
                echo "Running: $COMMAND (iteration $i)"
                eval $COMMAND >> $OUTPUT_FILE 2>&1
                echo "-------------------------" >> $OUTPUT_FILE
        done

        echo "End calculations for settings N=$N, core_id=$core_id" >> $OUTPUT_FILE
        echo "End calculations for settings N=$N, core_id=$core_id"
        echo "================================" >> $OUTPUT_FILE
done
echo "Results saved in $OUTPUT_FILE"








