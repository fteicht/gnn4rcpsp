#!/bin/bash

NB_PROCESSORS=`nproc --all`
# echo $NB_PROCESSORS

BENCHMARKS=(1460 509 459 450 514 234 237 391 285 1720 231 69 406 399 1728 1948 1949 502 461 1469 1425 471 464 527 2032 1244 1858 19 1112 303 1556 242 37 66 522 473 1555)
# echo ${BENCHMARKS[@]}

NB_BENCHMARKS=${#BENCHMARKS[@]}
# echo $NB_BENCHMARKS

NB_POOLS=$((NB_BENCHMARKS / (NB_PROCESSORS - 1) + 1))
# echo $NB_POOLS

DIRNAME="execute_schedules_$(date +"%Y%m%d%H%M%S")"
mkdir -p experiments/$DIRNAME
# echo $DIRNAME

BENCH_ID=0

for (( POOL = 0 ; POOL < NB_POOLS ; POOL++))
do
    # echo "POOL: $POOL"
    for (( PROCESS = 0 ; PROCESS < NB_PROCESSORS - 2 ; PROCESS++ ))
    do
        if [[ $BENCH_ID -lt $NB_BENCHMARKS ]];
        then
            # echo "   BENCH_ID: $BENCH_ID"
            BENCH=${BENCHMARKS[$BENCH_ID]}
            # echo "   BENCH: $BENCH"
            BENCH_ID=$((BENCH_ID + 1))
            python execute_schedules.py $DIRNAME $BENCH &
            pids[${PROCESS}]=$!
        fi
    done
    for pid in ${pids[*]}; do
        wait $pid
    done
done
