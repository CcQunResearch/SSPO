#!/bin/bash

k_num=2285206
num_iterations=30
for ((i=0; i<num_iterations; i++))
do
    kill -9 $k_num

    k_num=$((k_num + 1))
done