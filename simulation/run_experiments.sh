#!/bin/bash

ROOT=./experiment_results_december_3

for variant in $@
do

    if [ $variant = "pref" ]
    then
	activ_variant="--variant"
	seeds="123456789 987654321 12121212 34343434 565656 787878 909090 123123 456456 789789"
	# seeds="123123 456456 789789"
	# seeds="123456789"
    else
	activ_variant=""
	seeds="123456789"
    fi

    for seed in $seeds
    do
	for i in 1 2 3
	do
	    echo $i $activ_variant
	    python3 ./main.py --config experiments/decentralized-4worker.yaml --refs ../data/release01-2021-12-29/ref-solutions.csv ../data/release01-2021-12-29/data.csv  --experiment $i $activ_variant --seed $seed
	    mkdir -p $ROOT/$variant/$i/$seed/
	    mv *.pkl $ROOT/$variant/$i/$seed/
	done
    done
done
