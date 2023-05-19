#!/bin/bash

ROOT=./experiment_results_17052023

variant="pref"
activ_variant="--variant"
seeds="123456789 987654321 12121212 34343434 565656 787878 909090 123123 456456 789789"

infra_strat="ENER"

for i in 2 2 3
do
    for seed in $seeds
    do
	echo $i $activ_variant $infra_strat
	if [[ $infra_strat = "ENER" ]]; then
	    python3 ./main.py --config experiments/decentralized-4worker.yaml --refs ../data/release01-2021-12-29/ref-solutions.csv ../data/release01-2021-12-29/data.csv  --experiment $i $activ_variant --seed $seed
	else
	    python3 ./main.py --config experiments/decentralized-4worker.yaml --refs ../data/release01-2021-12-29/ref-solutions.csv ../data/release01-2021-12-29/data.csv  --experiment $i $activ_variant --seed $seed --infra_perf
	fi

	mkdir -p $ROOT/$variant/$i\_$infra_strat/$seed/
	mv *.pkl $ROOT/$variant/$i\_$infra_strat/$seed/
    done
    if [[ $infra_strat = "ENER" ]]; then
	infra_strat="PERF"
    fi
done

mkdir -p ./latex
python3 simdex-experiments-plots-v2.py
