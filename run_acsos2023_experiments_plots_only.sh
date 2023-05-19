#!/bin/bash

cd simulation/
mkdir -p ./latex

python3 simdex-experiments-plots-v2.py
python3 pydcop-experiment-plots-v2.py
