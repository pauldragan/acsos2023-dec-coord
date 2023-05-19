#!/bin/bash

python3 ./pydcop-experiment.py

mkdir -p ./latex
python3 pydcop-experiment-plots-v2.py
