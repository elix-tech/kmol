#!/usr/bin/env bash

INPUT=$1
OUTPUT=$2

for filename in $INPUT/*; do
    name=$(basename $filename | sed s/\.json//)
    python run.py train $filename
    python run.py find_best_checkpoint $filename > $OUTPUT/$name.out
done
