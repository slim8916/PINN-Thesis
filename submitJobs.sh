#!/bin/bash

input_file=experiments.txt
start=$1
end=$2
script=$3



# Read lines from n to m
sed -n "${start},${end}p" "$input_file" | while IFS= read -r line; do
  id_number=$(echo $line | grep -o "'id':[0-9]*" | sed 's/[^0-9]//g')
  if [ ! -f "./Res/${script%.py}/model_$id_number/final_model.keras" ]; then
    rm -rf ./Res/"${script%.py}"/model_"$id_number" && mkdir -p ./Res/"${script%.py}"/model_"$id_number"
    sbatch -J "$id_number${script%.py}" --output=./Res/"${script%.py}"/model_"$id_number"/output job_Mare5.slurm "$script" "$line"
  fi
done