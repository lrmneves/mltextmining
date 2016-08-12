#!/bin/bash
input="result_file.txt"
while IFS= read -r var
do
  echo "$var"
  python eval/eval_rmse.py eval/dev.golden $var
done < "$input"



# python eval/eval_rmse.py eval/dev.golden 