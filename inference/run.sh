#!/bin/bash


export task_name=trivia_qa_train_78785_dev_full
kNN_fnames=(random_0 roberta-large-nli-stsb-mean-tokens_cosine_mean)

mkdir result/
mkdir result/${task_name}/

# GPT-3 random vs KATE
for kNN_fname in ${kNN_fnames[@]}; do
    export PIK_name=full_knn_${kNN_fname}
    export kNN_dev_train=${task_name}_${kNN_fname}
    python3 autocomplete.py --task_type generation --category_name QA --task_name ${task_name} --train_name ${task_name} --epochs 1 --evaluate_dev --knn_num 10 --max_tokens 10 --PIK_name $PIK_name --kNN_dev_train $kNN_dev_train
done