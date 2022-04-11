#!/bin/bash

export task_name=trivia_qa_train_78785_dev_full

kNN_fnames=(roberta-large-nli-mean-tokens)

CUDA_VISIBLE_DEVICES=0 python ./kNN_preprocessing.py --train_fname ${task_name} --dev_fname ${task_name} --Q q --A a --encoder_name tfidf --metric cosine --embed_type mean --num_neighbors 100

### retrieve using roberta-large model
# CUDA_VISIBLE_DEVICES=1 python ./kNN_preprocessing.py --train_fname ToTTo --dev_fname ToTTo --Q table --A sentence --encoder_name roberta-large --metric euclidean --embed_type CLS --num_neighbors 100

### retrieve using fine-tuned roberta-large models
# kNN_fnames=(roberta-large-nli-mean-tokens roberta-large-stsb-mean-tokens)
# for kNN_fname in ${kNN_fnames[@]}; do
#     CUDA_VISIBLE_DEVICES=0 python ./kNN_preprocessing.py --train_fname ${task_name} --dev_fname ${task_name} --Q q --A a --encoder_name $kNN_fname --metric cosine --embed_type mean --num_neighbors 100
# done