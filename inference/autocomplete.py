import os
import sys
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
import pickle
import argparse
import sacrebleu
from utils import categories, templates, chunks, constructPrompt, cleanLabel, most_common

import openai
openai.api_key = os.getenv("GPT3_KEY")

parser = argparse.ArgumentParser()
parser.add_argument('--task_type', default='classification', type=str,
                   help="classification or generation")
parser.add_argument('--task_name', default='SST-2', type=str)
parser.add_argument('--train_name', default='', type=str)
parser.add_argument('--category_name', default='', type=str)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--bz_train', default=1, type=int)
parser.add_argument('--knn_num', default=1, type=int)
parser.add_argument('--bz_dev', default=1, type=int)
parser.add_argument('--max_tokens', default=10, type=int)
parser.add_argument('--digits', action='store_true')
parser.add_argument('--truncate', action='store_true')
parser.add_argument('--kNN_dev_train', default='', type=str)
parser.add_argument('--evaluate_train', action='store_true')
parser.add_argument('--evaluate_train_indices', nargs='+', default=[-1], type=int)
parser.add_argument('--evaluate_dev', action='store_true')
parser.add_argument('--PIK_name', default="tmp", type=str)
args = parser.parse_args()

##############################################################
# GPT-3 completion class
class glueComplete:
    def __init__(self):
        return
    def __call__(self, example="", max_tokens=5):
        prompt = example
        response = openai.Completion.create(engine="davinci-msft", prompt=prompt, stop='\n', max_tokens=max_tokens, temperature=0.0, logprobs=1, n=1)
        resp = []
        for c in response.choices:
            text = c.text
            resp.append(text.strip())
        return resp

##############################################################
task = glueComplete()

##############################################################
# experiment parameters
task_name = args.task_name
train_name = args.train_name
category_name = args.category_name
epochs = args.epochs # EPOCHS
bz_train = args.bz_train # N
bz_dev = args.bz_dev # M
max_tokens = args.max_tokens

##############################################################
# read files
train_fname = "dataset/" + train_name + "_train.tsv"
dev_fname = "dataset/" + task_name + "_dev.tsv"
train_df = pd.read_csv(train_fname, sep='(?<![\\t].)\\t', engine='python', header='infer', keep_default_na=False)
dev_df = pd.read_csv(dev_fname, sep='(?<![\\t].)\\t', engine='python', header='infer', keep_default_na=False)
print(len(dev_df))
##############################################################

train_indices = range(len(train_df))
train_labels = train_df.loc[:, categories[category_name]["A"][0].lower()].to_list()
dev_labels = dev_df.loc[:, categories[category_name]["A"][0].lower()].to_list()

if args.task_type == "classification":
    dev_unique_labels = list(set(dev_labels))
    dev_unique_labels.sort()
    dev_indices = [[] for _ in dev_unique_labels]
    for j, x in enumerate(dev_labels):
        for i, y in enumerate(dev_unique_labels):
            if x == y:
                dev_indices[i].append(j)
elif args.task_type == "generation":
    dev_unique_labels = [0]
    dev_indices = [list(range(len(dev_df)))]

track_train = [] # EPOCHS x N, indices for train set samples
track_dev = [[] for _ in dev_unique_labels] # D x M, indices for dev set samples
pred_dev = [[] for _ in dev_unique_labels] # D x EPOCHS x M, prediction for dev set samples
accuracy_dev = [[] for _ in dev_unique_labels] # EPOCHS x D, accuracy for dev set samples

# evaluate on the entire dev set or random samples for each epoch
if args.evaluate_dev:
    for i in range(len(dev_unique_labels)):
        track_dev[i].append(dev_indices[i])
else:
    random.seed(30)
    for i in range(len(dev_unique_labels)):
        tmp_dev_indices = random.sample(dev_indices[i], k=bz_dev)
        track_dev[i].append(tmp_dev_indices)

# prompt construction using kNN or random samples or fixed samples
evaluate_train_indices = []
if args.kNN_dev_train:
    PIK_kNN = "kNN_pretraining/" + args.kNN_dev_train + ".dat"
    with open(PIK_kNN, "rb") as f:
        kNN_data = pickle.load(f)
        kNN_dev_train = kNN_data["kNN_dev_train"]
    print("Evaluate on dev set using KNN on train set")
if args.evaluate_train:
    evaluate_train_indices = args.evaluate_train_indices
    print("Evaluate on training indices:", evaluate_train_indices)
##############################################################
# actual GPT-3 few-shot learning

Q_list = categories[category_name]["Qs"]
A_list = categories[category_name]["A"]
templates = templates[category_name]

for _ in tqdm(range(epochs)):
    if args.evaluate_train:
        tmp_train_indices = evaluate_train_indices
    else:
        tmp_train_indices = random.sample(train_indices, k=bz_train)
    track_train.append(tmp_train_indices) # store the training indices during each iteration
    
    tmp_example = constructPrompt(df=train_df, labels=train_labels, indices=tmp_train_indices, templates=templates, Q_list=Q_list, A_list=A_list, A_flag=True, truncate=args.truncate)
    
    for i in range(len(dev_unique_labels)):
        tmp_dev_indices = track_dev[i][0]
        
        counts = 0
        pred_dev[i].append([])
        for dev_indices in tqdm(chunks(tmp_dev_indices, n=20)):
            prompt_example = []
            for l in range(len(dev_indices)):
                if args.kNN_dev_train:
                    tmp_example = constructPrompt(df=train_df, labels=train_labels, indices=kNN_dev_train[dev_indices[l]][:args.knn_num], templates=templates, Q_list=Q_list, A_list=A_list, A_flag=True, truncate=args.truncate)
                
                prompt_example.append(tmp_example)
                dev_prompt = constructPrompt(df=dev_df, labels=dev_labels, indices=[dev_indices[l]], templates=templates, Q_list=Q_list, A_list=A_list, A_flag=False, truncate=args.truncate)
                prompt_example[l] += dev_prompt

            tmp_pred = task(example=prompt_example, max_tokens=max_tokens)
            pred_dev[i][-1] += tmp_pred
        
        # evaluation
        if args.task_type == "classification":
            counts = pred_dev[i][-1].count(dev_unique_labels[i])
            accuracy_dev[i].append(counts/bz_dev)
        elif args.task_type == "generation":
            reference_dev = dev_df.loc[tmp_dev_indices, categories[category_name]["A"][0].lower()].to_list()
            print(len(pred_dev[i][-1]), len(reference_dev))
            counts = sacrebleu.corpus_bleu(pred_dev[i][-1], [reference_dev])
            accuracy_dev[i] = counts.score

##############################################################
# summarize and save into file
print(accuracy_dev)
PIK = "result/" + task_name + "/" + args.PIK_name + ".dat"

data = dict()
data["task"] = task_name
data["bz_dev"] = bz_dev
data["bz_train"] = bz_train
data["epochs"] = epochs
data["max_tokens"] = max_tokens
data["evaluate_train"] = args.evaluate_train
data["evaluate_dev"] = args.evaluate_dev
data["track_train"] = track_train
data["track_dev"] = track_dev
data["pred_dev"] = pred_dev
data["accuracy_dev"] = accuracy_dev

with open(PIK, "wb") as f:
    pickle.dump(data, f)
