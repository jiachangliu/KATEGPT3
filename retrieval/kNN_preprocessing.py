import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import datasets
from matplotlib import pyplot as plt
from tqdm import tqdm
import sys
from transformers import RobertaTokenizer, RobertaModel
from transformers import AutoTokenizer, AutoModel
from transformers import BartTokenizer, BartModel
import torch
import os
from utils import chunks
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise
import pickle
import re
from sentence_transformers import SentenceTransformer

#########################################################################################################################################
print("Is CUDA available? ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--train_fname', default='wmt16-en-de', type=str)
parser.add_argument('--dev_fname', default='wmt16-en-de', type=str)
parser.add_argument('--encoder_name', default='bart-large', type=str,
                   help='roberta-base, roberta-large, bart-base, bart-large, \
                   bert-base-nli-stsb-mean-tokens, bert-large-nli-stsb-mean-tokens, \
                   roberta-base-nli-stsb-mean-tokens, roberta-large-nli-mean-tokens, roberta-large-nli-stsb-mean-tokens')
parser.add_argument('--metric', default='euclidean', type=str, help='euclidean or cosine')
parser.add_argument('--embed_type', default='CLS', type=str, help='CLS or mean')
parser.add_argument('--num_neighbors', default=100, type=int, help='10, 50, or 100')
parser.add_argument('--Q', default='sentence', help='sentence, source, table, text, q')
parser.add_argument('--A', default='label', help='label, target, sentence, code, a')
parser.add_argument('--reversed', action='store_true')
args = parser.parse_args()

Q = args.Q
A = args.A

print("The training dataset is {}".format(args.train_fname))
print("The dev dataset is {}".format(args.dev_fname))
print("The encoder to get {} {} embeddings is {}".format(args.embed_type, args.metric, args.encoder_name))


train_fname = "../inference/dataset/{}_train.tsv".format(args.train_fname)
dev_fname = "../inference/dataset/{}_dev.tsv".format(args.dev_fname)

HF_cache_dir = "/Research/huggingface/transformers/cached_transformers/"
if args.encoder_name == "tfidf":
    pass
elif args.encoder_name == "roberta-base-SST-2":
    tok = RobertaTokenizer.from_pretrained("you_roberta-base_SST-2_folder")
    model = RobertaModel.from_pretrained("you_roberta-base_SST-2_folder/checkpoint-8420")
elif args.encoder_name == "roberta-large-SST-2":
    tok = RobertaTokenizer.from_pretrained("you_roberta-large_SST-2_folder")
    model = RobertaModel.from_pretrained("you_roberta-base_SST-2_folder/checkpoint-6315")
elif args.encoder_name == "roberta-base" or args.encoder_name == "roberta-large":
    tok = RobertaTokenizer.from_pretrained(args.encoder_name, cache_dir=HF_cache_dir)
    model = RobertaModel.from_pretrained(args.encoder_name, cache_dir=HF_cache_dir)
elif args.encoder_name == "bart-base" or args.encoder_name == "bart-large":
    tok = BartTokenizer.from_pretrained("facebook/{}".format(args.encoder_name), cache_dir=HF_cache_dir)
    model = BartModel.from_pretrained("facebook/{}".format(args.encoder_name), cache_dir=HF_cache_dir)
else:
#     tok = AutoTokenizer.from_pretrained("{}{}/0_Transformer".format(HF_cache_dir, args.encoder_name))
#     model = AutoModel.from_pretrained("{}{}/0_Transformer".format(HF_cache_dir, args.encoder_name))
    tok = None
    model = SentenceTransformer("{}{}".format(HF_cache_dir, args.encoder_name))

metric = args.metric # "cosine" # "euclidean"

#########################################################################################################################################

sync_from_GDrive = "rclone sync -P remote:PhD_Research/GPT-3/{} dataset/".format(train_fname)
print(sync_from_GDrive)
os.system(sync_from_GDrive)
sync_from_GDrive = "rclone sync -P remote:PhD_Research/GPT-3/{} dataset/".format(dev_fname)
print(sync_from_GDrive)
os.system(sync_from_GDrive)

# re separator: (?<![\\t].)\\t
train_df = pd.read_csv(train_fname, sep='(?<![\\t].)\\t', quotechar='"', engine='python', header='infer', keep_default_na=False)
train_corpus = train_df.loc[:, Q].to_list()
train_labels = train_df.loc[:, A].to_list()

train_indices = list(range(len(train_corpus)))
# train_indices = list(range(2000))

train_corpus = [train_corpus[train_index] for train_index in train_indices]
train_labels = [train_labels[train_index] for train_index in train_indices]
for k in range(len(train_labels)):
    if train_labels[k] == 0:
        train_labels[k] = 2
    else:
        train_labels[k] = 3

# (?<!")\\t|(?<=")\\t(?=")
dev_df = pd.read_csv(dev_fname, sep='(?<![\\t].)\\t|\\t(?!\\")', quotechar='"', engine='python', header='infer', keep_default_na=False)
dev_corpus = dev_df.loc[:, Q].to_list()
dev_labels = dev_df.loc[:, A].to_list()

task_name = args.dev_fname
#########################################################################################################################################
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def decode(tok, model, corpus):
    embeddings = []
    
    if args.encoder_name == 'roberta-base' or args.encoder_name == 'roberta-large':
        print("Using non Sentence Transformer models")
        for corpus_tmp in tqdm(chunks(corpus, 32)):
            encoding = tok.batch_encode_plus(corpus_tmp, padding=True, truncation=True)
            sentence_batch, attn_mask = encoding["input_ids"], encoding["attention_mask"]
            sentence_batch, attn_mask = torch.LongTensor(sentence_batch).to(device), torch.LongTensor(attn_mask).to(device)

            with torch.no_grad():
                embedding_output_batch = model(sentence_batch, attn_mask)            
                if args.embed_type == 'mean':
                    sentence_embeddings = mean_pooling( embedding_output_batch, attn_mask)
                elif args.embed_type == 'CLS':
                    sentence_embeddings = embedding_output_batch[0][:, 0, :]
            embeddings.append(sentence_embeddings.detach().cpu().numpy())

    #         embedding_output_batch = model(sentence_batch, attn_mask)
    #         embeddings.append(embedding_output_batch[0][:, 0, :].detach().cpu())
            del sentence_batch, attn_mask, embedding_output_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        print("Using Sentence Transformer models")
        for corpus_tmp in tqdm(chunks(corpus, 32)):
            sentence_embeddings = model.encode(corpus_tmp)
            embeddings.append(sentence_embeddings)
    
    return np.concatenate(embeddings, axis=0)

#########################################################################################################################################

labels = np.asarray(dev_labels + train_labels)
unique_labels = list(set(labels))
dev_indices = [[] for _ in unique_labels]
for i, label in enumerate(labels):
    for j, unique_label in enumerate(unique_labels):
        if label == unique_label:
            dev_indices[j].append(i)
            
n_dev = len(dev_labels)
n_train = len(train_indices)

corpus = dev_corpus + train_corpus

if args.encoder_name == "bm25":
    pass
else: # deep learning model
    model.to(device)
    X = decode(tok, model, corpus)
emb_train = X[n_dev:]
emb_dev = X[:n_dev]

if metric == "euclidean":
    nbrs = NearestNeighbors(n_neighbors=args.num_neighbors, algorithm='ball_tree', n_jobs=-1).fit(emb_train)
    distances, indices = nbrs.kneighbors(emb_dev)
elif metric == "cosine":
    dist_matrix = pairwise.cosine_similarity(X=emb_dev, Y=emb_train)
    if args.reversed:
        values, indices = torch.topk(-torch.from_numpy(dist_matrix), k=args.num_neighbors, dim=-1)
    else:
        values, indices = torch.topk(torch.from_numpy(dist_matrix), k=args.num_neighbors, dim=-1)
    indices = indices.numpy()

train_indices_np = np.asarray(train_indices)
kNN_dev_train = [train_indices_np[indices[i]].reshape(1, -1) for i in range(len(indices))]
kNN_dev_train = np.concatenate(kNN_dev_train, axis=0)
print(kNN_dev_train.shape)

if args.reversed:
    PIK = "../inference/kNN_pretraining/" + task_name + "_{}_{}_{}_reversed.dat".format(args.encoder_name, args.metric, args.embed_type)
else:
    PIK = "../inference/kNN_pretraining/" + task_name + "_{}_{}_{}.dat".format(args.encoder_name, args.metric, args.embed_type)

data = dict()
data["kNN_dev_train"] = kNN_dev_train

with open(PIK, "wb") as f:
    pickle.dump(data, f)

sync_to_GDrive = "rclone sync -P {} remote:PhD_Research/GPT-3/kNN_pretraining/".format(PIK)
print(sync_to_GDrive)
os.system(sync_to_GDrive)
print("Finish kNN preprocessing!")
