# EPiDA Easy Plug-in Data Augumentation
import argparse
import os
import numpy as np
import re
import random
from tqdm import tqdm
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertModel, XLNetTokenizer, AutoTokenizer, AutoModel
from transformers import AdamW, BertForSequenceClassification,XLNetForSequenceClassification,AutoModelForSequenceClassification

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score,f1_score,classification_report
# from utils import SoftCrossEntropy,FocalLoss

from nlp_aug import eda_4
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from eda import eda,epda,epda_bert, eda_nll
from utils import move_to_device, setup_seed
from model_nll import *
from dataset import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # file path
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--dataset", default="irony", type=str)
    parser.add_argument("--data_split", default="10", type=str)    
    parser.add_argument("--seed", default=100, type=int) 
    

    # Data Augmentaion related:
    parser.add_argument("--num_aug", default=3, type=float)
    parser.add_argument("--method", default="EDA", choices=["EDA", "EPDA"])
    parser.add_argument("--epda_engine", default="EDA")
    parser.add_argument("--alpha_da", default=0.05, type=float)

    # IO options
    parser.add_argument('--save_path', default='aug_data') 
    parser.add_argument('--save_special_name', default="")

    args = parser.parse_args()

    setup_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)

    args.train_dir = os.path.join(args.data_dir, args.dataset, 'train_'+str(args.data_split)+'.txt')

    if 'irony' in args.train_dir:
        args.num_classes = 2
    elif 'agnews' in args.train_dir:
        args.num_classes = 4
    elif 'trec' in args.train_dir:
        args.num_classes = 6
    elif 'sentiment' in args.train_dir:
        args.num_classes = 3
    
    save_dir = os.path.join(args.save_path, args.dataset + args.data_split, 
        args.method + "_" + str(args.seed) + args.save_special_name + ".txt")
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    
    # train_dataset = EPDADataSet(,num_classes=args.num_classes)

    lines = open(args.train_dir, 'r').readlines()
    Xs, Ys=[], []

    for line in lines:
        y,x = line.split('\t')
        y = int(y)

        x = x[:-1]
        x = get_only_chars(x)
        Xs.append(x)
        Ys.append(y)
    if args.method in ["EDA"]:
        inputs, labels = eda_nll(Xs, Ys, args)
    with open(save_dir, "w") as writer:
        writer.writelines(["\t".join([str(l), t])+"\n" for t, l in zip(inputs,labels)])
