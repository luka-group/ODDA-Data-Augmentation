import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from itertools import chain
from utils import get_only_chars
import pdb

import re
import numpy as np



class Collect_FN():
    def __init__(self, TOKENIZER, with_label_hard, with_GT_labels = False):
        super(Collect_FN, self).__init__()
        self.tokenizer = TOKENIZER
        self.with_label = with_label_hard
        self.with_GT_labels = with_GT_labels

    def __call__(self, batchs):
        # print(batchs)
        if (self.with_label and self.with_GT_labels == False):
            sentences, labels = map(list, zip(*batchs))
        elif (self.with_label == True and self.with_GT_labels == True):
            sentences, labels, GT_labels = map(list, zip(*batchs))
        else:
            sentences = batchs
        encoding = self.tokenizer(sentences, return_tensors = 'pt', padding = True, truncation = True)
        # input_ids = encoding['input_ids']
        # attention_mask = encoding['attention_mask']
        # ans = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if (self.with_label):
            labels = torch.tensor(labels).long()
            encoding['labels'] = labels
        if (self.with_GT_labels):
            GT_labels = torch.tensor(GT_labels).long()
            encoding['GT_labels'] = GT_labels
        # encoding['sentences'] = sentences
        del encoding["token_type_ids"]
        return encoding

class Collect_FN_cons(): # for training consistency loss
    def __init__(self, TOKENIZER):
        super(Collect_FN_cons, self).__init__()
        self.tokenizer = TOKENIZER

    def __call__(self, batchs):
        # print(batchs)
        
        sentences, sentences_tild, labels = map(list, zip(*batchs))

        encoding = self.tokenizer(sentences, return_tensors = 'pt', padding = True, truncation = True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        results = {
                   'input_ids': input_ids,
                   'attention_mask': attention_mask,
                   'labels':torch.tensor(labels).long(),
                   }

        encoding_tild = self.tokenizer(sentences_tild, return_tensors = 'pt', padding = True, truncation = True)
        results_tild = {
                   'input_ids': encoding_tild["input_ids"],
                   'attention_mask': encoding_tild['attention_mask'],
                   'labels':torch.tensor(labels).long(),
                   }

        return results, results_tild

class Collect_FN_BC(): # for training consistency loss
    def __init__(self, TOKENIZER):
        super(Collect_FN_BC, self).__init__()
        self.tokenizer = TOKENIZER

    def __call__(self, batchs):
        # print(batchs)
        
        sentences, is_aug, labels = map(list, zip(*batchs))

        encoding = self.tokenizer(sentences, return_tensors = 'pt', padding = True, truncation = True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        results = {
                   'input_ids': input_ids,
                   'attention_mask': attention_mask,
                   'labels':torch.tensor(labels).long(),
                   'is_aug':torch.tensor(is_aug).bool(),
        }

        return results

class Collect_FN_reweight(): # for training with reweighting
    def __init__(self, TOKENIZER):
        super(Collect_FN_reweight, self).__init__()
        self.tokenizer = TOKENIZER

    def __call__(self, batchs):
        # print(batchs)
        
        sentences, labels, sentences_tild, labels_tild = map(list, zip(*batchs))
        sentences_tild = list(chain(*sentences_tild))
        labels_tild = list(chain(*labels_tild))
        # (b_s)  , (b_s),  (b_s * num_aug), (b_s * num_aug)

        encoding = self.tokenizer(sentences, return_tensors = 'pt', padding = True, truncation = True)
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        results = {
                   'input_ids': input_ids,
                   'attention_mask': attention_mask,
                   'labels':torch.tensor(labels).long(),
                   }

        encoding_tild = self.tokenizer(sentences_tild, return_tensors = 'pt', padding = True, truncation = True)
        results_tild = {
                   'input_ids': encoding_tild["input_ids"],
                   'attention_mask': encoding_tild['attention_mask'],
                   'labels':torch.tensor(labels_tild).long(),
                   }

        return results, results_tild

class BCDataSet(Dataset):
    # back ground class dataset
    def __init__(self,original_data, augmented_data, labels, labels_tild):
        self.X = original_data + augmented_data
        self.is_aug = [0] * len(original_data) + [1] * len(augmented_data)
        self.labels = labels + labels_tild
    def __getitem__(self, idx):
        return self.X[idx], self.is_aug[idx], self.labels[idx]
    def __len__(self):
        return len(self.X)

class ConsDataset():
    # consistency dataset
    def __init__(self,original_data, augmented_data, labels_tild, num_aug):
        assert num_aug == len(augmented_data) / len(original_data)
        self.X = list(chain(*[[sent for i in range(num_aug)] for sent in original_data] ) )
        self.X_tild = augmented_data
        self.labels_tild = labels_tild
    def __getitem__(self, idx):
        return self.X[idx], self.X_tild[idx], self.labels_tild[idx]
    def __len__(self):
        return len(self.X)


class ReweightDataset():
    # Data for re-weighting
    # (x, labels, [x']x, labels_aug, num_aug )
    def __init__(self,original_data, labels, augmented_data, labels_tild, num_aug):
        assert num_aug == len(augmented_data) / len(original_data)
        self.num_aug = num_aug
        self.X = original_data
        self.X_tild = augmented_data
        self.labels = labels
        self.labels_tild = labels_tild
    def __getitem__(self, idx):
        return self.X[idx], self.labels[idx], self.X_tild[idx*self.num_aug:(idx+1)*self.num_aug], self.labels_tild[idx*self.num_aug:(idx+1)*self.num_aug]
    def __len__(self):
        return len(self.X)


class EPDADataSet(Dataset):
    def __init__(self,input_dir,max_len=30,num_classes=2):
        self.max_len = max_len
        self.num_classes = num_classes
        self.dir = input_dir
        print("Start to read: ",input_dir, flush = True)
        #先预读一下
        lines = open(input_dir,'r').readlines()
        Xs,Ys=[],[]
        count = [0] * num_classes
        for line in lines:
            y,x = line.split('\t')
            y = int(y)
            # 最后后一个\n的
            x = x[:-1]
            # if 'train' in input_dir:
            #     if y==0 or y==2:
            #         continue
            # if count[y] >= int(434*int(data_split)/10*2) and 'train' in input_dir:
            #     continue
            count[y] += 1
            if len(x)<=2:
                continue
            x = get_only_chars(x)
            Xs.append(x)
            Ys.append(y)
        # weight_per_class = [0.] * num_classes
        # N = float(sum(count))                                                   
        # for i in range(num_classes):
        #     weight_per_class[i] = N/float(count[i])                     
        # weight = [0] * len(Ys)
        # for idx, val in enumerate(Ys):
        #     weight[idx] = weight_per_class[val]
        # self.weights = weight_per_class
        # print(weight_per_class,count)
        # os._exit(233)
        # if not 'test' in input_dir:
        #     Xs,Ys = self.upsample_balance(Xs,Ys)
        #     print("Balance dataset Over.")
        self.Xs = Xs
        self.Ys = Ys
        
        self.O_Xs = self.Xs
        self.O_Ys = self.Ys
        print("Load Over, Find: ",len(self.Xs)," datas.", flush = True)
    

    def __getitem__(self, idx):
        assert idx < len(self.Xs)      
        return self.Xs[idx],self.Ys[idx]
    def __len__(self):
        return len(self.Xs)

    def update(self,Xs,Ys):
        print("Start Update Dataset, Find ",len(self.Xs),'datas.', flush = True)
        # if not 'test' in self.dir:
        #     Xs,Ys = self.upsample_balance(Xs,Ys)
        #     print("Balance dataset Over.")
        self.Xs = Xs
        self.Ys = Ys
        print("Update Dataset Finish, Find ",len(self.Xs),'datas.', flush = True)
        return

    def reset(self):
        if self.O_Xs is not None:
            self.Xs = self.O_Xs
            self.Ys = self.O_Ys

    def upsample_balance(self, sentences, labels):
        sample_number_per_class = [0]*self.num_classes
        for y in labels:
            sample_number_per_class[y] +=1
        sample_number_per_class = np.array(sample_number_per_class)
        max_number = np.max(sample_number_per_class)
        fill_number_each_class = max_number - sample_number_per_class
        # print("??",sample_number_per_class,fill_number_each_class)
        sentence_each_class = [[] for i in range(self.num_classes)]
        for s, l in zip(sentences, labels):
            sentence_each_class[l].append(s)
        for class_index, (sentences_cur_class, fill_num_cur_class) in enumerate(
                zip(sentence_each_class, fill_number_each_class)):
            append_cur_class = []

            for i in range(fill_num_cur_class):
                append_cur_class.append(sentences_cur_class[i % len(sentences_cur_class)])
            sentence_each_class[class_index] = sentences_cur_class + append_cur_class
        ans_sentences = []
        ans_labels = []
        for class_index in range(self.num_classes):
            for s in sentence_each_class[class_index]:
                ans_sentences.append(s)
                ans_labels.append(class_index)
        return ans_sentences, ans_labels