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

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    # alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

device = torch.device('cuda')

def do_aug(inputs,labels,aug_method,get_embed_fn,model=None,num_aug=1,epda_engine="EDA"):
    if aug_method == 'EDA':
        aug_fn = eda_4
    elif aug_method == 'EEDA':
        aug_fn = eda
    elif aug_method == 'EPDA':
        aug_fn = epda_bert
    # print(len(inputs),'vs',len(labels))
    Xs,Ys= [],[]
    for i in range(len(inputs)):
        if aug_method == 'EPDA':
            translator = None
            if epda_engine == 'CWE':
                nlp_auger = naw.ContextualWordEmbsAug(action='insert',device='cuda')
            elif epda_engine == 'BT':
                nlp_auger = naw.BackTranslationAug(device='cuda')
            else:
                nlp_auger = None
            # if labels[i]==0 or labels[i]==2:
            #     # print("Dont aug")
            #     augedtxts = [inputs[i]]
            # else:
            MIX_UP = False # TQ: TODO!
            augedtxts,_ = aug_fn(txt=inputs[i],label=labels[i],num_aug=num_aug,model=model,translator=translator,
                engine=epda_engine,alpha=args.alpha_da,mix_up=MIX_UP,get_embed_fn=get_embed_fn,loss_fn=nn.CrossEntropyLoss(),nlp_auger = nlp_auger)
            Xs+=augedtxts
            # append with the same labels
            for j in range(len(augedtxts)):
                Ys.append(labels[i])
        else:
            txts = aug_fn(inputs[i],num_aug=num_aug)
            for txt in txts:
                embed = get_embed_fn(txt)
                # print("Size",embed.size())
                Xs.append(embed)
                label_tensor = torch.zeros(1)
                label_tensor[0] = labels[i]
                label_tensor = label_tensor.long()
                Ys.append(label_tensor)
    return Xs,Ys

def train(train_data_loader,test_data_loader,model,args,TOKENIZER,ema_model=None):
    EPOCHES = args.basic_epoch

    EPOCHES += args.da_epoch
    if args.method == "baseline" and args.da_epoch > 0:
        print("warning, da_epoch should be 0 for baseline models.")
    print("Update EPOCHES to",EPOCHES)
    
    input_dir = train_data_loader.dataset.dir

    devices = [i % args.n_gpu for i in range(args.n_model)]

    max_f1_score = 0.0
    trained_iter = 0
    # UPDATED = False

    optimizer = AdamW(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=1e-3)
    # optimizer = Adam(model.parameters(), lr=LR, eps=1e-8, weight_decay=1e-3)

    # reset the dataset.
    train_data_loader.dataset.reset()


    IS_AUG_TRAINING = False
    for epoch in tqdm(range(EPOCHES)):
        model.train()
        if args.method not in ["baseline", "baseline_NLL"] and epoch % 5==0 and epoch>=args.basic_epoch:
            # need augmentation 
            print(f"Do data augmentation at epoch {epoch}")
            
            lines = open(input_dir,'r').readlines()
            Xs,Ys=[],[]
            count = [0]*args.num_classes
            for line in lines:
                y,x = line.split('\t')
                y = int(y)
                # if count[y] >= int(434*int(data_split)*2):
                #     continue
                count[y] += 1
                x = x[:-1]
                x = get_only_chars(x)
                Xs.append(x)
                Ys.append(y)
            if args.method in ["EDA_NLL", "EDA", "EDA_EMA", "EDA_NLL_EMA"]:
                inputs, labels = eda_nll(Xs, Ys, args)
            elif "EPDA" in args.method: # EPDA, EPDA_NLL, EPDA_EMA
                inputs,labels = do_aug(Xs,Ys,"EPDA",get_embed_fn=TOKENIZER,model=model,num_aug=args.num_aug)

            print('Before',len(train_data_loader))
            if args.train_trick == "mix_training":
                train_data_loader.dataset.reset()
                train_data_loader.dataset.update(Xs + inputs,train_data_loader.dataset.O_Ys + labels)
            elif args.train_trick == "label_consistency":
                aug_dataset = ConsDataset(Xs, inputs, labels, args.num_aug)
                train_data_loader = DataLoader(dataset=aug_dataset,batch_size=args.batch_size,collate_fn=Collect_FN_cons(TOKENIZER),shuffle=True)
            else:    
                train_data_loader.dataset.update(inputs,labels)
            print("< Update Done.")
            print('After',len(train_data_loader))
            # UPDATED = True
            # finish training on augmented data

            #     optimizer = AdamW(model.parameters(), lr=LR, eps=1e-8, weight_decay=1e-3)

            IS_AUG_TRAINING = True
            if ema_model is not None:
                copy_params(model, ema_model, devices)
        for i,batch in enumerate(train_data_loader):
            if IS_AUG_TRAINING and args.train_trick == "label_consistency":
                batch_ori, batch = (batch[0], batch[1]) # train with augmented data. The original data is batch_ori
            batch = move_to_device(batch)
            optimizer.zero_grad()
            
            if args.method in ["EDA_NLL", "EPDA_NLL", "EDA_NLL_EMA"]: # NLL model
                output = model(no_nll=not IS_AUG_TRAINING, **batch)
            elif args.method in ["baseline_NLL"] and epoch >= args.basic_epoch: # baseline + NLL
                output = model(no_nll=False, **batch)
            else:
                output = model(**batch)
            
            loss = output[0]
            logits = output[1] # for TextClassification model, it's the logit. For NLL model, it's a list of two logits.

            if IS_AUG_TRAINING and args.train_trick == "label_consistency": 
                if isinstance(model, NLLModel):
                    ori_logits = [model.models[i](**move_to_device(batch_ori, devices[i]))[-1] for i in range(len(model.models))] # logit of model 1 and 2
                    for i in range(len(logits)):
                        probs = [F.softmax(logits[i], dim=-1), F.softmax(ori_logits[i].to(0), dim=-1)]
                        mask = (batch["labels"].view(-1) != -1).to(logits[i])
                        cons_loss = kl_div(probs[0], probs[1]) * mask
                        cons_loss = cons_loss.sum() / (mask.sum() + 1e-3)
                        loss += args.alpha_c * cons_loss
                else:
                    batch_ori = move_to_device(batch_ori)
                    ori_logit = model(**batch_ori)[-1]
                    probs = [F.softmax(logits, dim=-1), F.softmax(ori_logit.to(0), dim=-1)]
                    mask = (batch["labels"].view(-1) != -1).to(logits)
                    cons_loss = kl_div(probs[0], probs[1]) * mask
                    cons_loss = cons_loss.sum() / (mask.sum() + 1e-3)
                    loss += args.alpha_c * cons_loss
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            trained_iter += 1

            if IS_AUG_TRAINING and "EMA" in args.method: # keep an exponential moving average
                update_ema_variables(model, ema_model, args.ema_decay, trained_iter)


            del loss, batch

            if trained_iter % args.eval_step == 0:

                model.eval()
                pred_y,gt_y=[],[]
                for i,batch in enumerate(test_data_loader):

                    label = batch['labels']
                    del batch['labels']
                    batch = move_to_device(batch)

                    if IS_AUG_TRAINING and args.method in ["EDA_NLL_EMA"]:
                        outputs = ema_model(**batch)[-1][0]
                    else:
                        outputs = model(**batch)[-1][0]
                    # outputs = model(**batch).logits
                    b,_ = outputs.size()
                    outputs = torch.softmax(outputs,1)
                    # confidence_mat = torch.ones(outputs.size())

                    outputs = torch.argmax(outputs,1).detach().cpu()
                    for j in range(b):
                        pred_y.append(outputs[j])
                        gt_y.append(label[j])
                        # print(outputs[j],'vs',label[j])

                score = f1_score(gt_y, pred_y, average='macro')
                if score > max_f1_score:
                    save_dir = os.path.join(args.save_path, args.dataset + args.data_split, args.method + args.save_special_name + ".pt")
                    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
                    torch.save(model, save_dir)
                max_f1_score = max(max_f1_score,score)

        # os._exit(233)
    return max_f1_score

def copy_params(model, ema_model, devices):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data = param.data

def compute_single_model(args):
    f1_scores = []
    train_dataset = EPDADataSet(args.train_dir,num_classes=args.num_classes)
    test_dataset = EPDADataSet(args.test_dir,num_classes=args.num_classes)
    try:
        TOKENIZER = AutoTokenizer.from_pretrained(args.model_name_or_path,local_files_only=True)
    except:
        TOKENIZER = AutoTokenizer.from_pretrained(args.tokenizer_name,local_files_only=True)

    collate_fn = Collect_FN(TOKENIZER, True)
    # train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_dataset.weights, BATCH_SIZE)
    # print(max(train_dataset.weights),min(train_dataset.weights))
    train_data_loader = DataLoader(dataset=train_dataset,batch_size=args.batch_size,collate_fn=collate_fn,shuffle=True)
    test_data_loader = DataLoader(dataset=test_dataset,batch_size=args.test_batch_size,shuffle=False,collate_fn=collate_fn)
    # Test them for 5 times
    
    setup_seed(0)
    ema_model = None
    if  args.method in ["baseline_NLL", "EDA_NLL", "EPDA_NLL"]:
        model = NLLModel(args) 
    elif args.method in ["EDA_NLL_EMA"]:
        model = NLLModel(args)
        ema_model = NLLModel(args, ema=True)
    else: # baselines, EDA, EPDA
        model = TextClassificationModel(args) 
        model.to("cuda")


    f1 = train(train_data_loader,test_data_loader,model,args,TOKENIZER,ema_model=ema_model)
    f1_scores.append(f1)
    print("Current F1 Score",f1,"Average F1 Score: ",sum(f1_scores)/len(f1_scores), flush = True)
    print("> Done.", flush = True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # file path
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--dataset", default="irony", type=str)
    parser.add_argument("--data_split", default="10", type=str)    
    # data_split = '40'
    # data_split = 'full'
    
    # transformers
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str)
    parser.add_argument("--tokenizer_name", default="bert-base-uncased", type=str)
    parser.add_argument("--max_seq_length", default=512, type=int)

    # training
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--test_batch_size", default=32, type=int)
    parser.add_argument("--eval_step", default=1, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--eps", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--warmup_ratio", default=0.1, type=float)
    parser.add_argument("--dropout_prob", default=0.1, type=float)
    parser.add_argument("--basic_epoch", default=20, type=int)
    parser.add_argument("--da_epoch", default=20, type=int)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n_model", type=int, default=2)
    parser.add_argument("--alpha_t", type=float, default=50.0) # CR weight
    parser.add_argument("--alpha_c", type=float, default=50.0) # training consistency
    parser.add_argument("--alpha_warmup_ratio", default=0.1, type=float)

    # Data Augmentaion related:
    parser.add_argument("--train_trick", default="", choices=["mix_training", "label_consistency", ""],
        help="mix_training: mix the original training set and the augmented set together.")
    parser.add_argument("--num_aug", default=3, type=float)
    parser.add_argument('--mix_up', action='store_true', ) # default is false
    parser.add_argument("--method", default="EDA_NLL", 
                        choices=["baseline", "baseline_NLL", "EDA", "EPDA", "EDA_NLL", "EPDA_NLL", "EDA_NLL_EMA"])
    parser.add_argument("--epda_engine", default="EDA")
    parser.add_argument("--alpha_da", default=0.05, type=float)
    parser.add_argument("--ema_decay", default=0.999, type=float)

    # IO options
    parser.add_argument('--save_model', action='store_true', )  # whether to save the model
    parser.add_argument('--save_path', default='models') 
    parser.add_argument('--save_special_name', default="")

    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()
    os.makedirs(args.save_path, exist_ok=True)

    args.train_dir = os.path.join(args.data_dir, args.dataset, 'train_'+str(args.data_split)+'.txt')
    args.test_dir = os.path.join(args.data_dir, args.dataset, 'test.txt')

    if 'irony' in args.train_dir:
        args.num_classes = 2
    elif 'agnews' in args.train_dir:
        args.num_classes = 4
    elif 'trec' in args.train_dir:
        args.num_classes = 6
    elif 'sentiment' in args.train_dir:
        args.num_classes = 3

    compute_single_model(args)