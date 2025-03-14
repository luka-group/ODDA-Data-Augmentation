# EPiDA Easy Plug-in Data Augumentation
import argparse
import os
import numpy as np
import re
import random
from tqdm import tqdm
import wandb
from datetime import datetime, timedelta
import pdb

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
from eda import eda,epda,epda_bert, eda_nll, glitter, large_loss
from utils import move_to_device, setup_seed, get_only_chars
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
    elif aug_method == "glitter":
        aug_fn = glitter
    elif aug_method == "large_loss":
        aug_fun = large_loss
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
        elif aug_method == "glitter":
            augedtxts = glitter(inputs[i], labels[i], num_aug=num_aug, get_embed_fn=get_embed_fn, model=model)
            Xs+=augedtxts
            for j in range(len(augedtxts)):
                Ys.append(labels[i])
        elif aug_method == "large_loss":
            augedtxts = large_loss(inputs[i], labels[i], num_aug=num_aug, get_embed_fn=get_embed_fn, model=model)
            Xs+=augedtxts
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

def train(train_data_loader,test_data_loader,model,args,TOKENIZER,run_id=0,ema_model=None):
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
        if args.method not in ["baseline", "baseline_NLL"] and epoch % args.refresh_aug_epoch==0 and epoch>=args.basic_epoch:
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
            if "EDA" in args.method:
                inputs, labels = eda_nll(Xs, Ys, args)
            elif "EPDA" in args.method: # EPDA, EPDA_NLL, EPDA_EMA
                inputs,labels = do_aug(Xs,Ys,"EPDA",get_embed_fn=TOKENIZER,model=model,num_aug=args.num_aug)
            elif "glitter" in args.method: # Glitter for EDA
                inputs,labels = do_aug(Xs,Ys,"glitter",get_embed_fn=TOKENIZER,model=model,num_aug=args.num_aug)
            elif "large_loss" in args.method:
                inputs,labels = do_aug(Xs,Ys,"large_loss",get_embed_fn=TOKENIZER,model=model,num_aug=args.num_aug)
            elif "reweight" in args.method:
                inputs, labels = eda_nll(Xs, Ys, args)
            else:
                assert NotImplementedError


            if args.syn_noise:
                # add noise to the augmented data
                flip_idx = np.random.rand(len(labels)) < args.syn_noise_ratio
                get_new_idx = lambda x:list(range(0, x)) + list(range(x+1, args.num_classes))
                print("before adding noise", labels[:30])
                labels = [y if not flip_idx[i] else get_new_idx(y)[np.random.randint(args.num_classes-1)]  for i,y in enumerate(labels)]
                print("after adding noise", labels[:30])

            print('Before',len(train_data_loader))
            if args.method == "reweight":
                # no train_trick can be applied
                aug_dataset = ReweightDataset(Xs, Ys, inputs, labels, args.num_aug)
                train_data_loader = DataLoader(dataset=aug_dataset,batch_size=args.batch_size//(args.num_aug + 1),collate_fn=Collect_FN_reweight(TOKENIZER),shuffle=True) 
            else:
                if args.train_trick == "mix_training":
                    # train_data_loader.dataset.reset()
                    # train_data_loader.dataset.update(Xs + inputs,train_data_loader.dataset.O_Ys + labels)
                    aug_dataset = BCDataSet(Xs, inputs, Ys, labels)
                    train_data_loader = DataLoader(dataset=aug_dataset,batch_size=args.batch_size,collate_fn=Collect_FN_BC(TOKENIZER),shuffle=True)
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

            if args.method in ["EDA_self_teacher_v1", "EDA_self_teacher_v3", "EDA_self_teacher_v4", "EDA_self_teacher_v4_BC"] and IS_AUG_TRAINING == False:
                # initialize the teacher. Only once
                print("initializing teacher model.")
                teacher_model = TextClassificationModel(args)
                teacher_model.to(args.teacher_device)
                copy_params(model, teacher_model)
                teacher_model.eval()

            IS_AUG_TRAINING = True
            if ema_model is not None:
                copy_params(model, ema_model)
        for i,batch in enumerate(train_data_loader):
            if IS_AUG_TRAINING and args.train_trick == "label_consistency":
                batch_ori, batch = (batch[0], batch[1]) # train with augmented data. The original data is batch_ori
            if IS_AUG_TRAINING and args.method == "reweight":
                batch, batch_syn = (batch[0], batch[1])
            batch = move_to_device(batch)
            labels = batch["labels"]
            optimizer.zero_grad()

            if args.method in ["EDA_NLL", "EPDA_NLL", "EDA_NLL_EMA"]: # NLL model
                if "is_aug" in batch:
                    is_aug = batch.pop("is_aug")
                output = model(no_nll=not IS_AUG_TRAINING, **batch)
                loss = output[0]
                logits = output[1] # for TextClassification model, it's the logit. For NLL model, it's a list of two logits.
            elif args.method in ["EDA_EMA"]:
                if "is_aug" in batch:
                    is_aug = batch.pop("is_aug")
                output = model(**batch)
                loss = output[0]
                logits = output[1] # for TextClassification model, it's the logit. For NLL model, it's a list of two logits.
            elif args.method in ["EDA_NC", "EDA_NLL_NC"]:
                outputs = [] # outputs of different models in model_list
                labels = batch["labels"]
                if "is_aug" in batch:
                    is_aug = batch.pop("is_aug")
                for i in range(args.n_model):
                    output = model.models[i](**move_to_device(batch, i))
                    output = tuple([o.to(0) for o in output])
                    outputs.append(output)

                # the normal loss
                if IS_AUG_TRAINING:
                    null_labels = torch.tensor([args.num_classes-1]*len(labels)).to(labels)

                    labels_logit = F.one_hot(labels, num_classes=args.num_classes) * (1-args.bg_class_prior)\
                                     + F.one_hot(null_labels) * args.bg_class_prior

                    if args.train_trick == "mix_training":

                        labels_logit_ori = F.one_hot(labels, num_classes=args.num_classes)
                        labels_logit = labels_logit * is_aug.view(len(is_aug), -1) + labels_logit_ori * (~is_aug).view(len(is_aug), -1)
                        # print(is_aug)
                        # print(labels_logit)


                    loss_fn = lambda x,y:torch.mean(torch.sum(-y * F.log_softmax(x, dim=-1), -1))
                    loss = sum([loss_fn(output[1], labels_logit.to(0)) for k, output in enumerate(outputs)]) / args.n_model # average the loss
                    # loss = sum([nn.CrossEntropyLoss(output[1], labels_logit.to(k)) for k, output in enumerate(outputs)]) / args.n_model # average the loss

                    logits = [output[1] for output in outputs]
                    probs = [F.softmax(logit, dim=-1) for logit in logits]

                    avg_prob = torch.stack(probs, dim=0).mean(0)

                    mask = (labels.view(-1) != -1).to(logits[0])
                    reg_loss = sum([kl_div(avg_prob, prob) * mask for prob in probs]) / args.n_model
                    reg_loss = reg_loss.sum() / (mask.sum() + 1e-3)

                    loss = loss + args.alpha_t * reg_loss

                else:
                    loss = sum([output[0] for output in outputs]) / args.n_model
            elif args.method == "reweight":

                # train with batch ori
                output_ori = model(**batch)
                logits_ori = output_ori[1]
                loss = output_ori[0]

                if IS_AUG_TRAINING:
                    # calculate the loss of the num_aug synthetic data
                    labels_syn = F.one_hot(batch_syn["labels"], num_classes=args.num_classes).to(0)
                    batch_syn = move_to_device(batch_syn)
                    with torch.no_grad():
                        output_syn = model(**batch_syn) # (batch_size * num_aug, )
                        b_s = output_syn[1].shape[0] // args.num_aug
                        syn_loss = torch.sum(labels_syn * F.log_softmax(output_syn[1], dim=-1), -1).view(b_s, args.num_aug)
                        reweight_factor = F.softmax(syn_loss, dim=-1).view(b_s * args.num_aug)
                    output_syn = model(**batch_syn)
                    syn_logits = output_syn[1]
                    
                    loss_fn = lambda x,y,w:torch.mean(w * torch.sum(- y * F.log_softmax(x, dim=-1), -1))
                    loss_syn = loss_fn( syn_logits, 
                                    labels_syn,
                                    reweight_factor)
                    loss += loss_syn


            elif args.method in ["baseline_NLL"] and epoch >= args.basic_epoch: # baseline + NLL
                output = model(no_nll=False, **batch)
                loss = output[0]
                logits = output[1] # for TextClassification model, it's the logit. For NLL model, it's a list of two logits.
            elif args.method in ["EDA_NLL_self", "EDA_self_teacher_v1", "EDA_self_teacher_v3", "EDA_self_teacher_v4", "EDA_self_teacher_v4_BC"]:
                if "is_aug" in batch:
                    is_aug = batch.pop("is_aug")
                else:
                    is_aug = torch.ones(len(labels)).to(bool).to(0)
                output = [model(**batch) for i in range(args.n_model)] # outputs given two different dropouts

                if args.method in ["EDA_self_teacher_v1", "EDA_self_teacher_v3", "EDA_self_teacher_v4", "EDA_self_teacher_v4_BC"] and IS_AUG_TRAINING:
                    with torch.no_grad():
                        logits_teacher = teacher_model(**batch)[1]
                        logits_teacher = logits_teacher.to(0)
                    labels_logit = F.softmax(logits_teacher/args.teacher_temperature, dim=-1)
                    if args.method == "EDA_self_teacher_v4":
                        labels_logit_ori = F.one_hot(labels, num_classes=args.num_classes)
                        labels_logit = labels_logit * is_aug.view(len(is_aug), -1) + labels_logit_ori * (~is_aug).view(len(is_aug), -1)
                    elif args.method == "EDA_self_teacher_v4_BC":
                        
                        labels_logit_ori = F.one_hot(labels, num_classes=args.num_classes)
                        # label_logit: for ori, it's ground label. for aug, it's teacher's prediction.
                        labels_logit = labels_logit * is_aug.view(len(is_aug), -1) + labels_logit_ori * (~is_aug).view(len(is_aug), -1)

                        # add Background class here.
                        null_labels = torch.tensor([args.num_classes-1]*len(labels)).to(labels)
                        labels_logit = labels_logit * (1-args.bg_class_prior)\
                                 + F.one_hot(null_labels) * args.bg_class_prior # add prior to the background class
                else:
                    labels_logit = F.one_hot(labels, num_classes=args.num_classes)

                logits = [out[1] for out in output]
                
                loss_fn = lambda x,y:torch.mean(torch.sum(-y * F.log_softmax(x, dim=-1), -1))
                loss = sum([loss_fn(out[1], labels_logit.to(0)) for k, out in enumerate(output)]) / args.n_model # average the loss
                
                # loss = sum([out[0] for out in output]) / args.n_model # average the loss

                if IS_AUG_TRAINING:
                    

                    probs = [F.softmax(logit, dim=-1) for logit in logits]

                    avg_prob = torch.stack(probs, dim=0).mean(0)

                    mask = (labels.view(-1) != -1).to(logits[0])
                    reg_loss = sum([kl_div(avg_prob, prob) * mask for prob in probs]) / args.n_model
                    reg_loss = reg_loss.sum() / (mask.sum() + 1e-3)

                    if trained_iter % args.wandb_step == 0:
                        wandb.log({"CELoss":loss, "Regloss":reg_loss, f'step_{run_id}':trained_iter})

                    loss = loss + args.alpha_t * reg_loss # new loss
            elif args.method == "EDA_self_LS":
                if "is_aug" in batch:
                    is_aug = batch.pop("is_aug")
                output = [model(**batch) for i in range(args.n_model)] # outputs given two different dropouts

                

                if IS_AUG_TRAINING:
                    logits = [out[1] for out in output]

                    labels_logit_ori = F.one_hot(labels, num_classes=args.num_classes)
                    all_labels = [torch.tensor([i]*len(labels)).to(labels) for i in range(args.num_classes)]
                    labels_logit = F.one_hot(labels, num_classes=args.num_classes) * (1-args.label_smoothing)\
                        + sum([args.label_smoothing/args.num_classes * F.one_hot(l, num_classes=args.num_classes) for l in all_labels])

                    labels_logit = labels_logit * is_aug.view(len(is_aug), -1) + labels_logit_ori * (~is_aug).view(len(is_aug), -1)

                    probs = [F.softmax(logit, dim=-1) for logit in logits]

                    avg_prob = torch.stack(probs, dim=0).mean(0)

                    # avg_prob = (1-args.bg_class_prior) * avg_prob + args.bg_class_prior *  F.one_hot(null_labels) * args.bg_class_prior

                    mask = (labels.view(-1) != -1).to(logits[0])
                    reg_loss = sum([kl_div(avg_prob, prob) * mask for prob in probs[:len(logits)]]) / args.n_model
                    reg_loss = reg_loss.sum() / (mask.sum() + 1e-3)

                    # loss = sum([out[0] for out in output]) / args.n_model # average the loss

                    loss_fn = lambda x,y:torch.mean(torch.sum(-y * F.log_softmax(x, dim=-1), -1))
                    loss = sum([loss_fn(out[1], labels_logit.to(0)) for k, out in enumerate(output)]) / args.n_model # average the loss

                    loss = loss + args.alpha_t * reg_loss # new loss
                    if trained_iter % args.wandb_step == 0:
                        wandb.log({"CELoss":loss, "Regloss":reg_loss, f'step_{run_id}':trained_iter})

                    loss = loss + args.alpha_t * reg_loss # new loss

                else:
                    loss = sum([out[0] for out in output]) / args.n_model # average the loss
            elif args.method in ["EDA_self_BC"]:
                output = [model(**batch) for i in range(args.n_model)] # outputs given two different dropouts

                if IS_AUG_TRAINING:
                    logits = [out[1] for out in output]

                    null_labels = torch.tensor([args.num_classes-1]*len(labels)).to(labels)

                    if args.method in ["EDA_self_BC"]:
                        labels_logit = F.one_hot(labels, num_classes=args.num_classes) * (1-args.bg_class_prior)\
                                 + F.one_hot(null_labels) * args.bg_class_prior
                    else:
                        labels_logit = F.one_hot(labels, num_classes=args.num_classes)

                    probs = [F.softmax(logit, dim=-1) for logit in logits]

                    avg_prob = torch.stack(probs, dim=0).mean(0)

                    # avg_prob = (1-args.bg_class_prior) * avg_prob + args.bg_class_prior *  F.one_hot(null_labels) * args.bg_class_prior

                    mask = (labels.view(-1) != -1).to(logits[0])
                    reg_loss = sum([kl_div(avg_prob, prob) * mask for prob in probs[:len(logits)]]) / args.n_model
                    reg_loss = reg_loss.sum() / (mask.sum() + 1e-3)

                    # loss = sum([out[0] for out in output]) / args.n_model # average the loss

                    loss_fn = lambda x,y:torch.mean(torch.sum(-y * F.log_softmax(x, dim=-1), -1))
                    loss = sum([loss_fn(out[1], labels_logit.to(0)) for k, out in enumerate(output)]) / args.n_model # average the loss

                    loss = loss + args.alpha_t * reg_loss # new loss
                    if trained_iter % args.wandb_step == 0:
                        wandb.log({"CELoss":loss, "Regloss":reg_loss, f'step_{run_id}':trained_iter})

                    loss = loss + args.alpha_t * reg_loss # new loss

                else:
                    loss = sum([out[0] for out in output]) / args.n_model # average the loss

            else:
                if "is_aug" in batch:
                    is_aug = batch.pop("is_aug")
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

            if trained_iter % args.wandb_step == 0:
                wandb.log({f'loss': loss.item(), f'step_{run_id}':trained_iter})

            del loss, batch

            if trained_iter % args.eval_step == 0:

                model.eval()
                pred_y,gt_y=[],[]
                for i,batch in enumerate(test_data_loader):

                    label = batch['labels']
                    del batch['labels']
                    batch = move_to_device(batch)

                    if IS_AUG_TRAINING and args.method in ["EDA_NLL_EMA", "EDA_EMA"]:
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
                if trained_iter % args.wandb_step == 0:
                    wandb.log({f'f1': score, f'step_{run_id}':trained_iter})
                if score > max_f1_score and IS_AUG_TRAINING and args.method == "EDA_self_teacher_v3":
                    teacher_model = TextClassificationModel(args)
                    teacher_model.to(args.teacher_device)
                    copy_params(model, teacher_model)
                    teacher_model.eval()
                if args.syn_noise:
                    if IS_AUG_TRAINING:
                        max_f1_score = max(max_f1_score,score)
                    else:
                        max_f1_score = 0
                else:
                    max_f1_score = max(max_f1_score,score)
                model.train()

        # os._exit(233)
    return max_f1_score

def copy_params(model, ema_model):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data = param.data

def compute_model(args):
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

    for i in range(5):
        setup_seed(i+1)
        ema_model = None
        if  args.method in ["baseline_NLL", "EDA_NLL", "EPDA_NLL", 
                            "EDA_NLL_NC"]:
            model = NLLModel(args)
        elif args.method in ["EDA_NLL_EMA"]:
            model = NLLModel(args)
            ema_model = NLLModel(args, ema=True)
        elif args.method in ["EDA_EMA"]:
            model = TextClassificationModel(args)
            model.to("cuda")
            ema_model = TextClassificationModel(args, ema=True)
            ema_model.to("cuda")
        else: # baselines, EDA, EPDA, EDA_NLL_self
            if args.method in ["EDA_NLL_self", "EDA_self_BC", 
                               "EDA_self_teacher_v1", "EDA_self_teacher_v3", "EDA_self_teacher_v4",
                               "EDA_self_teacher_v4_BC", "EDA_self_LS"]:
                assert args.classifier_dropout is not None
            model = TextClassificationModel(args)
            model.to("cuda")
        f1 = train(train_data_loader,test_data_loader,model,args,TOKENIZER,ema_model=ema_model,run_id=i)
        f1_scores.append(f1)
        print("[IMPORTANT] i=",i,"Current F1 Score",f1,"Average F1 Score: ",sum(f1_scores)/len(f1_scores), flush = True)
    wandb.log({"final_f1":sum(f1_scores)/len(f1_scores)})
    wandb.log({"std_f1":np.std(100*np.array(f1_scores))})
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

    parser.add_argument("--project_name", type=str, default="DA-NLL")
    parser.add_argument("--wandb_step", type=int, default=5)

    parser.add_argument("--n_model", type=int, default=2)
    parser.add_argument("--alpha_t", type=float, default=50.0) # CR weight
    parser.add_argument("--alpha_c", type=float, default=50.0) # training consistency
    parser.add_argument("--alpha_warmup_ratio", default=0.1, type=float)

    # Data Augmentaion related:
    parser.add_argument("--train_trick", default="", choices=["mix_training", "label_consistency", ""],
        help="mix_training: mix the original training set and the augmented set together.")
    parser.add_argument("--num_aug", default=3, type=float)
    parser.add_argument("--refresh_aug_epoch", default=5, type=int)
    parser.add_argument('--mix_up', action='store_true', ) # default is false
    parser.add_argument("--method", default="EDA_NLL",
                        choices=["baseline", "baseline_NLL", "EDA", "EPDA", "EDA_NLL", "EPDA_NLL", "EDA_NLL_EMA",
                                 "EDA_NLL_self", 
                                 "EDA_NLL_NC",  
                                 "EDA_self_BC", 
                                 "EDA_self_teacher_v1", "EDA_self_teacher_v3", "EDA_self_teacher_v4",
                                 "EDA_self_teacher_v4_BC",
                                 "glitter",
                                 "large_loss",
                                 "reweight",
                                 "EDA_self_LS",
                                 "EDA_EMA",
                                 ],
                        help="EDA_NLL_NC: null class"
                             "EDA_self_BC: EDA + self-regularization + Background Class"
                             "EDA_self_teacher_v1: EDA + self-reg, and use the teacher model's logits"
                             "EDA_self_teacher_v3: EDA + self-reg, teacher is the best teacher"
                             "EDA_self_teacher_v4: EDA + self-reg, use teacher's logits for aug only"
                             "EDA_self_teacher_v4_BC: v4 + BC"
                             "glitter: glitter algorithm, filter out small loss triples"
                             "large_loss: filter out large loss triples"
                             "reweight: re-weighting mechanisms "
                             "EDA_self_LS: label smoothing"
                             "EDA_EMA: EDA + EMA",)
    parser.add_argument('--syn_noise',action='store_true', help="whether to add noise to augmented data")
    parser.add_argument("--syn_noise_ratio", default=0.05, type=float) # probability to flip label under EDA_noise

    parser.add_argument('--org_teacher',action='store_true', help="whether to use a teacher model")
    parser.add_argument('--teacher_device',type=int, default=0)
    parser.add_argument("--teacher_temperature", default=1.0, type=float) # temperature for KD

    parser.add_argument("--bg_class_prior", default=0.05, type=float)
    parser.add_argument("--eta", default=1e-4, type=float, help="learning rate of epsilon in NC_meta")
    parser.add_argument("--label_smoothing", default=0.02, type=float)

    parser.add_argument("--epda_engine", default="EDA")
    parser.add_argument("--alpha_da", default=0.05, type=float)
    parser.add_argument("--ema_decay", default=0.999, type=float)
    parser.add_argument("--classifier_dropout", default=None, type=float)

    # IO options
    parser.add_argument('--save_model', action='store_true', )  # whether to save the model
    parser.add_argument('--save_path', default='models')

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
    elif 'offense' in args.train_dir:
        args.num_classes = 4

    if args.method in ["EDA_NLL_NC", "EDA_self_BC", "EDA_self_teacher_v4_BC"]:
        args.num_classes += 1

    if 'hk' in os.uname()[1]: # machine node name
        now = datetime.now() + timedelta(hours=-15) # from HKT to PST
    else:
        now = datetime.now()

    if args.method == "EDA_self_BC":
        method_name = "EDA_self_BC_fixed"
    else:
        method_name = args.method
    run_name = "-".join([args.dataset, args.data_split, method_name, now.strftime("%m-%d-%H:%M")] )

    wandb.init(project=args.project_name+"-"+args.dataset,
               name=run_name,
               config=args)

    compute_model(args)
