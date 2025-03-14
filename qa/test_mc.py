# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for multiple choice (Bert, Roberta, XLNet)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer, XLNetConfig,
                          XLNetForMultipleChoice, XLNetTokenizer,
                          RobertaConfig, RobertaForMultipleChoice,
                          BertForMultipleChoice, RobertaTokenizer)

from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from torch.optim import Adam

from data_utils import (convert_examples_to_features, processors, get_save_dir)
# from winogrande_data_utils import convert_examples_to_features as convert_examples_to_features_winogrande
from model_mc import QACRModel, kl_div
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForMultipleChoice
# from ..utils import

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertForMultipleChoice, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForMultipleChoice, XLNetTokenizer),
    'roberta': (RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer)
}


def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features]
            for feature in features]


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def randargmax(b, axis=1):
    """ a random tie-breaking argmax"""
    return np.argmax(np.random.random(b.shape) * (b == np.repeat(
        np.expand_dims(b.max(axis=axis), axis), b.shape[axis], axis=axis)),
                     axis=axis)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def test(args, model, tokenizer, prefix="", test=True):
    eval_task_names = (args.task_name, )
    eval_outputs_dirs = (args.output_dir, )

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset, examples = load_and_cache_examples(args,
                                                         eval_task,
                                                         tokenizer,
                                                         evaluate=not test,
                                                         test=test, filename=args.test_file)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(
            1, args.n_gpu)

        eval_dataloader = DataLoader(eval_dataset,
                                     batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running test {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        random_state = torch.get_rng_state()
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    'input_ids':
                    batch[0],
                    'attention_mask':
                    batch[1],
                    'token_type_ids':
                    batch[2] if args.model_type in ['bert', 'xlnet'] else
                    None,  # XLM don't use segment_ids
                    'labels':
                    batch[3]
                }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids,
                    inputs['labels'].detach().cpu().numpy(),
                    axis=0)
        torch.set_rng_state(random_state)
        
        print(preds[:10, :])
        if args.return_soft:
            preds = F.softmax(torch.tensor(preds)/args.temperature, -1).detach().cpu().numpy()

            output_eval_file = os.path.join(
                eval_output_dir,
                args.test_file + f"_soft_logits_temp_{args.temperature}.txt")
            with open(output_eval_file, "w") as writer:
                for i in range(len(preds)):
                    writer.write(str(preds[i, 0])+"\t"+str(preds[i, 1])+"\n")
        else:
            preds = F.softmax(torch.tensor(preds), -1).detach().cpu().numpy()
            preds = randargmax(preds, axis=1)

            acc = simple_accuracy(preds, out_label_ids) # may not reflect the real acc as in test set no labels are provided. 

            output_eval_file = os.path.join(
                eval_output_dir,
                args.test_file + "_eval_results.txt")

            if args.task_name == "winogrande":
                label_map = {0: "1", 1: "2"}
            else:
                label_map = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}
            os.makedirs(os.path.dirname(output_eval_file), exist_ok=True)
            with open(output_eval_file, "w") as writer:
                for i in range(len(preds)):
                    if args.task_name != "winogrande":
                        writer.write(examples[i].example_id + "," +
                                    label_map[preds[i]] + "\n")
                    else:
                        writer.write(label_map[preds[i]] + "\n")
            print("final acc", acc)
    return results


def load_and_cache_examples(args, task, tokenizer, filename="", evaluate=False, test=False):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier(
        )  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    # Load data features from cache or dataset file
    if evaluate:
        cached_mode = 'dev_random'
    elif test:
        cached_mode = 'test'
    else:
        cached_mode = 'train'
    assert (evaluate == True and test == True) == False
    cached_features_file = os.path.join(
        args.data_dir, 'cached_{}_{}_{}_{}{}{}_method{}'.format(
            cached_mode,
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length), str(task),
            "_no_q" if args.mask_question else "", filename, args.method))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir)
        elif test:
            examples = processor.get_test_examples(args.data_dir, filename)
        else:
            if args.train_file != "":
                examples = processor.get_train_examples(args.data_dir, filename)
            else:
                examples = processor.get_train_examples(args.data_dir)


        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir)
        elif test:
            examples = processor.get_test_examples(args.data_dir, args.test_file)
        else:
            if args.train_file != "":
                examples = processor.get_train_examples(args.data_dir, file=args.train_file)
            else:
                examples = processor.get_train_examples(args.data_dir)
        logger.info("Training number: %s", str(len(examples)))
        if args.task_name != "winogrande":
            features = convert_examples_to_features(
                examples,
                label_list,
                args.max_seq_length,
                tokenizer,
                cls_token_at_end=bool(args.model_type in ['xlnet']
                                      ),  # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                sep_token=tokenizer.sep_token,
                sep_token_extra=False,
                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                pad_on_left=bool(
                    args.model_type in ['xlnet']),  # pad on the left for xlnet
                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                pad_token=tokenizer.pad_token_id,
                mask_question=args.mask_question,
                pad_qa= args.task_name in ["commonsenseqa", "arc"])
        else:
            # features = convert_examples_to_features_winogrande(
            features = convert_examples_to_features(
                examples,
                label_list,
                args.max_seq_length,
                tokenizer,
                cls_token_at_end=bool(args.model_type in ['xlnet']
                                      ),  # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                sep_token=tokenizer.sep_token,
                sep_token_extra=False,
                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                pad_on_left=bool(
                    args.model_type in ['xlnet']),  # pad on the left for xlnet
                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                pad_token=tokenizer.pad_token_id,
                mask_question=args.mask_question,
                pad_qa=True,
                is_winogrande=True)

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s",
                        cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier(
        )  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(select_field(features, 'input_ids'),
                                 dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'),
                                  dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, 'segment_ids'),
                                   dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)

    all_is_synthetic = torch.tensor([f.is_synthetic for f in features], dtype=torch.bool)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_label_ids, all_is_synthetic)
    if test:
        return dataset, examples
    else:
        return dataset


def main():
    

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default="/scratch/yyv959/commonsenseqa/",
        type=str,
        help=
        "The input data dir. Should contain the .tsv files (or other data files) for the task."
    )
    parser.add_argument(
        "--test_file",
        default="",
        type=str,
        help=
        "Name of the test csv"
    )
    parser.add_argument("--model_type",
                        default='bert',
                        type=str,
                        help="Model type selected in the list: " +
                        ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-uncased",
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument(
        "--task_name",
        default="commonsenseqa",
        type=str,
        help="The name of the task to train selected in the list: " +
        ", ".join(processors.keys()))
    parser.add_argument("--wandb_name", default="", type=str)
    parser.add_argument(
        "--output_dir",
        default="/scratch/yyv959/commonsenseqa/outputs/mc/",
        type=str,
        help=
        "The output directory where the model predictions and checkpoints will be written."
    )

    ## Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help=
        "Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument(
        "--max_seq_length",
        default=70,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--optimizer",
                        type=str,
                        default="adamw",
                        help="Whether to run training.")
    parser.add_argument(
        "--do_lower_case",
        action='store_true',
        help="Set this flag if you are using an uncased model.")
    parser.add_argument(
        "--mask_question",
        action='store_true',
        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size",
                        default=16,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size",
                        default=32,
                        type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon",
                        default=1e-6,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help=
        "If > 0: set total number of training steps to perform. Override num_train_epochs."
    )
    parser.add_argument("--warmup_ratio",
                        default=0.0,
                        type=float,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--linear_decay",
                        action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument('--logging_steps',
                        type=int,
                        default=609,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps',
                        type=int,
                        default=100000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action='store_true',
        help=
        "Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number"
    )
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir',
                        action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument(
        '--overwrite_cache',
        action='store_true',
        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")

    parser.add_argument(
        '--fp16',
        action='store_true',
        help=
        "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"
    )
    parser.add_argument(
        '--fp16_opt_level',
        type=str,
        default='O1',
        help=
        "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip',
                        type=str,
                        default='',
                        help="For distant debugging.")
    parser.add_argument('--server_port',
                        type=str,
                        default='',
                        help="For distant debugging.")
    parser.add_argument('--method',type=str,default='baseline',
                        choices=['baseline', 'baseline_CR', 'baseline_CR_BC', 'baseline_CR_BC_new'],
                        help="")
    parser.add_argument('--resume_from_checkpoint',type=str,default='')
    parser.add_argument('--resume_from_checkpoint_name',type=str,default='')
    parser.add_argument('--n_model',type=int,default=2,help="")
    parser.add_argument('--alpha_t',type=float,default=0,help="") # default is not apply CR
    parser.add_argument('--bg_class_prior',type=float,default=0.00,help="")
    parser.add_argument('--warmup_epoch',type=int,default=1e10,help="")
    parser.add_argument('--warmup_steps',type=int,default=1e10,help="") # default is not apply CR.
    parser.add_argument('--early_eval_steps',type=int,default=100000,help="early evaluation steps")
    
    # 
    parser.add_argument('--return_soft',action='store_true', help="whether to return soft label")
    parser.add_argument('--temperature',type=float,default=1,help="temperature of the softmax")

    args = parser.parse_args()

    if "uncase" in args.model_name_or_path:
        assert args.do_lower_case

    from datetime import datetime, timedelta

    if 'hk' in os.uname()[1]: # machine node name
        now = datetime.now() + timedelta(hours=-15) # from HKT to PST
    else:
        now = datetime.now()

    args.device = "cuda"
    args.n_gpu = torch.cuda.device_count()
    args.devices = [i % args.n_gpu for i in range(args.n_model)]

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1),
        args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()

    if args.method in ["baseline_CR_BC"]:
        args.task_name += "_bc"

    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    
    # config = config_class.from_pretrained(
    #     args.config_name if args.config_name else args.model_name_or_path,
    #     num_labels=num_labels,
    #     finetuning_task=args.task_name)

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name
        if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case)



    # model = model_class.from_pretrained(args.model_name_or_path)
    model = AutoModelForMultipleChoice.from_pretrained(args.model_name_or_path)

    checkpoint = torch.load(args.resume_from_checkpoint)
    if any([key.startswith("models") for key,val in checkpoint.items()]):
        c = dict([(key[9:], val) for key,val in checkpoint.items() if key.startswith("models.0.")])
    else:
        c = checkpoint

    model.load_state_dict(c, strict=True)
    model.to(args.device)

    test(args, model, tokenizer, test=True)

if __name__ == "__main__":
    main()



