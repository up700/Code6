# -*- coding:utf-8 -*-
import argparse
import glob
import logging
import os
import random
import copy
import math
import json
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import sys
import pickle as pkl
from torch.nn import MSELoss

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    RobertaConfig,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    BertTokenizer,
    BertConfig,
    AutoConfig,
    AutoTokenizer
)


from models.modeling_decom import Task_Decom
from utils.data_utils import load_and_cache_examples, get_labels
from utils.eval import evaluate
from utils.config import config

logger = logging.getLogger(__name__)


torch.set_printoptions(profile="full")

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def initialize(args, tokenizer, t_total, span_num_labels, type_num_labels):
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = Task_Decom.from_pretrained(
        args.model_name_or_path,
        config=config,
        span_num_labels=span_num_labels,
        type_num_labels=type_num_labels,
        lambda_0=args.lambda0, num_epochs=args.num_train_epochs,
        device=args.device,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model.to(args.device)

    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, \
            eps=args.adam_epsilon, betas=(args.adam_beta1, args.adam_beta2))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    model.zero_grad()

    return model, optimizer, scheduler


def validation(args, model, tokenizer, pad_token_label_id, best_dev, best_test, best_dev_bio, best_test_bio,\
         global_step, t_total, epoch):
    best_dev, best_dev_bio, is_updated_dev = evaluate(args, model, tokenizer, \
        pad_token_label_id, best_dev, best_dev_bio, mode="dev", logger=logger, \
        prefix='dev [Step {}/{} | Epoch {}/{}]'.format(global_step, t_total, epoch, args.num_train_epochs), verbose=False)


    if args.save_best and args.local_rank in [-1, 0] and is_updated_dev:
        path = os.path.join(args.output_dir, "checkpoint-best-dev")
        logger.info("Saving span model checkpoint to %s", path)
        if not os.path.exists(path):
            os.makedirs(path)
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(path)


    return best_dev, best_test, best_dev_bio, best_test_bio, is_updated_dev


def train(args, train_dataset, id_to_label_span, id_to_label_type, tokenizer, pad_token_label_id):
    """ Train the model """
    span_num_labels = len(id_to_label_span)
    type_num_labels = len(id_to_label_type)-1
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank==-1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps//(len(train_dataloader)//args.gradient_accumulation_steps)+1
    else:
        t_total = len(train_dataloader)//args.gradient_accumulation_steps*args.num_train_epochs


    model, optimizer, scheduler = initialize(args, tokenizer, t_total, span_num_labels, type_num_labels)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0

    tr_loss, logging_loss = 0.0, 0.0
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    best_dev, best_test = [0, 0, 0], [0, 0, 0]
    best_dev_bio, best_test_bio = [0, 0, 0], [0, 0, 0]

    len_dataloader = len(train_dataloader)
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels_sp": batch[2], "labels_tp": batch[3], "entity_mask": batch[5], "input_ids_ctx": batch[6], "epoch": epoch, "beta1": args.beta1, "beta2": args.beta2}
            outputs = model(**inputs)
            loss_sp = outputs[0]
            loss_tp = outputs[1]
            loss_reg = outputs[2]

            loss = loss_sp + loss_tp + args.alpha*loss_reg

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss/args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()

            if (step+1)%args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step%args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:
                        logger.info("***** training loss : %.4f *****", loss.item())
                        best_dev, best_test, best_dev_bio, best_test_bio, _ = validation(args, model, tokenizer, \
                            pad_token_label_id, best_dev, best_test, best_dev_bio, best_test_bio,\
                            global_step, t_total, epoch)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    results = (best_dev, best_test)

    return results

def main():
    args = config()
    args.do_train = args.do_train.lower()
    args.do_test = args.do_test.lower()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s", "%m/%d/%Y %H:%M:%S")
    logging_fh = logging.FileHandler(os.path.join(args.output_dir, 'log.txt'))
    logging_fh.setLevel(logging.DEBUG)
    logging_fh.setFormatter(formatter)
    logger.addHandler(logging_fh)
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)
    id_to_label_span, id_to_label_type = get_labels(args.data_dir, args.dataset)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train=="true":
        train_dataset = load_and_cache_examples(args, tokenizer, pad_token_label_id, mode="train")
        best_results = train(args, train_dataset,  \
            id_to_label_span, id_to_label_type, tokenizer, pad_token_label_id)

    if args.do_test=="true":
        predict(args, id_to_label_span, id_to_label_type, pad_token_label_id, mode="test")

def predict(args, id_to_label_span, id_to_label_type, pad_token_label_id, mode="test"):
    span_num_labels = len(id_to_label_span)
    type_num_labels = len(id_to_label_type)-1

    path = os.path.join(args.output_dir, "checkpoint-best-dev")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, do_lower_case=args.do_lower_case)
    model = Task_Decom.from_pretrained(
        path, 
        span_num_labels=span_num_labels, 
        type_num_labels=type_num_labels,
        lambda_0=args.lambda0, 
        num_epochs=args.num_train_epochs, 
        device=args.device
    )
    model.to(args.device)

    best_test, best_test_bio = [0, 0, 0], [0, 0, 0]
    best_test, best_test_bio, _ = evaluate(args, model, tokenizer, \
        pad_token_label_id, best_test, best_test_bio, mode, logger=logger, \
        prefix='{}'.format(mode), verbose=False)


if __name__ == "__main__":
    main()
