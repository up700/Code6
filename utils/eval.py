# -*- coding:utf-8 -*-
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from utils.data_utils import load_and_cache_examples, tag_to_id, get_chunks
from flashtool import Logger


def evaluate(args, model, tokenizer, pad_token_label_id, best, best_bio, mode, logger, prefix="", verbose=True):
    
    eval_dataset = load_and_cache_examples(args, tokenizer, pad_token_label_id, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation %s *****", prefix)
    if verbose:
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds_type = None
    preds_bio = None
    span_label_ids = None
    type_label_ids = None
    # type_label_ids = None
    out_label_ids_type = None
    out_label_ids_bio = None
    att_mask = None
    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels_sp": batch[2], "labels_tp": batch[3], "epoch": 0}
            outputs = model(**inputs)
            loss_sp = outputs[0]
            loss_tp = outputs[1]
            loss_reg = outputs[2]
            span_logits = outputs[5]
            type_logits = outputs[6]
            loss = loss_sp + loss_tp + loss_reg

            if args.n_gpu > 1:
                loss = loss.mean()

            eval_loss += loss.item()

        nb_eval_steps += 1
        
        if preds_type is None:
            preds_type = type_logits.detach() # B, L, C
            preds_bio = span_logits.detach() # B, L, C
            out_label_ids_bio = batch[2] # B, L
            out_label_ids_type = batch[3] # B, L
        else:
            preds_type = torch.cat((preds_type, type_logits.detach()), dim=0)
            preds_bio = torch.cat((preds_bio, span_logits.detach()), dim=0)
            out_label_ids_bio = torch.cat((out_label_ids_bio, batch[2]), dim=0)
            out_label_ids_type = torch.cat((out_label_ids_type, batch[3]), dim=0)

    # preds: nb, type_num, L, span_num
    # out_label_ids: nb*type_num, L
    
    eval_loss = eval_loss/nb_eval_steps
    preds_type = torch.argmax(preds_type, dim=-1)
    preds_bio = torch.argmax(preds_bio, dim=-1)
    
    out_id_list_type = [[] for _ in range(out_label_ids_type.shape[0])]
    preds_id_list_type = [[] for _ in range(out_label_ids_type.shape[0])]

    out_id_list_bio = [[] for _ in range(out_label_ids_bio.shape[0])]
    preds_id_list_bio = [[] for _ in range(out_label_ids_bio.shape[0])]

    for i in tqdm(range(out_label_ids_type.shape[0]), desc="Selecting1"):
        for j in range(out_label_ids_type.shape[1]):
            if out_label_ids_type[i, j] != pad_token_label_id:
                out_id_list_type[i].append(out_label_ids_type[i][j])
                preds_id_list_type[i].append(preds_type[i][j])

    for i in tqdm(range(out_label_ids_bio.shape[0]), desc="Selecting2"):
        for j in range(out_label_ids_bio.shape[1]):
            if out_label_ids_bio[i, j] != pad_token_label_id:
                out_id_list_bio[i].append(out_label_ids_bio[i][j])
                preds_id_list_bio[i].append(preds_bio[i][j])

    correct_preds, total_correct, total_preds = 0., 0., 0. # i variables
    correct_preds_bio, total_correct_bio, total_preds_bio = 0., 0., 0. # i variables

    for ground_truth_id_type, predicted_id_type, ground_truth_id_bio, predicted_id_bio in tqdm(zip(out_id_list_type, \
                                                            preds_id_list_type, out_id_list_bio, preds_id_list_bio), desc="Caculating"):
        # We use the get chunks function defined above to get the true chunks
        # and the predicted chunks from true labels and predicted labels respectively
        lab_chunks, lab_chunks_bio = get_chunks(ground_truth_id_type, ground_truth_id_bio, tag_to_id(args.data_dir, args.dataset))

        lab_chunks      = set(lab_chunks)
        lab_chunks_bio  = set(lab_chunks_bio)
        lab_pred_chunks, lab_pred_chunks_bio = get_chunks(predicted_id_type, predicted_id_bio, tag_to_id(args.data_dir, args.dataset))


        lab_pred_chunks = set(lab_pred_chunks)
        lab_pred_chunks_bio = set(lab_pred_chunks_bio)

        # Updating the i variables
        correct_preds += len(lab_chunks & lab_pred_chunks)
        total_preds   += len(lab_pred_chunks)
        total_correct += len(lab_chunks)

        # Updating the i variables
        correct_preds_bio += len(lab_chunks_bio & lab_pred_chunks_bio)
        total_preds_bio   += len(lab_pred_chunks_bio)
        total_correct_bio += len(lab_chunks_bio)

    p   = correct_preds / total_preds if correct_preds > 0 else 0
    r   = correct_preds / total_correct if correct_preds > 0 else 0
    new_F  = 2 * p * r / (p + r) if correct_preds > 0 else 0

    p_bio   = correct_preds_bio / total_preds_bio if correct_preds_bio > 0 else 0
    r_bio   = correct_preds_bio / total_correct_bio if correct_preds_bio > 0 else 0
    new_F_bio  = 2 * p_bio * r_bio / (p_bio + r_bio) if correct_preds_bio > 0 else 0

    is_updated = False
    if new_F > best[-1]:
        best = [p, r, new_F]
        is_updated = True

    if new_F_bio > best_bio[-1]:
        best_bio = [p_bio, r_bio, new_F_bio]
        # is_updated = True

    results = {
       "loss": eval_loss,
       "precision": p,
       "recall": r,
       "f1": new_F,
       "best_precision": best[0],
       "best_recall": best[1],
       "best_f1": best[-1],
       "precision_bio": p_bio,
       "recall_bio": r_bio,
       "f1_bio": new_F_bio,
       "best_precision_bio": best_bio[0],
       "best_recall_bio": best_bio[1],
       "best_f1_bio": best_bio[-1]
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return best, best_bio, is_updated

