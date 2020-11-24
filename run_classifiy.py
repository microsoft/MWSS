'''
Copyright (c) Microsoft Corporation, Yichuan Li and Kai Shu.
Licensed under the MIT license.
Authors: Guoqing Zheng (zheng@microsoft.com), Yichuan Li and Kai Shu
'''
import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import trange
from dataset import FakeNewsDataset, SnorkelDataset
# from dataset import FakeNewsDataset
from l2w import step_l2w
from l2w import step_l2w_group_net
# from model import RobertaForSequenceClassification, CNN_Text, GroupWeightModel
from model import RobertaForSequenceClassification, CNN_Text, FullWeightModel, GroupWeightModel, \
    BertForSequenceClassification
import time

import os, sys
from model import DistilBertForSequenceClassification

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    DistilBertTokenizer,
    DistilBertConfig,
    RobertaConfig,
    RobertaTokenizer,
    BertTokenizer,
    BertConfig
)

import shutil
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import os

writer = None

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

#
# MODEL_CLASSES = {
#     "cnn":(None, CNN_Text, DistilBertTokenizer),
#     "albert":(AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer)
# }
TRAIN_TYPE = ["gold", "silver", "gold_con_silver"]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def acc_f1_confusion(preds, labels):
    acc_f1 = {'acc': accuracy_score(y_pred=preds, y_true=labels), 'f1': f1_score(y_pred=preds, y_true=labels)}
    acc_f1.update({"acc_and_f1": (acc_f1['acc'] + acc_f1['f1']) / 2})
    c_m = ",".join([str(i) for i in confusion_matrix(y_true=labels, y_pred=preds).ravel()])
    acc_f1.update({"c_m": c_m})
    return acc_f1


'''
Test ray
'''


def ray_meta_train(config):
    for key, value in config.items():
        if "tuneP_" in key:
            key = key.replace("tuneP_", "")
            setattr(config['args'], key, value)
    # train_mnist(config, config['args'])
    meta_train(config['args'], config['gold_ratio'])
    # with open("/home/yichuan/ray_results/group_weight_np_array_{}.pkl".format(config['gold_ratio']),'wb') as f1:
    #     pickle.dump(instance_weight, f1)
    # return np.mean(meta_train(config['args'], config['gold_ratio'])[-1][0:2])


def build_model(args):
    if args.clf_model.lower() == "cnn":
        # easy for text tokenization
        tokenizer = DistilBertTokenizer.from_pretrained(
            args.model_name_or_path,
            do_lower_case=args.do_lower_case)
        model = CNN_Text(args)
    
    elif args.clf_model.lower() == "robert":
        print("name is {}".format(args.model_name_or_path))
        tokenizer = RobertaTokenizer.from_pretrained(
            args.model_name_or_path,
            do_lower_case=args.do_lower_case
        )
        
        config = RobertaConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=args.num_labels,
            finetuning_task=args.task_name)
        
        model = RobertaForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            config=config
        )
        # freeze the weight for transformers
        if args.freeze:
            for n, p in model.named_parameters():
                if "bert" in n:
                    p.requires_grad = False
    elif args.clf_model.lower() == "bert":
        tokenizer = BertTokenizer.from_pretrained(
            args.model_name_or_path,
            do_lower_case=args.do_lower_case
        )
        
        config = BertConfig.from_pretrained(
            args.model_name_or_path,
            num_labels=args.num_labels,
            finetuning_task=args.task_name)
        
        model = BertForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            config=config
        )
        # freeze the weight for transformers
        # if args.freeze:
        #     for n, p in model.named_parameters():
        #         if "bert" in n:
        #             p.requires_grad = False
    
    else:
        tokenizer = DistilBertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
        config = DistilBertConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_labels,
                                                  finetuning_task=args.task_name)
        model = DistilBertForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    
    model.expand_class_head(args.multi_head)
    model = model.to(args.device)
    return tokenizer, model


def train(args, train_dataset, val_dataset, model, tokenizer, gold_ratio, **kwargs):
    """ Train the model """
    best_acc = 0.
    best_f1 = 0.
    val_acc_and_f1 = 0.
    best_acc_and_f1 = 0.
    best_c_m = ""
    best_loss_val = 10000
    
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    # train_dataloader = DataLoader(train_dataset,  batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader)) + 1
    else:
        t_total = len(train_dataloader) * args.num_train_epochs
    
    if args.clf_model is not "cnn":
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        
    logger.info("Optimizer type: ")
    logger.info(type(optimizer).__name__)
    
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
    )
    logger.info("  Total optimization steps = %d", t_total)
    
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    tr_loss, logging_loss = 0.0, 0.0
    loss_scalar = 0.
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch"
    )
    set_seed(args)  # Added here for reproductibility
    
    for _ in train_iterator:
        for step, batch in enumerate(train_dataloader):
            
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            
            model.train()
            
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as loss:
                    loss.backward()
            else:
                loss.backward()
            
            tr_loss += loss.item()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            if args.clf_model is not 'cnn':
                scheduler.step()
            else:
                scheduler.step(loss)
            
            model.zero_grad()
            
            global_step += 1
            
            if args.logging_steps != 0 and global_step % args.logging_steps == 0:
                logs = {}
                if (
                        args.evaluate_during_training
                ):
                    
                    results = evaluate(args, model, tokenizer, gold_ratio, eval_dataset=val_dataset)
                    results.update({"type": "val"})
                    if (val_acc_and_f1 < results['acc_and_f1'] and args.val_acc_f1) \
                            or (best_loss_val > results['loss'] and args.val_acc_f1 is False):
                        val_acc_and_f1 = results['acc_and_f1']
                        best_loss_val = results['loss']
                        results = evaluate(args, model, tokenizer, gold_ratio)
                        best_acc = results['acc']
                        best_f1 = results['f1']
                        best_acc_and_f1 = results["acc_and_f1"]
                        best_c_m = results['c_m']
                        results.update({"type": "test"})
                    print(json.dumps(results))
                    for key, value in results.items():
                        eval_key = "eval_{}".format(key)
                        logs[eval_key] = value
                    logging.info(
                        "Training Loss is {}".format(loss_scalar if loss_scalar > 0 else tr_loss / args.logging_steps))
                loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                learning_rate_scalar = optimizer.defaults.get("lr", 0)
                logs["learning_rate"] = learning_rate_scalar
                logs["loss"] = loss_scalar
                # if results['type'] == "test":
                
                logging_loss = tr_loss
            
            if args.max_steps > 0 and global_step > args.max_steps:
                break
        # logs_epoch = {}
        # results = evaluate(args, model, tokenizer, gold_ratio)
        # if val_acc_and_f1 < results['acc_and_f1']:
        #     best_f1 = results['f1']
        #     best_acc = results['acc']
        #     best_c_m = results["c_m"]
        #     val_acc_and_f1 = results['acc_and_f1']
        # for key, value in results.items():
        #     eval_key = "eval_{}".format(key)
        #     logs_epoch[eval_key] = value
        # print("EPOCH Finish")
        # print("EPOCH Result {}".format(json.dumps(logs_epoch)))
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    
    return global_step, tr_loss / global_step, (best_f1, best_acc, best_c_m)


import copy


def meta_train(args, gold_ratio):
    """ Train the model """
    best_acc = 0.
    best_f1 = 0.
    best_loss_val = 100000
    val_acc_and_f1 = 0.
    best_cm = ""
    fake_acc_and_f1 = 0.
    fake_best_f1 = 0.
    fake_best_acc = 0.
    writer = None
    tokenizer, model = build_model(args)
    g_dataset = load_fake_news(args, tokenizer, evaluate=False, train_path=args.gold_train_path)
    s_dataset = load_fake_news(args, tokenizer, evaluate=False, train_path=args.silver_train_path, is_weak=True,
                               weak_type=args.weak_type)
    val_dataset = load_fake_news(args, tokenizer, evaluate=False, train_path=args.val_path)
    
    eval_dataset = copy.deepcopy(val_dataset)

    # make a copy of train and test towards similar size as the weak source
    if True:
        max_length = max(len(g_dataset), len(s_dataset), len(val_dataset))
        g_dataset = torch.utils.data.ConcatDataset([g_dataset] * int(max_length / len(g_dataset)))
        s_dataset = torch.utils.data.ConcatDataset([s_dataset] * int(max_length / len(s_dataset)))
        val_dataset = torch.utils.data.ConcatDataset([val_dataset] * int(max_length / len(val_dataset)))
    
    g_sampler = RandomSampler(val_dataset)
    g_dataloader = DataLoader(val_dataset, sampler=g_sampler, batch_size=args.g_train_batch_size)
    
    train_sampler = RandomSampler(g_dataset)
    train_dataloader = DataLoader(g_dataset, sampler=train_sampler, batch_size=args.g_train_batch_size)
    
    s_sampler = RandomSampler(s_dataset)
    s_dataloader = DataLoader(s_dataset, sampler=s_sampler, batch_size=args.s_train_batch_size)
    
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(g_dataloader)) + 1
    else:
        if gold_ratio == 0:
            t_total = min(len(g_dataloader), len(s_dataloader)) * args.num_train_epochs
        else:
            t_total = min(len(g_dataloader), len(train_dataloader), len(s_dataloader)) * args.num_train_epochs
    
    if args.clf_model is not "cnn":
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if
                           not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.0},
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, t_total / args.num_train_epochs)
    
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.use_group_weight or args.use_group_net:
        #
        if args.use_group_weight:
            group_weight = GroupWeightModel(n_groups=args.multi_head)
        else:
            group_weight = FullWeightModel(n_groups=args.multi_head, hidden_size=args.hidden_size)
        group_weight = group_weight.to(args.device)
        parameters = [i for i in group_weight.parameters() if i.requires_grad]
        if "adam" in args.group_opt.lower():
            
            if "w" in args.group_opt.lower():
                group_optimizer = AdamW(parameters, lr=args.group_lr, eps=args.group_adam_epsilon,
                                        weight_decay=args.group_weight_decay)
            else:
                group_optimizer = torch.optim.Adam(parameters, lr=args.group_lr, eps=args.group_adam_epsilon,
                                                   weight_decay=args.group_weight_decay)
            
            group_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(group_optimizer,
                                                                         t_total / args.num_train_epochs)
            # group_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
            #                                                   num_training_steps=t_total)
        elif args.group_opt.lower() == "sgd":
            group_optimizer = torch.optim.SGD(parameters, lr=args.group_lr, momentum=args.group_momentum,
                                              weight_decay=args.group_weight_decay)
            group_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(group_optimizer, 'min')
    
        if args.fp16:
            group_weight, group_optimizer= amp.initialize(group_weight, group_optimizer, opt_level=args.fp16_opt_level)

    # # Train!
    logger.info("***** Running training *****")
    logger.info("  Num Gold examples = %d, Silver Examples = %d", len(val_dataset), len(s_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d, %d",
        args.g_train_batch_size, args.s_train_batch_size
    )
    logger.info("  Total optimization steps = %d", t_total)
    
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    
    g_loss, logging_g_loss, logging_s_loss, s_loss = 0.0, 0.0, 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch"
    )
    set_seed(args)  # Added here for reproductibility
    temp_output = open(args.flat_output_file+"_step", "w+", 1)
    for _ in train_iterator:
        be_changed = False
        for step, (g_batch, s_batch, train_batch) in enumerate(zip(g_dataloader, s_dataloader, train_dataloader)):
            
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            
            model.train()
            g_batch = tuple(t.to(args.device) for t in g_batch)
            g_input = {"input_ids": g_batch[0], "attention_mask": g_batch[1], "labels": g_batch[2]}
            
            s_batch = tuple(t.to(args.device) for t in s_batch)
            s_input = {"input_ids": s_batch[0], "attention_mask": s_batch[1], "labels": s_batch[2],
                       "reduction": 'none'}
            
            train_batch = tuple(t.to(args.device) for t in train_batch)
            train_input = {"input_ids": train_batch[0], "attention_mask": train_batch[1], "labels": train_batch[2]}
            # ATTENTION: RoBERTa does not need token types id
            if args.multi_head > 1:
                s_input.update({"is_gold": False})
            
            if (global_step + 1) % args.logging_steps == 0:
                step_input = global_step
            else:
                step_input = None
            info = {"gold_ratio": gold_ratio, "step": step_input}
            

            if args.use_group_net:
                outputs = step_l2w_group_net(model, optimizer, scheduler, g_input, s_input, train_input, args,
                                             group_weight, group_optimizer, group_scheduler, gold_ratio)

                loss_g, loss_s, instance_weight = outputs
            else:
                outputs = step_l2w(model, optimizer, scheduler, g_input, s_input, train_input, args, gold_ratio)
                loss_g, loss_s = outputs
            
            g_loss += loss_g.item()
            s_loss += loss_s.item()
            global_step += 1
            
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logs = {}
                results = {}
                if (args.evaluate_during_training) or True:
                    
                    results = evaluate(args, model, tokenizer, gold_ratio, eval_dataset=eval_dataset)
                    results = {key + "_val": value for key, value in results.items()}
                    results.update({"type": "val"})
                    print(json.dumps(results))
                    if val_acc_and_f1 < results['acc_and_f1_val'] :
                        be_changed = True
                        best_loss_val = results['loss_val']
                        val_acc_and_f1 = results['acc_and_f1_val']
                        test_results = evaluate(args, model, tokenizer, gold_ratio)
                        best_acc = test_results['acc']
                        best_f1 = test_results['f1']
                        best_cm = test_results['c_m']
                        best_acc_and_f1 = test_results["acc_and_f1"]
                        temp_output.write("Step: {}, Test F1: {}, Test ACC: {}; Val Acc_and_F1: {}, Val Loss: {}\n".format(global_step, best_f1, best_acc, val_acc_and_f1, best_loss_val))
                        temp_output.flush()
                        # save the model
                        if args.save_model:
                            save_path = args.flat_output_file + "_save_model"
                            save_dic = {"BaseModel": model,
                                        "LWN":group_weight,
                                        "step":global_step,
                                        "tokenizer":tokenizer
                                        }
                            torch.save(save_dic, save_path)
                        
                        test_results = {key + "_test": value for key, value in test_results.items()}
                        test_results.update({"type": "test"})
                        print(json.dumps(test_results))
                    for key, value in results.items():
                        eval_key = "eval_{}".format(key)
                        logs[eval_key] = value
                
                loss_scalar = (g_loss - logging_g_loss) / args.logging_steps
                learning_rate_scalar = optimizer.defaults.get("lr", 0)
                logs["train_learning_rate"] = learning_rate_scalar
                logs["train_g_loss"] = loss_scalar
                logs["train_s_loss"] = (s_loss - logging_s_loss) / args.logging_steps
                logging_g_loss = g_loss
                logging_s_loss = s_loss
                
                # writer.add_scalar("Loss/g_train_{}".format(gold_ratio), logs['train_g_loss'], global_step)
                # writer.add_scalar("Loss/s_train_{}".format(gold_ratio), logs['train_s_loss'], global_step)
                # writer.add_scalar("Loss/val_train_{}".format(gold_ratio), results['loss_val'], global_step)
                
                
                if args.use_group_weight:
                    try:
                        eta_group = group_optimizer.get_lr()
                    except:
                        eta_group = group_optimizer.defaults.get("lr", 0)
                        
                        # writer.add_scalar("Loss/group_lr_{}".format(gold_ratio), eta_group, global_step)
                
                print(json.dumps({**{"step": global_step}, **logs}))
            
            if args.max_steps > 0 and global_step > args.max_steps:
                break
        if (args.use_group_net or args.use_group_weight) and isinstance(group_scheduler,
                                                                        torch.optim.lr_scheduler.CosineAnnealingLR):
            group_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(group_optimizer,
                                                                         t_total / args.num_train_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, t_total / args.num_train_epochs)


        print("EPOCH Finish")
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
    temp_output.close()
    # return cache_instance_weight
    return global_step, g_loss / global_step, (best_f1, best_acc, best_cm)


def evaluate(args, model, tokenizer, gold_ratio, prefix="", eval_dataset=None):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    
    
    results = {}
    
    if eval_dataset is None:
        eval_dataset = load_fake_news(args, tokenizer, evaluate=True)
    
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    
    # Eval!
    print("\n")
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        
        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[2]}
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
    
    eval_loss = eval_loss / nb_eval_steps
    
    preds = np.argmax(preds, axis=1)
    
    result = acc_f1_confusion(preds, out_label_ids)
    results.update(result)
    results.update({"loss": eval_loss})
    
    logger.info("***** Eval results {} Gold Ratio={} *****".format(prefix, gold_ratio))
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))
    
    return results


def load_fake_news(args, tokenizer, is_weak=False, evaluate=False, train_path=None, weak_type=""):
    file_path = args.eval_path if evaluate else train_path
    if args.use_snorkel and "noise" in file_path:
        dataset = SnorkelDataset(file_path, tokenizer, args.max_seq_length, overwrite=True)
    else:
        dataset = FakeNewsDataset(file_path, tokenizer, is_weak, args.max_seq_length, weak_type, args.overwrite_cache,
                                  args.balance_weak)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--ray_dir',
        default='.',
        type=str,
        help='Path to Ray tuned results (Default: current directory)',
    )
    
    parser.add_argument(
        "--model_name_or_path",
        default="distilbert-base-uncased",
        type=str,
        help="Path to pre-trained model or shortcut name selected in the list",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    
    # Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=256,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )
    
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--g_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for gold training.")
    parser.add_argument("--s_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for silver training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=128, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=1e-3,type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    # parser.add_argument("--train_path", type=str, default="/home/yichuan/MSS/data/gossip/weak", help="For distant debugging.")
    # parser.add_argument("--eval_path", type=str, default="/home/yichuan/MSS/data/gossip/test.csv", help="For distant debugging.")
    
    parser.add_argument("--train_path", type=str, default="./data/gossip/weak",
                        help="For distant debugging.")
    parser.add_argument("--eval_path", type=str, default="./data/gossip/test.csv",
                        help="For distant debugging.")
    parser.add_argument("--meta_learn", action="store_true", help="Whether use meta learning or not")
    parser.add_argument("--train_type", type=int, default=0,
                        help="0: only clean data, 1: only noise data, 2: concat clean and noise data")
    parser.add_argument("--weak_type", type=str, default="most_vote",
                        help="method for the weak superivision; for multi-head please set as none")
    parser.add_argument("--multi_head", type=int, default=1, help="count of head for classification task")
    
    # CNN parameters
    parser.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('--max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
    parser.add_argument('--kernel-num', type=int, default=100, help='number of each kind of kernel')
    parser.add_argument('--kernel-sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for classification cross-entropy classification')
    parser.add_argument("--use_group_weight", action="store_true")
    parser.add_argument("--use_group_net", action="store_true")
    
    parser.add_argument("--clf_model", type=str, default="cnn", help="fake news classification model 'cnn', 'bert' ")
    parser.add_argument("--group_lr", type=float, default=1e-5, help="learn rate for group weight")
    parser.add_argument("--group_momentum", type=float, default=0.9, help="momentum for group weight")
    parser.add_argument("--group_weight_decay", type=float, default=0.0, help="weight decay for group weight")
    parser.add_argument("--group_adam_epsilon", type=float, default=1e-8, help="adam epsilon")
    parser.add_argument("--group_opt", type=str, default="sgd", help="optimizer type for group weight")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--balance_weak", action="store_true")
    
    # validation setting
    parser.add_argument("--val_acc_f1", action="store_true",
                        help="Whether use the (f1+acc)/2 as the metric for model selection on validation dataset")
    parser.add_argument("--gold_ratio", default=0, type=float, help="gold ratio selection")
    
    # baseline setting
    parser.add_argument("--use_snorkel", action="store_true",
                        help="Snorkel baseline which use LabelModel to combine multiple weak source")
    parser.add_argument("--fp16", action='store_true', help='whehter use fp16 or not')
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument(
        "--id",
        type=str,
        default="",
        help="id for this group of parameters"
    )

    parser.add_argument(
        "--save_model",
        action="store_true",
    )
    
    args = parser.parse_args()
    
    args.clf_model = args.clf_model.lower()
    # args.gold_ratio = [args.gold_ratio] if (
    #     args.gold_ratio != 0 and args.gold_ratio in [0.02, 0.04, 0.06, 0.08, 0.1]) else [0.06, 0.04, 0.08, 0.02, 0.1]
    
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    args.hidden_size = len(args.kernel_sizes) * args.kernel_num if args.clf_model == "cnn" else 768
    
    # Setup CUDA, GPU & distributed training
    if args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        device = torch.device("cuda")
        args.n_gpu = 1
    args.device = device
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Set seed
    set_seed(args)
    
    # binary classification task
    args.task_name = "mrpc".lower()
    # 0 for genuine news; 1 for fake news
    args.num_labels = 2
    if "/" != args.eval_path[0]:
        args.eval_path = os.path.join(os.getcwd(), args.eval_path)
    if "/" != args.train_path[0]:
        args.train_path = os.path.join(os.getcwd(), args.train_path)
    
    dataset_type = "political" if "political" in args.eval_path else "gossip"
    args.t_out_dir = os.path.join(args.output_dir, "{}_{}_{}_{}".format(args.clf_model,
                                                                        "meta" if args.meta_learn else TRAIN_TYPE[
                                                                            args.train_type],
                                                                        dataset_type, args.weak_type))
    
    assert args.use_group_weight + args.use_group_net != 2, "You should choose GroupWeight or GroupNet, not both of them "
    
    if args.use_group_weight:
        args.t_out_dir += "_group"
    if args.use_group_net:
        args.t_out_dir += "_group_net"
    elif args.meta_learn:
        args.t_out_dir += "_L2W"
    if args.use_snorkel:
        args.t_out_dir += "_snorkel"

    # ATTENTION: for batch run, the gold ratio should settup manually
    args.gold_ratio = [args.gold_ratio]
    assert len(args.gold_ratio) == 1, "For computation efficiency, please run one gold ratio at a time"
    flat_output_file = os.path.join(args.t_out_dir, "result_{}.txt".format(args.gold_ratio[0]))

    if len(args.id) > 0:
        flat_output_file += "-" + str(args.id)
    else:
        flat_output_file += "-" + str(time.time())


    # will overwrite now
    #if os.path.exists(flat_output_file):
    #    raise FileExistsError("The result file already exist, please check it")
    setattr(args, "flat_output_file", flat_output_file)
    logging.info(args.t_out_dir)
    
    # try:
    #     shutil.rmtree(args.t_out_dir)
    # except FileNotFoundError:
    #     print("File Already deleted")
    global writer
    writer = SummaryWriter(args.t_out_dir)
    fout1 = open(flat_output_file, "w")
    fout1.write("GoldRatio\tF1\tACC\tCM\n")

    # 1e-5; GN 1e-5
    for gold_ratio in args.gold_ratio:
        gold_train_path = os.path.join(args.train_path, "gold_{}.csv".format(gold_ratio))
        silver_train_path = os.path.join(args.train_path, "noise_{}.csv".format(gold_ratio))
        val_path = os.path.join(args.train_path, "../val.csv")
        
        logger.info("Training/evaluation parameters %s", args)
        
        # Training
        if args.do_train:
            
            args.gold_train_path = gold_train_path
            args.silver_train_path = silver_train_path
            args.val_path = val_path
            if args.meta_learn:
                _,  _, (best_f1, best_acc, best_cm) = meta_train(args, gold_ratio)
                fout1.write("{}\t{}\t{}\t{}\n".format(gold_ratio, best_f1, best_acc, best_cm))
            else:
                tokenizer, model = build_model(args)
                gold_dataset = load_fake_news(args, tokenizer, evaluate=False, train_path=gold_train_path)
                silver_dataset = load_fake_news(args, tokenizer, evaluate=False, train_path=silver_train_path,
                                                is_weak=True,
                                                weak_type=args.weak_type)
                val_dataset = load_fake_news(args, tokenizer, evaluate=False, train_path=val_path)
                if args.train_type == 0:
                    train_dataset = gold_dataset
                elif args.train_type == 1:
                    train_dataset = silver_dataset
                else:
                    # make a copy here for data imbalance
                    gold_dataset = torch.utils.data.ConcatDataset(
                        [gold_dataset] * int(len(silver_dataset) / len(gold_dataset)))
                    train_dataset = torch.utils.data.ConcatDataset([gold_dataset, silver_dataset])
                global_step, tr_loss, (f1, acc, c_m) = train(args, train_dataset, val_dataset, model, tokenizer,
                                                             gold_ratio=gold_ratio)
                fout1.write("{}\t{}\t{}\t{}\n".format(gold_ratio, f1, acc, c_m))
            logger.info("Gold Ratio {} Training Finish".format(gold_ratio))
            # logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
            # logger.info(" f1 = %s, acc = %s, c_m = %s", f1, acc, c_m)
            # writer.add_scalar("BestResult/F1",f1,global_step=int(gold_ratio * 100))
            # writer.add_scalar("BestResult/Acc",acc,global_step=int(gold_ratio * 100))
            # writer.add_text("BestResult/ConfusionMatrix",c_m, global_step=int(gold_ratio * 100))
            # fout1.write("{}\t{}\t{}\t{}\n".format(gold_ratio, f1, acc, c_m))
    fout1.close()


if __name__ == "__main__":
    main()
