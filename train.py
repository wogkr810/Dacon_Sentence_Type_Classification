import random
import pandas as pd
import numpy as np
import os
import sys

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm

import warnings
warnings.filterwarnings(action='ignore') 

from arguments import parse_args
from custom_dataset import make_data, CustomDataset, make_roberta_data, RobertaDataset
from model import BaseModel, RobertaLinear, RobertaDocument
from utils.scheduler import get_scheduler
import wandb
from datetime import datetime, timedelta

from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    AutoConfig,
)

from sklearn.model_selection import StratifiedKFold
from trainer import train

import sys

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main():
    num_cores = os.cpu_count()
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # logging
    now_date = datetime.now()
    diff_hours = timedelta(hours=9)
    now_date += diff_hours
    print_now = str(now_date.month) + '_' + str(now_date.day) + '_' + str(now_date.hour) + '_' + str(now_date.minute)

    # wandb.login(key='your key')

    if args.use_tfidf:
        train_vec, train_labels, val_vec, val_labels, test_vec, label_encoder = make_data(args)
        
        train_dataset = CustomDataset(train_vec, train_labels)
        train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers = num_cores)

        val_dataset = CustomDataset(val_vec, val_labels)
        val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle=True, num_workers = num_cores)

        model = BaseModel()
        model.eval()
        optimizer = torch.optim.Adam(params = model.parameters(), lr = args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,threshold_mode='abs',min_lr=1e-8, verbose=True)

        # wandb.init(
        #     entity="your name",
        #     project="your project name",
        #     name=print_now
        # )


        # wandb.config.update(args)

        train(model, optimizer, train_loader, val_loader, scheduler, device,args,0)

    elif args.use_roberta:
        df, test, label_encoder = make_roberta_data(args)
        df = df.reset_index(drop =True)

        config = AutoConfig.from_pretrained(args.PLM)
        config.model_name = args.model_name
        tokenizer = AutoTokenizer.from_pretrained(args.PLM)

        # train_df, val_df = train_test_split(df ,test_size = args.split_ratio , random_state = args.seed)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = args.seed)

        for i, (train_idx, valid_idx) in enumerate(skf.split(df,df['label'])):
            train_df = df.loc[train_idx]
            val_df = df.loc[valid_idx]

            train_df = train_df.reset_index(drop =True)
            val_df = val_df.reset_index(drop =True)

            train_dataset = RobertaDataset(
                train_df.reset_index(drop=True),
                tokenizer,
                args.max_input_length,
                'train' 
            )

            train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers = num_cores)

            val_dataset = RobertaDataset(
                val_df.reset_index(drop=True),
                tokenizer,
                args.max_input_length,
                'train' 
            )

            val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle=True, num_workers = num_cores)


            ###########
            #Model
            ###########
            if args.model_name in ["roberta_class", "roberta_dacon", "roberta_linear", "roberta_sds"]:
                model = RobertaLinear.from_pretrained(args.PLM, config = config)

            elif args.model_name in ["roberta_document_linear", "roberta_document_sds", "roberta_document_concat_hidden", "roberta_document_mean_max", "roberta_document_lstm", "roberta_document_weighted"]:
                model = RobertaDocument.from_pretrained(args.PLM, config = config)
            model.eval()


            ###########
            #optimizer
            ###########

            if args.optimizer_type == "adam":
                optimizer = torch.optim.Adam(params = model.parameters(), lr = args.lr)
            elif args.optimizer_type == "adamw":
                optimizer = torch.optim.AdamW(
                    params = model.parameters(),
                    lr = args.lr, 
                    betas = (0.9, 0.999),
                    eps = 1e-6,
                    weight_decay = 1e-2
                )

            ###########
            #Scheduler
            ###########
            if args.scheduler_type == 'reduce':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                     mode='min',
                      factor=0.5, 
                      patience=2,
                      threshold_mode='abs',
                      min_lr=1e-8, 
                      verbose=True
                )
            elif args.scheduler_type == 'lambda':
                scheduler = optim.lr_scheduler.LambdaLR(
                    optimizer = optimizer,
                    lr_lambda=lambda epoch: 0.95 ** epoch
                )       

            elif args.scheduler_type == 'linear':
                total_batch_ = len(train_loader)
                scheduler = get_scheduler(optimizer, args, total_batch_)

            elif args.scheduler_type == 'linear_custom':
                scheduler = get_linear_schedule_with_warmup(
                    optimizer = optimizer,
                    num_warmup_steps = 0.1 * args.epochs,
                    num_training_steps = args.epochs
                )

            args.print_name = f"k{i}_" + args.model_name

            # wandb.init(
            #     entity="your name",
            #     project="your project name",
            #     name=args.print_name
            # )

     
            # wandb.config.update(args)

            train(model, optimizer, train_loader, val_loader, scheduler, device,args,i)

            if not args.use_kfold:
                break
            
if __name__ == "__main__":
  main()