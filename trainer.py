import numpy as np
import os

from sklearn.metrics import f1_score
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils.loss import FocalLoss, F1Loss, LabelSmoothingLoss 

from tqdm.auto import tqdm

import wandb

def train(model, optimizer, train_loader, val_loader, scheduler, device, args, k_index):
    model.to(device)
    
    if args.loss_name == "cross_entropy":
        criterion = {
            'type' : nn.CrossEntropyLoss().to(device),
            'polarity' : nn.CrossEntropyLoss().to(device),
            'tense' : nn.CrossEntropyLoss().to(device),
            'certainty' : nn.CrossEntropyLoss().to(device)
        }

    elif args.loss_name == "focal":
        criterion = {
            'type' : FocalLoss().to(device),
            'polarity' : FocalLoss().to(device),
            'tense' : FocalLoss().to(device),
            'certainty' : FocalLoss().to(device)
        }

    elif args.loss_name == "f1":
        criterion = {
            'type' : F1Loss(num_labels=4).to(device),
            'polarity' : F1Loss(num_labels=3).to(device),
            'tense' : F1Loss(num_labels=3).to(device),
            'certainty' : F1Loss(num_labels=2).to(device)
        }

    elif args.loss_name == "label_smoothing":
        criterion = {
            'type' : LabelSmoothingLoss(classes=4).to(device),
            'polarity' :LabelSmoothingLoss(classes=3).to(device),
            'tense' : LabelSmoothingLoss(classes=3).to(device),
            'certainty' : LabelSmoothingLoss(classes=2).to(device)
        }
        


    best_loss = 999999
    best_f1 = 0
    best_model_val_loss = None
    best_model_mean_f1 = None

    total_logs = defaultdict(list)

    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = []
        
        if args.use_roberta:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(f"epoch : {epoch} / {args.epochs} ")
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

            for input_ids, attention_mask, type_label, polarity_label, tense_label, certainty_label in tqdm(iter(train_loader)):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                type_label = type_label.to(device)
                polarity_label = polarity_label.to(device)
                tense_label = tense_label.to(device)
                certainty_label = certainty_label.to(device)
                
                optimizer.zero_grad()

                type_logit, polarity_logit, tense_logit, certainty_logit = model(input_ids,attention_mask)
                
                loss = 0.25 * criterion['type'](type_logit, type_label) + \
                        0.25 * criterion['polarity'](polarity_logit, polarity_label) + \
                        0.25 * criterion['tense'](tense_logit, tense_label) + \
                        0.25 * criterion['certainty'](certainty_logit, certainty_label)
                
                loss.backward()
                optimizer.step()
                # scheduler.step()
                # for learning_rate in scheduler.get_lr():
                #     wandb.log({"learning_rate": learning_rate})
                
                if torch.isnan(loss):
                    print('Loss NAN. Train finish.')
                    break
                
                train_loss.append(loss.item())
            
            train_lr = optimizer.param_groups[0]['lr']
            val_loss, val_type_f1, val_polarity_f1, val_tense_f1, val_certainty_f1 = validation(model, val_loader, criterion, device)
            mean_f1 = (val_type_f1 + val_polarity_f1 + val_tense_f1 + val_certainty_f1)/4
            print(f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}] 유형 F1 : [{val_type_f1:.5f}] 극성 F1 : [{val_polarity_f1:.5f}] 시제 F1 : [{val_tense_f1:.5f}] 확실성 F1 : [{val_certainty_f1:.5f}] 평균 F1 : [{mean_f1:.5f}] LR : [{train_lr}]')

            logs = {
                'Train Loss': np.mean(train_loss),
                'Train lr' : train_lr,
                'Val Loss' : val_loss,
                '유형 F1' : val_type_f1,
                '극성 F1' : val_polarity_f1,
                '시제 F1' : val_tense_f1,
                '확실성 F1' : val_certainty_f1,
                '평균' : mean_f1
            }

            for key, value in logs.items():
                total_logs[key].append(value)


            if scheduler is not None:
                scheduler.step(val_loss)

            # 그냥 모두 저장
            # torch.save(model.state_dict(), os.path.join(args.saved_path, f'model(best_scores)_{epoch}.pth'))
            # torch.save(model, os.path.join(args.saved_path, f'model_{epoch}.pth'))


            if not os.path.exists(os.path.join(args.saved_path, args.model_name, args.print_name)):
                os.makedirs(os.path.join(args.saved_path, args.model_name, args.print_name))
                
            # best 저장
            # val loss 기준
            # if best_loss > val_loss:
            #     best_loss = val_loss
            #     torch.save(model.state_dict(), os.path.join(args.saved_path, 'model(best_scores)_val_loss.pth'))
            #     torch.save(model, os.path.join(args.saved_path, 'model_val_loss.pth'))
            #     best_model_val_loss = model

            # mean f1 기준
            if best_f1 < mean_f1:
                best_f1 = mean_f1
                torch.save(model.state_dict(), os.path.join(args.saved_path, args.model_name, args.print_name,f'model(best_scores)_{k_index}_mean_f1.pth'))
                torch.save(model, os.path.join(args.saved_path, args.model_name, args.print_name, f'model_{k_index}_mean_f1.pth'))
                best_model_mean_f1 = model

            # wandb.log(logs)
        
    # wandb.finish()
                
    return best_model_val_loss, best_model_mean_f1


        # # forward가 달라서 따로 만들어둚. TFIDF + MLP 쓸거면 이거
        # elif args.use_tfidf:
        #     for sentence, type_label, polarity_label, tense_label, certainty_label in tqdm(iter(train_loader)):
        #             sentence = sentence.to(device)
        #             type_label = type_label.to(device)
        #             polarity_label = polarity_label.to(device)
        #             tense_label = tense_label.to(device)
        #             certainty_label = certainty_label.to(device)
                    
        #             optimizer.zero_grad()
                    
        #             type_logit, polarity_logit, tense_logit, certainty_logit = model(sentence)
                    
        #             loss = 0.25 * criterion['type'](type_logit, type_label) + \
        #                     0.25 * criterion['polarity'](polarity_logit, polarity_label) + \
        #                     0.25 * criterion['tense'](tense_logit, tense_label) + \
        #                     0.25 * criterion['certainty'](certainty_logit, certainty_label)
                    
        #             loss.backward()
        #             optimizer.step()
                    
        #             if torch.isnan(loss):
        #                 print('Loss NAN. Train finish.')
        #                 break
                    
        #             train_loss.append(loss.item())
                
        #         val_loss, val_type_f1, val_polarity_f1, val_tense_f1, val_certainty_f1 = validation(model, val_loader, criterion, device)
        #         print(f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}] 유형 F1 : [{val_type_f1:.5f}] 극성 F1 : [{val_polarity_f1:.5f}] 시제 F1 : [{val_tense_f1:.5f}] 확실성 F1 : [{val_certainty_f1:.5f}]')


        #         logs = {
        #             'Train Loss': np.mean(train_loss),
        #             'Train lr' : np.around(optimizer.param_groups[0]['lr'],5),
        #             'Val Loss' : val_loss,
        #             '유형 F1' : val_type_f1,
        #             '극성 F1' : val_polarity_f1,
        #             '시제 F1' : val_tense_f1,
        #             '확실성 F1' : val_certainty_f1
        #         }

        #         for key, value in logs.items():
        #             total_logs[key].append(value)


        #         if scheduler is not None:
        #             scheduler.step(val_loss)
                    
        #         if best_loss > val_loss:
        #             best_loss = val_loss
        #             torch.save(model.state_dict(), os.path.join(args.saved_path, 'model(best_scores).pth'))
        #             torch.save(model, os.path.join(args.saved_path, 'model.pth'))
        #             best_model = model

        #         wandb.log(logs)
            
        #     wandb.finish()
                    
        #     return best_model

def validation(model, val_loader, criterion, device):
    model.eval()
    val_loss = []
    
    type_preds, polarity_preds, tense_preds, certainty_preds = [], [], [], []
    type_labels, polarity_labels, tense_labels, certainty_labels = [], [], [], []
    
    
    with torch.no_grad():
        for input_ids, attention_mask, type_label, polarity_label, tense_label, certainty_label in tqdm(iter(val_loader)):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            type_label = type_label.to(device)
            polarity_label = polarity_label.to(device)
            tense_label = tense_label.to(device)
            certainty_label = certainty_label.to(device)
            
            type_logit, polarity_logit, tense_logit, certainty_logit = model(input_ids, attention_mask)
            
            loss = 0.25 * criterion['type'](type_logit, type_label) + \
                    0.25 * criterion['polarity'](polarity_logit, polarity_label) + \
                    0.25 * criterion['tense'](tense_logit, tense_label) + \
                    0.25 * criterion['certainty'](certainty_logit, certainty_label)
            
            val_loss.append(loss.item())
            
            type_preds += type_logit.argmax(1).detach().cpu().numpy().tolist()
            type_labels += type_label.detach().cpu().numpy().tolist()
            
            polarity_preds += polarity_logit.argmax(1).detach().cpu().numpy().tolist()
            polarity_labels += polarity_label.detach().cpu().numpy().tolist()
            
            tense_preds += tense_logit.argmax(1).detach().cpu().numpy().tolist()
            tense_labels += tense_label.detach().cpu().numpy().tolist()
            
            certainty_preds += certainty_logit.argmax(1).detach().cpu().numpy().tolist()
            certainty_labels += certainty_label.detach().cpu().numpy().tolist()
    
    type_f1 = f1_score(type_labels, type_preds, average='weighted')
    polarity_f1 = f1_score(polarity_labels, polarity_preds, average='weighted')
    tense_f1 = f1_score(tense_labels, tense_preds, average='weighted')
    certainty_f1 = f1_score(certainty_labels, certainty_preds, average='weighted')
    
    return np.mean(val_loss), type_f1, polarity_f1, tense_f1, certainty_f1