
import random
import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from arguments import parse_args
from custom_dataset import CustomDataset, make_data, make_roberta_data, RobertaDataset
from model import BaseModel, RobertaDocument, RobertaLinear

from datetime import datetime, timedelta

from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoConfig

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    
    type_preds, polarity_preds, tense_preds, certainty_preds = [], [], [], []
    
    with torch.no_grad():
        for input_ids, attention_mask in tqdm(test_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)  

            type_logit, polarity_logit, tense_logit, certainty_logit = model(input_ids, attention_mask)
            
            type_preds += type_logit.argmax(1).detach().cpu().numpy().tolist()
            polarity_preds += polarity_logit.argmax(1).detach().cpu().numpy().tolist()
            tense_preds += tense_logit.argmax(1).detach().cpu().numpy().tolist()
            certainty_preds += certainty_logit.argmax(1).detach().cpu().numpy().tolist()
            
    return type_preds, polarity_preds, tense_preds, certainty_preds

def main():
    num_cores = os.cpu_count()
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # train_vec, train_labels, val_vec, val_labels, test_vec, label_encoder = make_data(args)
    df, test,label_encoder = make_roberta_data(args)
    test = test.reset_index(drop =True)

    tokenizer = AutoTokenizer.from_pretrained(args.PLM)
    config = AutoConfig.from_pretrained(args.PLM)
    config.model_name = args.model_name

    test_dataset = RobertaDataset(
      test.reset_index(drop=True),
      tokenizer,
      args.max_input_length,
      None
    )
    test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, num_workers = num_cores)

    # test_dataset = CustomDataset(test_vec, None)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # load model
    if not os.path.exists(os.path.join(args.saved_path, args.model_name,f'k0_{args.model_name}','model_0_mean_f1.pth')):
      if args.model_name in ["roberta_class", "roberta_dacon", "roberta_linear", "roberta_sds"]:
          model = RobertaLinear.from_pretrained(args.PLM, config = config)
      elif args.model_name in ["roberta_document_linear", "roberta_document_sds", "roberta_document_concat_hidden", "roberta_document_mean_max", "roberta_document_lstm", "roberta_document_weighted"]:
          model = RobertaDocument.from_pretrained(args.PLM, config = config)
    else:
      model = torch.load(os.path.join(args.saved_path, args.model_name,f'k0_{args.model_name}','model_0_mean_f1.pth'))
    model.load_state_dict(torch.load(os.path.join(args.saved_path, args.model_name,f'k0_{args.model_name}','model(best_scores)_0_mean_f1.pth')))

    type_preds, polarity_preds, tense_preds, certainty_preds = inference(model, test_loader, device)

    type_le = label_encoder['type']
    polarity_le = label_encoder['polarity']
    tense_le = label_encoder['tense']
    certainty_le = label_encoder['certainty']

    type_preds = type_le.inverse_transform(type_preds)
    polarity_preds = polarity_le.inverse_transform(polarity_preds)
    tense_preds = tense_le.inverse_transform(tense_preds)
    certainty_preds = certainty_le.inverse_transform(certainty_preds)

    predictions = []
    for type_pred, polarity_pred, tense_pred, certainty_pred in zip(type_preds, polarity_preds, tense_preds, certainty_preds):
        predictions.append(type_pred+'-'+polarity_pred+'-'+tense_pred+'-'+certainty_pred)


    # 저장 관련
    now_date = datetime.now()
    diff_hours = timedelta(hours=9)
    now_date += diff_hours
    print_now = str(now_date.month) + '_' + str(now_date.day) + '_' + str(now_date.hour) + '_' + str(now_date.minute)

    if not os.path.exists(args.output_path):
      os.makedirs(args.output_path)

    submit = pd.read_csv(os.path.join(args.data_path,'sample_submission.csv'))
    submit['label'] = predictions

    submit.to_csv(os.path.join(args.output_path,'single_model', f'submission{print_now}.csv'), index=False)

    print("Inference Finish")
if __name__ == "__main__":
  main()