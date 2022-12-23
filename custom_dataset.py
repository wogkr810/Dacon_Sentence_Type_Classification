import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from arguments import parse_args
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def make_data(args):
    args = parse_args()

    df = pd.read_csv(os.path.join(args.data_path,'train.csv'))
    # EDA 이후 중복, 미스라벨링 제거
    df = df.loc[df.ID != 'TRAIN_14989']
    df = df.loc[df.ID != 'TRAIN_03364']
    df = df.loc[df.ID != 'TRAIN_07099']
    df = df.loc[df.ID != 'TRAIN_02108']
    df = df.drop_duplicates('문장', keep = 'first')

    test = pd.read_csv(os.path.join(args.data_path,'test.csv'))

    # 제공된 학습데이터를 학습 / 검증 데이터셋으로 재 분할
    train, val, _, _ = train_test_split(df, df['label'], test_size=args.split_ratio, random_state=args.seed)

    # 1. 문장(Text) 벡터화 -> TfidfVectorizer
    vectorizer = TfidfVectorizer(min_df = 4, analyzer = 'word', ngram_range=(1, 2))
    vectorizer.fit(np.array(train["문장"]))

    train_vec = vectorizer.transform(train["문장"])
    val_vec = vectorizer.transform(val["문장"])
    test_vec = vectorizer.transform(test["문장"])

    print(train_vec.shape, val_vec.shape, test_vec.shape)

    # 2. Label Encoding (유형, 극성, 시제, 확실성)
    type_le = preprocessing.LabelEncoder()
    train["유형"] = type_le.fit_transform(train["유형"].values)
    val["유형"] = type_le.transform(val["유형"].values)

    polarity_le = preprocessing.LabelEncoder()
    train["극성"] = polarity_le.fit_transform(train["극성"].values)
    val["극성"] = polarity_le.transform(val["극성"].values)

    tense_le = preprocessing.LabelEncoder()
    train["시제"] = tense_le.fit_transform(train["시제"].values)
    val["시제"] = tense_le.transform(val["시제"].values)

    certainty_le = preprocessing.LabelEncoder()
    train["확실성"] = certainty_le.fit_transform(train["확실성"].values)
    val["확실성"] = certainty_le.transform(val["확실성"].values)

    train_type = train["유형"].values # sentence type
    train_polarity = train["극성"].values # sentence polarity
    train_tense = train["시제"].values # sentence tense
    train_certainty = train["확실성"].values # sentence certainty

    train_labels = {
        'type' : train_type,
        'polarity' : train_polarity,
        'tense' : train_tense,
        'certainty' : train_certainty
    }

    val_type = val["유형"].values # sentence type
    val_polarity = val["극성"].values # sentence polarity
    val_tense = val["시제"].values # sentence tense
    val_certainty = val["확실성"].values # sentence certainty

    val_labels = {
        'type' : val_type,
        'polarity' : val_polarity,
        'tense' : val_tense,
        'certainty' : val_certainty
    }

    label_encoder = {
        'type' : type_le,
        'polarity' : polarity_le,
        'tense' : tense_le,
        'certainty' : certainty_le

    }

    return train_vec, train_labels, val_vec, val_labels, test_vec, label_encoder

class CustomDataset(Dataset):
    def __init__(self, st_vec, st_labels):
        self.st_vec = st_vec
        self.st_labels = st_labels

    def __getitem__(self, index):
        st_vector = torch.FloatTensor(self.st_vec[index].toarray()).squeeze(0)
        if self.st_labels is not None:
            st_type = self.st_labels['type'][index]
            st_polarity = self.st_labels['polarity'][index]
            st_tense = self.st_labels['tense'][index]
            st_certainty = self.st_labels['certainty'][index]
            return st_vector, st_type, st_polarity, st_tense, st_certainty
        else:
            return st_vector

    def __len__(self):
        return len(self.st_vec.toarray())


def make_roberta_data(args):
    args = parse_args()

    df = pd.read_csv(os.path.join(args.data_path,'train.csv'))

    # EDA 이후 중복, 미스라벨링 제거
    df = df.loc[df.ID != 'TRAIN_14989']
    df = df.loc[df.ID != 'TRAIN_03364']
    df = df.loc[df.ID != 'TRAIN_07099']
    df = df.loc[df.ID != 'TRAIN_02108']
    df = df.drop_duplicates('문장', keep = 'first')
    
    test = pd.read_csv(os.path.join(args.data_path,'test.csv'))

    type_le = preprocessing.LabelEncoder()
    polarity_le = preprocessing.LabelEncoder()
    tense_le = preprocessing.LabelEncoder()
    certainty_le = preprocessing.LabelEncoder()

    df["유형"] = type_le.fit_transform(df["유형"].values)
    df["극성"] = polarity_le.fit_transform(df["극성"].values)
    df["시제"] = tense_le.fit_transform(df["시제"].values)
    df["확실성"] = certainty_le.fit_transform(df["확실성"].values)

    label_encoder = {
        'type' : type_le,
        'polarity' : polarity_le,
        'tense' : tense_le,
        'certainty' : certainty_le

    }

    return df, test, label_encoder

class RobertaDataset(Dataset):
    def __init__(self, df, tokenizer, max_length ,mode = 'train'):
        self.df_data = df
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __getitem__(self, index):
        sentence = self.df_data.loc[index, '문장']
        inputs = self.tokenizer(
            sentence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        if self.mode == "train":
            st_type = torch.tensor(self.df_data.loc[index, '유형'])
            st_polarity = torch.tensor(self.df_data.loc[index, '극성'])
            st_tense = torch.tensor(self.df_data.loc[index, '시제'])
            st_certainty = torch.tensor(self.df_data.loc[index, '확실성'])
            return input_ids, attention_mask, st_type, st_polarity, st_tense, st_certainty
        else:
            return input_ids, attention_mask

    def __len__(self):
        return len(self.df_data)