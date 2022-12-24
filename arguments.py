import argparse
import os
import torch

# baseline default hyper parameter

# CFG = {
#     'EPOCHS':10,
#     'LEARNING_RATE':1e-4,
#     'BATCH_SIZE':256,
#     'SEED':41
# }


def parse_args():
    parser = argparse.ArgumentParser()
    # =========================================================================
    # training args
    # =========================================================================
    parser.add_argument("--seed", default="41", type=int, help="Random Seed")
    parser.add_argument("--batch_size", default="64", type=int, help="train : # of batch size")
    parser.add_argument("--lr", default="3e-5", type=float, help="learning rate")
    parser.add_argument("--epochs", default="6", type=int, help="# of epochs")
    parser.add_argument("--split_ratio", default="0.2", type=float, help="train,test split ratio")
    parser.add_argument("--scheduler_type", default="reduce", type=str, help="type of learning rate scheduler : reduce,lambda, linear, linear_custom")
    parser.add_argument("--optimizer_type", default="adam", type=str, help="type of optimizer : adam,adamw")
    parser.add_argument("--warmup_steps", default="500", type=int, help="# of warmup steps")
    parser.add_argument("--use_kfold", default=True, type=bool, help="use k-fold")
    parser.add_argument("--print_name", default='zz', type=str, help="name of weights file")
    # =========================================================================
    # Path args
    # =========================================================================
    parser.add_argument("--data_path", default="./data/", type=str, help="Data Path")
    parser.add_argument("--output_path", default="./results", type=str, help="Output Path")
    parser.add_argument("--saved_path", default="./saved", type=str, help="Saved Path")
    # =========================================================================
    # Model args
    # =========================================================================
    parser.add_argument("--max_input_length", default="128", type=int, help="Max Input Length")
    parser.add_argument("--PLM", default="klue/roberta-large", type=str, help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--loss_name", default="cross_entropy", type=str, help="loss : cross_entropy, focal, f1, label_smoothing")

    parser.add_argument("--use_tfidf", default=False, type=bool, help="use tf-idf")
    parser.add_argument("--use_roberta", default=True, type=bool, help="use roberta")

    parser.add_argument(
        "--model_name", 
    default='roberta_document_weighted',
     type=str,
      help=
      "BaseModel : base " \
      "RobertaModel : roberta_class, roberta_dacon, roberta_linear, roberta_sds " \
      "RobertaDocument : roberta_document_linear, roberta_document_sds, roberta_document_concat_hidden, roberta_document_mean_max, roberta_document_lstm, roberta_document_weighted")

    # 주피터에서 사용할 경우 커널 에러 나기때문에, 밑에 있는 args=[] 로 선언해두면 편함
    # args = parser.parse_args()
    args = parser.parse_args(args=[])
    return args