import json
import os
import pickle
import random
import re

import jieba
import numpy
import pandas as pd
import torch
import torch.nn as nn
from mytokenizer import MyTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from Transformer_sent_model import Transformer
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_fscore_support, jaccard_score
from setting import HID_DIM

RANDOM_SEED = 19
torch.manual_seed(RANDOM_SEED)

#用的时候改改属性即可，中文的需要bert来tokenizer
class myDataset(Dataset):
    def __init__(self, view, claim_label2):
        self.claim_label2 = torch.LongTensor(claim_label2)
        self.view = view

    def __getitem__(self, idx):
        #为了匹配bert_tokenizer的返回结果
        return {
                "claim_label2": self.claim_label2[idx],
                "view":
                {
                    "input_ids": self.view["input_ids"][idx],
                    "token_type_ids": self.view["token_type_ids"][idx],
                    "attention_mask": self.view["attention_mask"][idx],
                }
                }

    def __len__(self):
        return len(self.claim_label2)

def get_split_dataset(idx, view, claim_label2):
    #此处根据ID返回对应的数据，因为dataset需要根据id进行划分
    claim_label2_cur=pd.Series(claim_label2)[idx].tolist()
    view_cur = {
        "input_ids": view["input_ids"][idx],
        "token_type_ids": view["token_type_ids"][idx],
        "attention_mask": view["attention_mask"][idx],
    }
    return myDataset(view_cur, claim_label2_cur)

def one_hot_labels(labels_index, arg_map):
    label=[0]*len(arg_map)
    for item in labels_index:
        label[int(item)] = 1
    return label

def load_pred_view(pred_sents, claim_label2, tokenizer, maps, max_len):
    view = tokenizer(pred_sents, return_tensors="pt",padding="max_length", max_length=max_len, truncation=True)
    muclaim_label=[[] for _ in range(len(pred_sents))]
    for i in range(len(pred_sents)):
        muclaim_label[i]=one_hot_labels([maps["muclaim2idx"][el] for el in claim_label2[i]], maps["muclaim2idx"])
    idx = list(range(len(pred_sents)))
    dataset = get_split_dataset(idx, view, muclaim_label)
    return dataset

def get_one_hot(pred):
    zero = torch.zeros_like(pred)
    one = torch.ones_like(pred)
    threshold = 0
    pred = torch.where(pred <= threshold, zero, pred)
    pred = torch.where(pred > threshold, one, pred)
    return pred

def cal_feedback(pred_lst, trg_lst, flag=None):
    score_lst = []
    for i in range(len(pred_lst)):
        pred = set(np.nonzero(pred_lst[i]).squeeze(1).tolist())
        trg = set(np.nonzero(trg_lst[i]).squeeze(1).tolist())
        score_lst.append(len(pred.intersection(trg))/len(trg))
    return score_lst

def calculate_threshold2(pred,label):
    zero = torch.zeros_like(pred)
    one = torch.ones_like(pred)
    threshold = 0
    pred = torch.where(pred <= threshold, zero, pred)
    pred = torch.where(pred > threshold, one, pred)
    # label, pred = self.get_multi(label, pred)
    s_p, s_r, s_f, _ =precision_recall_fscore_support(label.detach().cpu().numpy(), pred.detach().cpu().numpy(), average="micro")
    m_p, m_r, m_f, _ =precision_recall_fscore_support(label.detach().cpu().numpy(), pred.detach().cpu().numpy(), average="macro")
    s_j=jaccard_score(label.detach().cpu().numpy(), pred.detach().cpu().numpy(), average="micro")
    m_j=jaccard_score(label.detach().cpu().numpy(), pred.detach().cpu().numpy(), average="macro")
    return s_p, s_r, s_f, s_j,m_p, m_r, m_f, m_j, pred

def cal_ans(pred, label_data, args):
    mytokenizer=MyTokenizer(args.embedding_path)
    padding_idx=mytokenizer.get_pad_idx()
    maps = {}
    with open(args.claim_map_path) as f:
        c2i = json.load(f)
        maps["muclaim2idx"] = c2i
        maps["idx2muclaim"] = {v: k for k, v in c2i.items()}
    eva_model=Transformer(input_dim=mytokenizer.vocab_size,hid_dim=HID_DIM,
                max_len=args.generate_max_len,pad_idx=padding_idx,maps=maps).to(args.device)
    eva_model.load_state_dict(torch.load(args.disc_model_path))
    eva_model.eval()

    pred_view_set=load_pred_view(pred, label_data, mytokenizer, maps, max_len=args.generate_max_len)
    pred_view_iter=DataLoader(pred_view_set,batch_size=16,shuffle=False,drop_last=False)
    tq = tqdm(pred_view_iter)
    score_sum = 0
    test_out = []
    for data in tq:
        for k in data:
            if type(data[k]) is dict:
                for k2 in data[k]:
                    data[k][k2] = data[k][k2].to(args.device)
            else:
                data[k] = data[k].to(args.device)
        with torch.no_grad():
            label_logits = eva_model(data["view"])
            test_out.append((label_logits["muclaim_label"],data["claim_label2"]))
        label_one_hot = get_one_hot(label_logits["muclaim_label"])

        feedback_score = cal_feedback(label_one_hot, data["claim_label2"])
        score_sum += sum(feedback_score)
        for i in range(len(feedback_score)):
            with open(args.ans_filename,"a") as f:
                f.write(str(feedback_score[i])+"\t"+str(label_one_hot[i])+"\t"+str(data["claim_label2"][i])+"\n")
    pred=torch.cat([i[0] for i in test_out])
    truth=torch.cat([i[1] for i in test_out])
    s_p, s_r, s_f, s_j,m_p, m_r, m_f, m_j, pred = calculate_threshold2(pred,truth)

    print("*"*10+"muclaim_micro"+"*"*10)
    print("test_precision:{0:.4f},test_recall:{1:.4f},test_F1:{2:.4f},test_jaccard:{3:.4f}".format(s_p, s_r, s_f, s_j))

    print("*"*10+"muclaim_macro"+"*"*10)
    print("test_precision:{0:.4f},test_recall:{1:.4f},test_F1:{2:.4f},test_jaccard:{3:.4f}".format(m_p, m_r, m_f, m_j))

    return score_sum/len(pred)
