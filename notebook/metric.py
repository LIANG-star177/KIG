from torch.utils.data import DataLoader, Dataset
import torch
import os
import random
import numpy as np
import argparse
from transformers import BertTokenizer
import torch.nn.functional as F
import copy
import sys 
sys.path.append("disc") 
import pandas as pd
import json
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from disc_judge import load_pred_view, get_one_hot, cal_feedback, calculate_threshold2
from Transformer_sent_model import Transformer
from mytokenizer import MyTokenizer

def calculate(pred, trg, filename):
    rouge = Rouge()
    bleuscore1, bleuscore2, bleuscoren = 0,0,0
    rougescore1, rougescore2, rougescorel = 0,0,0
    num = len(pred)
    for i in range(num):
        trg[i]=" ".join([w for w in trg[i].replace(" ","")])
        pred[i]=" ".join([w for w in pred[i].replace(" ","")])
        reference = [trg[i]]
        candidate = pred[i]
        rouge_score = rouge.get_scores(pred[i], trg[i])
        r1 = rouge_score[0]["rouge-1"]['r']
        r2 = rouge_score[0]["rouge-2"]['r']
        rl = rouge_score[0]["rouge-l"]['r']
        b1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
        b2 = sentence_bleu(reference, candidate,weights=(0, 1, 0, 0))
        bn = sentence_bleu(reference, candidate,weights=(0.25, 0.25, 0.25, 0.25))
        rougescore1 += r1
        rougescore2 += r2
        rougescorel += rl
        bleuscore1 += b1
        bleuscore2 += b2
        bleuscoren += bn
        with open(filename,"a") as f:
            f.write(str(b1)+"\t"+ str(b2) +"\t"+str(bn) +"\t"+str(r1) +"\t"+str(r2) +"\t"+str(rl)+"\n")
    bleuscore1, bleuscore2, bleuscoren =  bleuscore1/num, bleuscore2/num, bleuscoren/num
    rougescore1, rougescore2, rougescorel = rougescore1/num, rougescore2/num, rougescorel/num
    return {"b1": bleuscore1, "b2": bleuscore2, "bn": bleuscoren,
            "r1": rougescore1, "r2": rougescore2, "rl": rougescorel}

def cal_ans(pred, label_data, device, filename):
    mytokenizer=MyTokenizer('model/word2vec/word2vec.model')
    padding_idx=mytokenizer.get_pad_idx()
    maps = {}
    with open("data/claim_l2i_multi.json") as f:
        c2i = json.load(f)
        maps["muclaim2idx"] = c2i
        maps["idx2muclaim"] = {v: k for k, v in c2i.items()}
    eva_model=Transformer(input_dim=mytokenizer.vocab_size,hid_dim=256,
                max_len=300,pad_idx=padding_idx,maps=maps).to(device)
    eva_model.load_state_dict(torch.load('model/disc_model/model_9'))
    eva_model.eval()

    pred_view_set=load_pred_view(pred, label_data, mytokenizer, maps, max_len=300)
    pred_view_iter=DataLoader(pred_view_set,batch_size=16,shuffle=False,drop_last=False)
    tq = tqdm(pred_view_iter)
    score_sum = 0
    test_out = []
    for data in tq:
        for k in data:
            if type(data[k]) is dict:
                for k2 in data[k]:
                    data[k][k2] = data[k][k2].to(device)
            else:
                data[k] = data[k].to(device)
        with torch.no_grad():
            label_logits = eva_model(data["view"])
            test_out.append((label_logits["muclaim_label"],data["claim_label2"]))
        label_one_hot = get_one_hot(label_logits["muclaim_label"])

        feedback_score = cal_feedback(label_one_hot, data["claim_label2"])
        score_sum += sum(feedback_score)
        for i in range(len(feedback_score)):
            with open(filename,"a") as f:
                f.write(str(feedback_score[i])+"\t"+str(label_one_hot[i])+"\t"+str(data["claim_label2"][i])+"\n")
    pred=torch.cat([i[0] for i in test_out])
    truth=torch.cat([i[1] for i in test_out])
    s_p, s_r, s_f, s_j,m_p, m_r, m_f, m_j, pred = calculate_threshold2(pred,truth)
    print("*"*10+"muclaim_macro"+"*"*10)
    print("test_precision:{0:.4f},test_recall:{1:.4f},test_F1:{2:.4f},test_jaccard:{3:.4f}".format(m_p, m_r, m_f, m_j))
    print("*"*10+"muclaim_micro"+"*"*10)
    print("test_precision:{0:.4f},test_recall:{1:.4f},test_F1:{2:.4f},test_jaccard:{3:.4f}".format(s_p, s_r, s_f, s_j))

    return score_sum/len(pred)

    
model_name = "pure_GPT"
path = "notebook/output_json/{}.json".format(model_name)
br_path = "notebook/metric_log/{}.json".format(model_name)
ans_path = "notebook/ans_log/{}.json".format(model_name)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
res = json.load(open(path))
pred, truth,claim_label = res["pred"], res["truth"], res["claim_label"] 

metric_sum_lst = calculate(pred, truth, br_path)
pred_ans = cal_ans(pred, claim_label, device, ans_path)

print("*"*10+"ans"+"*"*10)
print("pred_ans:{0:.4f}".format(pred_ans)) 

print("==="*20+"相似度指标")
print("*"*10+"bleu"+"*"*10)
print("bleu1:{0:.4f},bleu2:{1:.4f},bleun:{2:.4f}".format(
    metric_sum_lst["b1"], metric_sum_lst["b2"], metric_sum_lst["bn"]))
print("*"*10+"rouge"+"*"*10)
print("rouge1:{0:.4f},rouge2:{1:.4f},rougel:{2:.4f}".format(
    metric_sum_lst["r1"], metric_sum_lst["r2"], metric_sum_lst["rl"]))