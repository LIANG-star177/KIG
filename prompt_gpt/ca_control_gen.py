# -*- coding:utf-8 -*-
# @project: GPT2-NewsTitle
# @filename: generate_title.py
# @author: 刘聪NLP
# @contact: logcongcong@gmail.com
# @time: 2020/12/16 16:29
"""
    文件说明：
    根据训练好的模型，进行新闻标题生成，预测文件
"""
from torch.utils.data import DataLoader, Dataset
import torch
import os
import argparse
from transformers import BertTokenizer
import torch.nn.functional as F
import copy
import sys 
import random
import numpy as np  
sys.path.append("disc") 
import pandas as pd
import json
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from dialog_tuning import Distill_Tuning
from disc_judge import cal_ans
from Transformer_sent_model import Transformer
from mytokenizer import MyTokenizer
from prompt_head import CaPromptHead


def set_args():
    """设置模型预测所需参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0', type=str, help='设置预测时使用的显卡,使用CPU设置成-1即可')
    parser.add_argument('--pretrained_model_path', default="log/checkpoint-41680", type=str, help='预训练的GPT2模型的路径')
    parser.add_argument('--prompt_model_path', default='log/ca_multi_mean_prompt_saved_model/checkpoint-20840', type=str, help='模型输出路径')
    parser.add_argument('--vocab_path', default='new_gpt/vocab.txt', type=str, help='词表，该词表为小词表，并增加了一些新的标记')
    parser.add_argument('--batch_size', default=1, type=int, help='生成标题的个数')
    parser.add_argument('--generate_max_len', default=300, type=int, help='生成标题的最大长度')
    parser.add_argument('--repetition_penalty', default=1.2, type=float, help='重复处罚率')
    parser.add_argument('--top_k', default=5, type=float, help='解码时保留概率最高的多少个标记')
    parser.add_argument('--top_p', default=0.95, type=float, help='解码时保留概率累加大于多少的标记')
    parser.add_argument('--max_len', type=int, default=900, help='输入模型的最大长度，要比config中n_ctx小')
    parser.add_argument('--test_file_path', default='data/new_split/test.json', type=str, help='新闻标题生成的测试数据')
    parser.add_argument('--test_num', default=500, type=int, help='测试数量')

    #判别器设置
    parser.add_argument('--disc_model_path', default='model/disc_model/model_9', type=str)
    parser.add_argument('--embedding_path', default='model/word2vec/word2vec.model', type=str)
    parser.add_argument('--claim_map_path', default="data/claim_l2i_multi.json", type=str)

    # prompt参数
    parser.add_argument("--multi_prompt", type=bool, default=True, help='是否多维prompt')
    parser.add_argument("--context_aware", type=bool, default=True, help='是否注意上下文')
    parser.add_argument("--use_lm_finetune", type=bool, default=False, help='是否finetune')
    parser.add_argument('--template_len', default=5, type=int, required=False,help='prompt长度')
    parser.add_argument("--pseudo_token", type=str, default='##🔥')
    parser.add_argument("--lstm_dropout", type=float, default=0.0)
 
    # 结果输出
    parser.add_argument('--ans_filename', default="log/pred_ans2.txt", type=str)
    parser.add_argument('--txt_filename', default="log/pred2.txt", type=str)
    parser.add_argument('--metric_filename', default="log/pred_metric2.txt", type=str)
    return parser.parse_args()


def top_k_top_p_filtering(logits, top_k, top_p, filter_value=-float("Inf")):
    """
    top_k或top_p解码策略，仅保留top_k个或累积概率到达top_p的标记，其他标记设为filter_value，后续在选取标记的过程中会取不到值设为无穷小。
    Args:
        logits: 预测结果，即预测成为词典中每个词的分数
        top_k: 只保留概率最高的top_k个标记
        top_p: 只保留概率累积达到top_p的标记
        filter_value: 过滤标记值

    Returns:

    """
    # logits的维度必须为2，即size:[batch_size, vocab_size]
    assert logits.dim() == 2
    # 获取top_k和字典大小中较小的一个，也就是说，如果top_k大于字典大小，则取字典大小个标记
    top_k = min(top_k, logits[0].size(-1))
    # 如果top_k不为0，则将在logits中保留top_k个标记
    if top_k > 0:
        # 由于有batch_size个预测结果，因此对其遍历，选取每个预测结果的top_k标记
        for logit in logits:
            indices_to_remove = logit < torch.topk(logit, top_k)[0][..., -1, None]
            logit[indices_to_remove] = filter_value
    # 如果top_p不为0，则将在logits中保留概率值累积达到top_p的标记
    if top_p > 0.0:
        # 对logits进行递减排序
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        # 对排序后的结果使用softmax归一化，再获取累积概率序列
        # 例如：原始序列[0.1, 0.2, 0.3, 0.4]，则变为：[0.1, 0.3, 0.6, 1.0]
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # 删除累积概率高于top_p的标记
        sorted_indices_to_remove = cumulative_probs > top_p
        # 将索引向右移动，使第一个标记也保持在top_p之上
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for index, logit in enumerate(logits):
            # 由于有batch_size个预测结果，因此对其遍历，选取每个预测结果的累积概率达到top_p的标记
            indices_to_remove = sorted_indices[index][sorted_indices_to_remove[index]]
            logit[indices_to_remove] = filter_value
    return logits

def one_hot_labels(labels_index, arg_map):
    label=[0]*len(arg_map)
    for item in labels_index:
        label[int(item)] = 1
    return label

def predict_one_sample(model, prompt_head, tokenizer, device, args, content, claim_label2 = None):
    """
    对单个样本进行预测
    Args:
        model: 模型
        tokenizer: 分词器
        device: 设备信息
        args: 配置项信息
        content: 新闻正文

    Returns:

    """
    # 对新闻正文进行预处理，并判断如果超长则进行截断
    if args.multi_prompt:
        maps = {}
        with open("data/claim_l2i_multi.json") as f:
            c2i = json.load(f)
            maps["muclaim2idx"] = c2i
            maps["idx2muclaim"] = {v: k for k, v in c2i.items()}
        claim_label2 = one_hot_labels([maps["muclaim2idx"][el] for el in claim_label2], maps["muclaim2idx"])
    content_tokens = tokenizer.tokenize(content)
    if len(content_tokens) > args.max_len - 3 - args.generate_max_len:
        content_tokens = content_tokens[:args.max_len - 3 - args.generate_max_len]
    # 获取content_id、title_id、unk_id、sep_id值
    content_id = tokenizer.convert_tokens_to_ids("[Content]")
    title_id = tokenizer.convert_tokens_to_ids("[Title]")
    unk_id = tokenizer.convert_tokens_to_ids("[UNK]")
    sep_id = tokenizer.convert_tokens_to_ids("[SEP]")
    # 将tokens索引化，变成模型所需格式
    content_tokens = ["[CLS]"] + content_tokens + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(content_tokens)
    # 将input_ids和token_type_ids进行扩充，扩充到需要预测标题的个数，即batch_size
    input_ids = [copy.deepcopy(input_ids) for _ in range(args.batch_size)]
    token_type_ids = [[content_id] * len(content_tokens) for _ in range(args.batch_size)]
    # 将input_ids和token_type_ids变成tensor
    input_tensors = torch.tensor(input_ids).long().to(device)
    token_type_tensors = torch.tensor(token_type_ids).long().to(device)

    next_token_type = torch.tensor([[title_id] for _ in range(args.batch_size)]).long().to(device)
    next_attention_mask = torch.tensor([[1] for _ in range(args.batch_size)]).long().to(device)
    # 用于存放每一步解码的结果
    generated = []
    # 用于存放，完成解码序列的序号
    finish_set = set()

    input_ids, past_key_values_prompt, attention_mask, position_ids, token_type_ids = prompt_head(input_ids=input_tensors, \
                                                                    token_type_ids=token_type_tensors, claim_label2 = claim_label2)
    with torch.no_grad():
        # 遍历生成标题最大长度
        for _ in range(args.generate_max_len):

            transformer_outputs = model.model.transformer(input_ids=input_ids,
                past_key_values = past_key_values_prompt,
                attention_mask=attention_mask,
                position_ids=position_ids, 
                token_type_ids=token_type_ids)

            hidden_states = transformer_outputs[0]
            lm_logits = model.model.lm_head(hidden_states)

            next_token_logits = lm_logits[:, -1, :]
            # 对batch_size进行遍历，将词表中出现在序列中的词的概率进行惩罚
            for index in range(args.batch_size):
                for token_id in set([token_ids[index] for token_ids in generated]):
                    next_token_logits[index][token_id] /= args.repetition_penalty
            # 对batch_size进行遍历，将词表中的UNK的值设为无穷小
            for next_token_logit in next_token_logits:
                next_token_logit[unk_id] = -float("Inf")
            # 使用top_k_top_p_filtering函数，按照top_k和top_p的值，对预测结果进行筛选
            filter_logits = top_k_top_p_filtering(next_token_logits, top_k=args.top_k, top_p=args.top_p)
            # 对filter_logits的每一行做一次取值，输出结果是每一次取值时filter_logits对应行的下标，即词表位置（词的id）
            # filter_logits中的越大的值，越容易被选中
            next_tokens = torch.multinomial(F.softmax(filter_logits, dim=-1), num_samples=1)
            # 判断如果哪个序列的预测标记为sep_id时，则加入到finish_set
            for index, token_id in enumerate(next_tokens[:, 0]):
                if token_id == sep_id:
                    finish_set.add(index)
            # 判断，如果finish_set包含全部的序列序号，则停止预测；否则继续预测
            finish_flag = True
            for index in range(args.batch_size):
                if index not in finish_set:
                    finish_flag = False
                    break
            if finish_flag:
                break
            # 将预测标记添加到generated中
            generated.append([token.item() for token in next_tokens[:, 0]])
            # 将预测结果拼接到input_tensors和token_type_tensors上，继续下一次预测
            
            input_ids = torch.cat([input_ids, next_tokens], dim=1)
            attention_mask = torch.cat((attention_mask, next_attention_mask), dim=-1)
            next_pos_ids = torch.tensor([[input_ids.shape[1]]]).long().to(device)
            position_ids = torch.cat((position_ids, next_pos_ids), dim=-1)
            token_type_ids = torch.cat((token_type_ids, next_token_type), dim=-1)
        # 用于存储预测结果
        candidate_responses = []
        # 对batch_size进行遍历，并将token_id变成对应汉字
        for index in range(args.batch_size):
            responses = []
            for token_index in range(len(generated)):
                # 判断，当出现sep_id时，停止在该序列中添加token
                if generated[token_index][index] != sep_id:
                    responses.append(generated[token_index][index])
                else:
                    break
            # 将token_id序列变成汉字序列，去除"##"，并将[Space]替换成空格
            candidate_responses.append(
                "".join(tokenizer.convert_ids_to_tokens(responses)).replace("##", "").replace("[space]", " "))
    return candidate_responses

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

def save_decode(pred, trg, filename=None):
    with open(filename,"a") as f:
        for i in range(len(pred)):
            f.write(str(trg[i]).replace(" ","").replace("\n","")+"\t"+\
            str(pred[i]).replace(" ","").replace("\n","").replace(" ","")+"\n")

RANDOM_SEED = 2022

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    """主函数"""
    # 设置预测的配置参数
    setup_seed(RANDOM_SEED)
    args = set_args()
    # 获取设备信息
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICE"] = args.device
    # 获取device信息，用于模型训练
    device = torch.device("cuda:0" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")
    args.device = device
    # 实例化tokenizer和model
    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True)

    model = Distill_Tuning(args).to(device)
    model.prompt_encoder.load_state_dict(torch.load(args.prompt_model_path))
    model.eval()


    prompt_head = CaPromptHead(args, model.model, model.prompt_encoder)

    print('开始对新闻生成标题，输入CTRL + Z，则退出')
    source = []
    with open(args.test_file_path,'r',encoding="utf-8") as f:
        for line in f:
            source.append((json.loads(line)))
    df=pd.DataFrame(source)
    pred_all = []
    num = args.test_num if args.test_num!=-1 else len(df) #-1用所有数据
    for i in tqdm(range(num)):
        tmp=""
        pos=[]
        cur_len=0
        for el in df["claim"][i]:
            tmp+=el
            cur_len+=len(el)
            pos.append(cur_len)
        df["claim"][i]=tmp
        df["fact"][i] = df["fact"][i] + df["claim"][i] 
        if args.multi_prompt:
            titles = predict_one_sample(model, prompt_head, tokenizer, device, args, df["fact"][i].replace("\n","").replace(" ","").replace("xx",""), df["claim_label2"][i])
        else:
            titles = predict_one_sample(model, prompt_head, tokenizer, device, args, df["fact"][i].replace("\n","").replace(" ","").replace("xx",""))
        pred_all.append(titles[0])
    metric_sum_lst = calculate(pred_all, df["view"][:num], args.metric_filename)
    save_decode(pred_all, df["view"][:num], args.txt_filename)
    pred_ans = cal_ans(pred_all, df["claim_label2"][:num], args)
    print("*"*10+"bleu"+"*"*10)
    print("bleu1:{0:.4f},bleu2:{1:.4f},bleun:{2:.4f}".format(
        metric_sum_lst["b1"], metric_sum_lst["b2"], metric_sum_lst["bn"]))
    print("*"*10+"rouge"+"*"*10)
    print("rouge1:{0:.4f},rouge2:{1:.4f},rougel:{2:.4f}".format(
        metric_sum_lst["r1"], metric_sum_lst["r2"], metric_sum_lst["rl"]))
    print("*"*10+"ans"+"*"*10)
    print("pred_ans:{0:.4f}".format(pred_ans)) 


if __name__ == '__main__':
    main()

