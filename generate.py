import torch
import torch.nn.functional as F
import os
import argparse
import pickle
from tqdm import trange
from transformers import GPT2LMHeadModel, GPT2Config, CpmTokenizer
from torch.utils.data import Dataset, DataLoader
from utils import top_k_top_p_filtering, set_logger
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from tqdm import tqdm
from train import collate_fn
import pandas as pd
import json
from dataset import CPMDataset
from os.path import join, exists


def generate_next_token(input_ids):
    """
    对于给定的上文，生成下一个单词
    """
    outputs = model(input_ids=input_ids)
    logits = outputs.logits
    # next_token_logits表示最后一个token的hidden_state对应的prediction_scores,也就是模型要预测的下一个token的概率
    next_token_logits = logits[0, -1, :]
    next_token_logits = next_token_logits / args.temperature
    # 对于<unk>的概率设为无穷小，也就是说模型的预测结果不可能是[UNK]这个token
    next_token_logits[unk_id] = -float('Inf')
    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=args.topk, top_p=args.topp)
    # torch.multinomial表示从候选集合中选出无放回地进行抽取num_samples个元素，权重越高，抽到的几率越高，返回元素的下标
    next_token_id = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
    return next_token_id


def generate(max_len, title_ids):
    # 对title与context进行tokenize
    # title_ids = tokenizer.encode(title, add_special_tokens=False)
    # context_ids = tokenizer.encode(context, add_special_tokens=False)
    # title_ids = title_ids.tolist()
    # title_ids = title_ids[:title_ids.index(sep_id)]+ [sep_id]
    cur_len = len(title_ids)
    last_token_id = title_ids[-1]  # 已生成的内容的最后一个token
    input_ids = torch.tensor([title_ids], dtype=torch.long, device=device)

    while True:
        next_token_id = generate_next_token(input_ids[:, -args.context_len:])
        input_ids = torch.cat((input_ids, next_token_id.unsqueeze(0)), dim=1)
        cur_len += 1
        word = tokenizer.convert_ids_to_tokens(next_token_id.item())
        # if cur_len >= max_len:
        #     break
        # 超过最大长度，并且换行
        if cur_len >= max_len and last_token_id == 8 and next_token_id == 3:
            break
        # 超过最大长度，并且生成标点符号
        if cur_len >= max_len and word in [".", "。", "！", "!", "?", "？", ",", "，"]:
            break
        # 生成结束符
        if next_token_id == eod_id:
            break
    result = tokenizer.decode(input_ids.squeeze(0))
    result = result.split("<sep>")[1]
    return result

def load_data(test_path):
    source=[]
    with open("data/new_split/test.json",'r',encoding="utf-8") as f:
        for line in f:
            source.append((json.loads(line)))
    df=pd.DataFrame(source)
    view = df["view"][:20]
    result_all = []
    for i in range(len(df)):
        tmp=""
        pos=[]
        cur_len=0
        for el in df["claim"][i]:
            tmp+=el
            cur_len+=len(el)
            pos.append(cur_len)
        df["claim"][i]=tmp
        df["fact"][i] = df["fact"][i] + df["claim"][i]

    for i in tqdm(range(20)):
        title = df["fact"][i].replace("\n","").replace(" ","").replace("xx","")   # 取出标题
        title_ids = tokenizer.encode(title, add_special_tokens=False)
        token_ids = title_ids + [sep_id]
        result = generate(args.max_len, token_ids)
        result_all.append(result)

    # with open(test_path, "rb") as f:
    #     train_list = pickle.load(f)[:20]
    # test_dataset = CPMDataset(train_list, args.max_len)
    # test_dataloader = DataLoader(
    # test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn,
    # drop_last=False
    # )
    # for batch_idx, (input_ids, labels) in enumerate(test_dataloader):
    #     input_ids = input_ids.to(device)
    #     labels = labels.to(device)
        # generate(args.max_len, input_ids)
    return view, result_all

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


if __name__ == '__main__':
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='1', type=str, required=False, help='生成设备')
    parser.add_argument('--temperature', default=1, type=float, required=False, help='生成温度')
    parser.add_argument('--topk', default=0, type=int, required=False, help='最高几选一')
    parser.add_argument('--topp', default=0.85, type=float, required=False, help='最高积累概率')
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False, help='重复惩罚参数')
    parser.add_argument('--context_len', default=200, type=int, required=False, help='每一步生成时，参考的上文的长度')
    parser.add_argument('--max_len', default=900, type=int, required=False, help='生成的最长长度')
    parser.add_argument('--log_path', default='log/generate.log', type=str, required=False, help='日志存放位置')
    parser.add_argument('--no_cuda', action='store_true', help='不使用GPU进行预测')
    parser.add_argument('--model_path', type=str, default='model/epoch11', help='模型存放位置')
    parser.add_argument('--batch_size', default=32, type=int, required=False, help='test的batch size')
    parser.add_argument('--num_workers', type=int, default=0, help="dataloader加载数据时使用的线程数量")
    # parser.add_argument('--title', type=str, default='徜徉在书籍的阳光世界', help='作文标题')
    # parser.add_argument('--context', type=str, default='一本书是一个人的眼睛，它可以让你看到另一个世界的奇妙', help='作文上文')
    parser.add_argument('--title', type=str, default='家乡的四季', help='作文标题')
    parser.add_argument('--context', type=str, default='家乡的四季,最美不过了', help='作文上文')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
    args.cuda = torch.cuda.is_available() and not args.no_cuda  # 当用户使用GPU,并且GPU可用时
    device = 'cuda:0' if args.cuda else 'cpu'
    # device = 'cpu'

    # 创建日志对象
    logger = set_logger(args.log_path)

    # 初始化tokenizer
    tokenizer = CpmTokenizer(vocab_file="vocab/chinese_vocab.model")
    eod_id = tokenizer.convert_tokens_to_ids("<eod>")  # 文档结束符
    sep_id = tokenizer.sep_token_id
    unk_id = tokenizer.unk_token_id

    # 加载模型
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.eval()
    model = model.to(device)

    title = args.title
    context = args.context
    # result_all = []
    # logger.info("title:{}".format(title))
    # logger.info("context:{}".format(context))

    # # 开始生成
    # result = generate(args.max_len)
    # result = result.split("<sep>")[1]
    # logger.info("result:{}\n".format(result))
    view, result_all= load_data("data/test.pkl")
    metric_sum_lst = calculate(result_all, view, "log/metric.txt")
    print("*"*10+"bleu"+"*"*10)
    print("bleu1:{0:.4f},bleu2:{1:.4f},bleun:{2:.4f}".format(
        metric_sum_lst["b1"], metric_sum_lst["b2"], metric_sum_lst["bn"]))
    print("*"*10+"rouge"+"*"*10)
    print("rouge1:{0:.4f},rouge2:{1:.4f},rougel:{2:.4f}".format(
        metric_sum_lst["r1"], metric_sum_lst["r2"], metric_sum_lst["rl"]))
    # print("*"*10+"ans"+"*"*10)
    # print("pred_ans:{0:.4f}".format(pred_ans)) 

    # 通过控制台循环生成
    # print('开始生成，输入CTRL + Z以退出')
    # while True:
    #     try:
    #         # 用户输入title与context
    #         title = input("请输入作文标题：")
    #         context = input("请输入作文起始句子：")
    #
    #         logger.info("title:{}".format(title))
    #         logger.info("context:{}".format(context))
    #
    #         # 开始生成
    #         result = generate(args.max_len)
    #         result = result.split("<sep>")[1]
    #         logger.info("result:{}\n".format(result))
    #         break
    #
    #     except KeyboardInterrupt:
    #         break


