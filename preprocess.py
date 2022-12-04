import argparse
from utils import set_logger
from transformers.models.cpm.tokenization_cpm import CpmTokenizer
import os
import pickle
import json
import pandas as pd
from tqdm import tqdm


def preprocess():
    """
    对故事数据集进行预处理
    """
    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', default='vocab/chinese_vocab.model', type=str, required=False,
                        help='词表路径')
    parser.add_argument('--log_path', default='log/preprocess.log', type=str, required=False, help='日志存放位置')
    parser.add_argument('--data_path', default='data/new_split/train.json', type=str, required=False, help='数据集存放位置')
    parser.add_argument('--save_path', default='data/train.pkl', type=str, required=False, help='对训练数据集进行tokenize之后的数据存放位置')
    parser.add_argument('--win_size', default=200, type=int, required=False, help='滑动窗口的大小，相当于每条数据的最大长度')
    parser.add_argument('--step', default=200, type=int, required=False, help='滑动窗口的滑动步幅')
    args = parser.parse_args()

    # 初始化日志对象
    logger = set_logger(args.log_path)

    # 初始化tokenizer
    tokenizer = CpmTokenizer(vocab_file="vocab/chinese_vocab.model")
    eod_id = tokenizer.convert_tokens_to_ids("<eod>")   # 文档结束符
    sep_id = tokenizer.sep_token_id

    # 读取作文数据集目录下的所有文件
    train_list = []
    logger.info("start tokenizing data")
    source=[]
    with open(args.data_path,'r',encoding="utf-8") as f:
        for line in f:
            source.append((json.loads(line)))
    df=pd.DataFrame(source)
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

    maps = {}
    # c2i_path = "data/claim_l2i.json"
    # with open(c2i_path) as f:
    #     c2i = json.load(f)
    #     maps["claim2idx"] = c2i
    #     maps["idx2claim"] = {v: k for k, v in c2i.items()}
    
    with open("data/claim_l2i_multi.json") as f:
        c2i = json.load(f)
        maps["muclaim2idx"] = c2i
        maps["idx2muclaim"] = {v: k for k, v in c2i.items()}
    for i in tqdm(range(len(df))):
        title = df["fact"][i].replace("\n","").replace(" ","").replace("xx","")   # 取出标题
        article = df["view"][i].replace("\n","").replace(" ","").replace("xx","")
        titles = tokenizer.encode_plus(title, add_special_tokens=False)
        articles = tokenizer.encode_plus(article, add_special_tokens=False)
        title_ids,  = titles["input_ids"]

        token_ids = title_ids + [sep_id] + article_ids + [eod_id]
        # train_list.append(token_ids)

        # 对于每条数据，使用滑动窗口对其进行截断
        win_size = args.win_size
        step = args.step
        start_index = 0
        end_index = win_size
        data = token_ids[start_index:end_index]
        train_list.append(data)
        start_index += step
        end_index += step
        while end_index+50 < len(token_ids):  # 剩下的数据长度，大于或等于50，才加入训练数据集
            data = token_ids[start_index:end_index]
            train_list.append(data)
            start_index += step
            end_index += step

    # 序列化训练数据
    with open(args.save_path, "wb") as f:
        pickle.dump(train_list, f)


if __name__ == '__main__':
    preprocess()


