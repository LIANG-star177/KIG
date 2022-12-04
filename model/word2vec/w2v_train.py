import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import os
import numpy as np
import re
import jieba
from tqdm import tqdm
import random
from gensim.models import Word2Vec
import time
import json
import pickle

def text_cleaner(text):
    def load_stopwords(filename):
        stopwords = []
        with open(filename, "r", encoding="utf-8") as fr:
            for line in fr:
                line = line.replace("\n", "")
                stopwords.append(line)
        return stopwords

    stop_words = load_stopwords("data/stopwords.txt")

    rules = [
        {r'>\s+': u'>'},  # remove spaces after a tag opens or closes
        {r'\s+': u' '},  # replace consecutive spaces
        {r'\s*<br\s*/?>\s*': u'\n'},  # newline after a <br>
        # newline after </p> and </div> and <h1/>...
        {r'</(div)\s*>\s*': u'\n'},
        # newline after </p> and </div> and <h1/>...
        {r'</(p|h\d)\s*>\s*': u'\n\n'},
        {r'<head>.*<\s*(/head|body)[^>]*>': u''},  # remove <head> to </head>
        # show links instead of texts
        {r'<a\s+href="([^"]+)"[^>]*>.*</a>': r'\1'},
        {r'[ \t]*<[^<]*?/?>': u''},  # remove remaining tags
        {r'^\s+': u''}  # remove spaces at the beginning
    ]

    # 替换html特殊字符
    text = text.replace("&ldquo;", "“").replace("&rdquo;", "”")
    text = text.replace("&quot;", "\"").replace("&times;", "x")
    text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&sup3;", "")
    text = text.replace("&divide;", "/").replace("&hellip;", "...")
    text = text.replace("&laquo;", "《").replace("&raquo;", "》")
    text = text.replace("&lsquo;", "‘").replace("&rsquo;", '’')
    text = text.replace("&gt；", ">").replace(
        "&lt；", "<").replace("&middot;", "")
    text = text.replace("&mdash;", "—").replace("&rsquo;", '’')

    for rule in rules:
        for (k, v) in rule.items():
            regex = re.compile(k)
            text = regex.sub(v, text)
        text = text.rstrip()
        text = text.strip()
    text = text.replace('+', ' ').replace(',', ' ').replace(':', ' ')
    text = re.sub("([0-9]+[年月日])+", "", text)
    text = re.sub("[a-zA-Z]+", "", text)
    text = re.sub("[0-9\.]+元", "", text)
    stop_words_user = ["年", "月", "日", "时", "分", "许", "某", "甲", "乙", "丙"]
    word_tokens = jieba.cut(text)

    def str_find_list(string, words):
        for word in words:
            if string.find(word) != -1:
                return True
        return False

    text = [w for w in word_tokens if w not in stop_words if not str_find_list(w, stop_words_user)
            if len(w) >= 1 if not w.isspace()]
    return " ".join(text)

def confusing_data(text_clean):
    source=[]
    with open("data/new_split/total_data.json",'r',encoding="utf-8") as f:
        for line in f:
            source.append((json.loads(line)))
    dataset=pd.DataFrame(source)
    if text_clean:
        for i in tqdm(range(len(dataset))):
            dataset["fact"][i] = text_cleaner(dataset["fact"][i])
    fact = dataset["fact"]
    view = dataset["view"]
    for i in range(len(dataset)):
        tmp=""
        cur_len=0
        for el in dataset["claim"][i]:
            tmp+=el
            cur_len+=len(el)
        dataset["claim"][i]=tmp
    claim = dataset["claim"]
    fileTrainSeg=[]
    for i in tqdm(range(len(fact))):
        fileTrainSeg.append(jieba.lcut(fact[i]))
        fileTrainSeg.append(jieba.lcut(claim[i]))
        fileTrainSeg.append(jieba.lcut(view[i]))
    return fileTrainSeg
# 保存分词结果到文件中

def train_word2vec():
    sentences = confusing_data(text_clean=False)

    start_time = time.time()
    model = Word2Vec(sentences, vector_size = 300,max_vocab_size=50000, workers=4, min_count=2)
    end_time = time.time()
    print("cost time: ",end_time - start_time)

    model.save("gensim_train/word2vec.model")

def load_word2vec(file_path):
    model = Word2Vec.load(file_path)
    # a = 1
    print(len(model.wv.index_to_key))
    for key in model.wv.similar_by_word('借贷', topn =3):
        print(key)

train_word2vec()
load_word2vec("gensim_train/word2vec.model")

# with open("gensim_model.pkl", "wb") as f:
#     pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)