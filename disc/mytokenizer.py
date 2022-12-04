from gensim.models import Word2Vec
import torch
from tqdm import tqdm
import numpy as np
import jieba
import re

class MyTokenizer:
    #往word2vec生成的word2id中加入特殊token，加在最前面
    def __init__(self,embedding_path) -> None:
        model=Word2Vec.load(embedding_path)
        # self.special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
        #一开始此处出错是因为将罪名id加入到词表前，但其实词表中本身已经有与罪名一样的词汇，造成重叠，id2word和word2id不匹配
        #现在的方法：在每个罪名后加入"[#]"用于标记罪名，避免重复.
        self.special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
        self.PAD_IDX=0
        # self.special_tokens = list(maps["charge2idx"].keys())+["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
        self.id2word=self.special_tokens+model.wv.index_to_key
        self.word2id=model.wv.key_to_index
        #其他对应的id往后移动特殊token总数的长度
        for k in self.word2id.keys():
            self.word2id[k]+=len(self.special_tokens)
        #特殊token对应id加在最前
        for i in range(len(self.special_tokens)):
            # if self.special_tokens[i] in self.word2id:

            self.word2id[self.special_tokens[i]]=i

        #这个size在模型中会用到
        self.vocab_size = len(self.word2id)
        #更改新的向量size
        self.vector_size=model.wv.vector_size
        special_token_size=np.zeros((len(self.special_tokens),self.vector_size))
        self.vectors=model.wv.vectors
        self.vectors=np.concatenate((special_token_size,self.vectors))

    def load_embedding(self):
        return self.vectors
    
    def __call__(self, *args, **kwds):
        return self.encode(*args, **kwds)

    def encode(self,sents,max_length=512,return_tensors="ls",padding="max_length",truncation=True):
        input_ids=[]
        token_type_ids=[]
        attention_mask=[]

        for sent in tqdm(sents):
            #将词映射到对应id
            sent=sent.replace(" ", "")
            sent=jieba.lcut(sent)
            sent=[self.word2id[w] if w in self.word2id.keys() else self.word2id["[UNK]"] for w in sent]
            #句子开头加上SOS，句尾加上EOS
            sent=[self.word2id["[SOS]"]]+sent+[self.word2id["[EOS]"]]
            mask=[1]*len(sent)+[self.PAD_IDX]*max_length
            #padding
            sent+=[self.PAD_IDX]*max_length
            #截取
            sent=sent[:max_length]

            input_ids.append(sent)
            token_type_ids.append([self.PAD_IDX]*max_length)
            attention_mask.append(mask[:max_length])

        if return_tensors=="pt":
            input_ids = torch.LongTensor(input_ids)
            token_type_ids = torch.LongTensor(token_type_ids)
            attention_mask = torch.LongTensor(attention_mask)
        
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask
        }
    #检验encode
    def decode(self,token_ids):
        res=[]
        for token in token_ids:
            sent=[]
            for id in token:
                if id==self.word2id["[EOS]"]:
                    break
                sent.append(self.id2word[id])
            sent=" ".join(sent)
            res.append(sent)
        return res
    
    def get_pad_idx(self):
        return self.PAD_IDX


if __name__=="__main__":
    tokenizer=MyTokenizer("gensim_train/word2vec.model")
    tokens=tokenizer.encode(["[伪造、倒卖]伪造的有价票证"])
    sents=tokenizer.decode(tokens)
    print(tokens)
    print(sents)


        



