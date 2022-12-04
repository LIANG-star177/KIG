#只是同时预测罪名、法条
import numpy
import torch
import torch.nn as nn
from setting import BATCH_SIZE, DEVICE, PRE_MODEL, PRETRAIN
from transformers import AutoModel, AutoTokenizer

# from utils.nn import LSTM, Linear
# 如果要按token使用disc, transformer不行，encoder看见了后面的信息。

class Encoder(nn.Module):
    def __init__(self,input_dim,hid_dim,n_layers,n_heads,encoder_output_dim,dropout,max_length=500) -> None:
        super().__init__()
        self.token_embedding=nn.Embedding(input_dim, hid_dim)
        self.pos_embedding=nn.Embedding(max_length,hid_dim)
        self.layers=nn.ModuleList([EncoderLayer(hid_dim,n_heads,encoder_output_dim,dropout) for _ in range(n_layers)])
        self.dropout=nn.Dropout(dropout)
        self.hid_dim = hid_dim
        
    def forward(self, src, src_mask):
        batch_size=src.shape[0]
        src_len=src.shape[1]
        scale=torch.sqrt(torch.FloatTensor([self.hid_dim])).to(DEVICE)
        pos=torch.arange(0,src_len).unsqueeze(0).repeat(batch_size,1).to(DEVICE)#batch_size*seq_len
        src=self.dropout((self.token_embedding(src)*scale)+self.pos_embedding(pos))#batch_size*seq_len*hid_dim
        for layer in self.layers:
            src=layer(src, src_mask)
        return src

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, encoder_output_dim, dropout):
        super().__init__()        
        #self_attn的layer norm 和feed forward的layer norm
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, encoder_output_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len]，这里的mask是padding mask
        #self_attn,分别分配给QKV,再layer norm
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))  
        #feed_forward,再layer norm     
        _src = self.positionwise_feedforward(src)       
        src = self.ff_layer_norm(src + self.dropout(_src))       
        #src = [batch size, src len, hid dim]       
        return src

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()       
        assert hid_dim % n_heads == 0       
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        #每一头的hidden_dim为总的hid_dim/头数，
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim) 
        self.fc_k_2 = nn.Linear(hid_dim*2, hid_dim*2)
        self.fc_v_2 = nn.Linear(hid_dim*2, hid_dim*2)          
        self.fc_o = nn.Linear(hid_dim, hid_dim)        
        self.dropout = nn.Dropout(dropout)      
        
        
    def forward(self, query, key, value, mask = None):     
        scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(DEVICE)   
        batch_size = query.shape[0]     
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]            
        Q = self.fc_q(query)
        if query.shape[-1]==key.shape[-1]:
            K = self.fc_k(key)
            V = self.fc_v(value)
        else:
            K = self.fc_k_2(key)
            V = self.fc_v_2(value)     
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]   
        # permute函数交换维度位置,这一步扩展了Q,K,V的维度为多头，不同head有不同的QKV          
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)   
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]   
        # 对K转置后点乘，结果很大，这样会导致softmax梯度很小，除以hid_dim来让方差为1            
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / scale       
        #energy = [batch size, n heads, query len, key len]   
        # padding mask，把mask为0对应位置全部替换成无穷小，这样softmax时会变成0  
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)    
        #对分数做softmax，得到attn   
        attention = torch.softmax(energy, dim = -1)               
        #attention = [batch size, n heads, query len, key len]  
        # attn dropout后与V相乘得到logits             
        x = torch.matmul(self.dropout(attention), V)       
        #x = [batch size, n heads, query len, head dim] 
        # 将数据占用的内存块放在一起，这样才能用view      
        x = x.permute(0, 2, 1, 3).contiguous()       
        #x = [batch size, query len, n heads, head dim]  
        # 不同head的self_attn的hidden_dim聚合到一起
        x = x.view(batch_size, -1, self.hid_dim)       
        #x = [batch size, query len, hid dim]   
        # 经过全连接层得到输出    
        x = self.fc_o(x)       
        #x = [batch size, query len, hid dim]       
        return x, attention

#feed_forward的作用：①强化attn出来的结果，attn高的更高，低的更低，用relu进行激活，这样就需要layer_norm标准化到relu作用区。
# ②先将数据映射到高维空间（encoder_output_dim），再映射到低维空间(hid_dim)，这样模型更能抽象出来单词与单词间关系。
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, encoder_output_dim, dropout):
        super().__init__()       
        self.fc_1 = nn.Linear(hid_dim, encoder_output_dim)
        self.fc_2 = nn.Linear(encoder_output_dim, hid_dim)       
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):        
        #x = [batch size, seq len, hid dim]        
        x = self.dropout(torch.relu(self.fc_1(x)))        
        #x = [batch size, seq len, pf dim]        
        x = self.fc_2(x)        
        #x = [batch size, seq len, hid dim]        
        return x       

#decoder基本结构与encoder相同，不过需要加seq_mask
class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, decoder_output_dim, dropout, max_length = 500):
        super().__init__()           
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)       
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, decoder_output_dim, dropout) for _ in range(n_layers)])        
        self.fc_out = nn.Linear(hid_dim, output_dim)        
        self.dropout = nn.Dropout(dropout)   
        self.hid_dim = hid_dim
        
        
    def forward(self, trg, enc_src, trg_mask):        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]      
        scale = torch.sqrt(torch.FloatTensor([self.hid_dim])).to(DEVICE)        
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]       
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(DEVICE)                          
        #pos = [batch size, trg len]            
        trg = self.dropout((self.tok_embedding(trg) * scale) + self.pos_embedding(pos))               
        #trg = [batch size, trg len, hid dim]    
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask)        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]        
        output = self.fc_out(trg)     
        #output = [batch size, trg len, output dim]          
        return output, attention  

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, decoder_output_dim, dropout):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, decoder_output_dim, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask):       
        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]     
        #读取目标文本的self_attn
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)  
        #layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))       
        #trg = [batch size, trg len, hid dim]           
        #结合encoder输出和目标文本self_attn结果的self_attn,query是target，key和value是encoder输出
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src)       
        #layer norm 
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))                 
        #trg = [batch size, trg len, hid dim]     
        #feed forward
        _trg = self.positionwise_feedforward(trg)      
        #layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]     
        return trg, attention    

class Transformer(nn.Module):
    def __init__(self ,input_dim,hid_dim,max_len,pad_idx,maps):
        super().__init__()      
        self.fact_encoder = Encoder(input_dim=input_dim,hid_dim=hid_dim,n_layers=1,n_heads=8,
                        encoder_output_dim=2048,dropout=0.1,max_length=max_len)
        self.pad_idx = pad_idx
        if PRETRAIN:
            self.electra = AutoModel.from_pretrained("electra-small") 
        self.fc_claim = nn.Linear(hid_dim, len(maps["muclaim2idx"]))
        
    def el_enc(self, text):
        x = self.electra(text)
        out = x.last_hidden_state # [256]
        return out
        
    #padding mask
    def make_src_mask(self, src):       
        #src = [batch size, src len]  
        # 看当前id是不是不等于padding的id，如果等于就是0，不等于就是1，之后再扩两维    
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        #src_mask = [batch size, 1, 1, src len]
        return src_mask

    def forward(self,view):   
        view = view["input_ids"].to(DEVICE)        
        view_mask = self.make_src_mask(view)     
        if PRETRAIN:
            view_enc = self.el_enc(view)[:,0]
        else:
            view_enc = self.fact_encoder(view, view_mask)
            view_enc = torch.sum(view_enc, dim=1)
        #enc_src = [batch size, src len, hid dim]            
        out = self.fc_claim(view_enc)      
        return {"muclaim_label": out}