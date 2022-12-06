import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import GPT2Config, GPT2Model 
import numpy as np
import sys 
sys.path.append("new_gpt") 
from model import GPT2LMHeadModel
from transformers import BertTokenizer

class CaPromptEncoder(torch.nn.Module):
    '''
    Conditional Prompt Encoder which Generates Prompt Encodings conditioned on Utterance (using Encoder-Decoder)
    '''
    def __init__(self, template_len,  init_embedding, args):
        super().__init__()
        self.args = args
        self.num_trigs = template_len
        self.seq_indices = torch.arange(template_len).long() 
        self.config = GPT2Config.from_pretrained(args.pretrained_model_path)
        # self.config = GPT2Config.from_json_file(args.config_path)
        self.config.vocab_size = template_len
        self.config.n_positions = template_len
        self.config.n_ctx = template_len
        self.config.n_head = 12
        self.config.n_layer = 3
        self.config.add_cross_attention=True
        self.transformer = GPT2Model(self.config)
        self.transformer = self.transformer.to(self.args.device)
        with torch.no_grad():
            self.transformer.wte.weight[:template_len,:] = init_embedding.weight[5:template_len+5,:].data
            
    def forward(self, context_hiddens, context_attn_mask):
        device = context_attn_mask.device
        batch_size, maxlen = context_attn_mask.size()
        prompt_tokens = self.seq_indices[None, :].expand(batch_size, -1).to(device)
        output = self.transformer(input_ids = prompt_tokens, encoder_hidden_states=context_hiddens, encoder_attention_mask=context_attn_mask)
        # a list of num_layers tensors, with each of a size [2, batch_size, num_heads, seq_len, embed_size_per_head]
        return output.past_key_values 


class Distill_Tuning(nn.Module):
    '''
    Conditional Promt Tuning for Dialogue.
    Adopted from PADA: https://arxiv.org/pdf/2102.12206.pdf
    Issue1: padding tokens in LSTM input?
    Issue2: target length is set to 20 at most, but the generation step can generate 30 token at most.
    '''
    def __init__(self, args):
        super(Distill_Tuning, self).__init__()   
        self.args = args
        self.spell_length = self.args.template_len
        self.model = GPT2LMHeadModel.from_pretrained(args.pretrained_model_path)
        self.tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True)
        
        self.model = self.model.to(self.args.device)
        for param in self.model.parameters():
            param.requires_grad = self.args.use_lm_finetune
            
        # get model's embeddings
        self.embeddings = self.model.get_input_embeddings()
        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        self.pseudo_token_id = self.tokenizer.convert_tokens_to_ids(self.args.pseudo_token)
        
    
        print(f"number of basic parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
        if self.args.multi_prompt:
            self.prompt_encoder = []
            for i in range(5):
                self.prompt_encoder.append(CaPromptEncoder(self.spell_length, self.embeddings, args))
            self.prompt_encoder = nn.ModuleList(self.prompt_encoder).to(self.args.device)
        else:
            self.prompt_encoder = CaPromptEncoder(self.spell_length, self.embeddings, args)
        # self.prompt_encoder = CaPromptEncoder(self.spell_length,  self.embeddings, self.args)
        print(f"number of additional parameters: {sum(p.numel() for p in self.prompt_encoder.parameters() if p.requires_grad)}")
        
    def init_weights(self, m):# Initialize Linear Weight for GAN
        if isinstance(m, nn.Linear):        
            m.weight.data.uniform_(-0.08, 0.08)#nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.)  
    
    def get_query_head(self, x_h, prompt_tokens, x_t = None, flag=None):
        prompt_tensor_head = torch.tensor(prompt_tokens* (self.spell_length)).to(self.args.device)
        trans_inputs = []
        token_type_inputs = []
        content_id = self.tokenizer.convert_tokens_to_ids("[Content]")
        title_id = self.tokenizer.convert_tokens_to_ids("[Title]")
        index_musk =  (x_h == self.tokenizer.pad_token_id).type(torch.uint8) # only calculte the token which is not eos
        valid_number_length = torch.sum(index_musk, 1)
        for index, seq in zip(valid_number_length, x_h):
            if flag=="pre":
                pos = seq.tolist().index(self.tokenizer.sep_token_id)+1
                padding_len = self.args.max_len - self.spell_length - len(seq[:pos])
                padding_tensor =  torch.tensor([self.tokenizer.pad_token_id] * padding_len).to(self.args.device)
                trans_inputs.append(torch.cat([seq[:pos], prompt_tensor_head, padding_tensor]))

                padding_content = torch.tensor([content_id] * pos).to(self.args.device)
                padding_title = torch.tensor([title_id] * self.spell_length).to(self.args.device)
                token_type_inputs.append(torch.cat([padding_content, padding_title, padding_tensor]))
            else:
                trans_inputs.append(torch.cat([seq[:1], prompt_tensor_head, seq[1:]]))
        
        if flag=="pre":
            res = torch.stack(trans_inputs, dim=0)
            res_type = torch.stack(token_type_inputs, dim=0)
            return res, res_type
        else:
            res = torch.stack(trans_inputs, dim=0)
            return res

    def get_att_mask(self, src):
        src_mask = (src != self.tokenizer.pad_token_id)
        return src_mask

    def forward(self, input_ids=None, past=None, token_type_ids=None, labels=None, title_id=None, claim_label2 = None):
        self.train()
        batch_size, max_seq_len = input_ids.size()
        # prompt_tokens = [self.pseudo_token_id]
        # queries = self.get_query_head(input_ids, prompt_tokens)

        # mask
        att_mask = self.get_att_mask(input_ids)

        labels = torch.clone(input_ids)
        labels.masked_fill_(att_mask==0, -100)

        position_ids = att_mask.long().cumsum(-1)- 1
        position_ids.masked_fill_(att_mask == 0, 0)

        context_attn_mask = att_mask.clone()
        context_attn_mask[labels>0]=0
        context_encoding = self.model.transformer(input_ids, None, context_attn_mask)
        if self.args.multi_prompt:
            past_key_values_all = []
            for i, prompt_enc in enumerate(self.prompt_encoder):
                past_key_values_all.append(prompt_enc(context_encoding[0], context_attn_mask))
            past_key_values_all = torch.stack(past_key_values_all, dim=0)
            mask = claim_label2
            past_key_values_prompt = torch.sum(past_key_values_all * mask, dim=-1)
        else:
            past_key_values_prompt = self.prompt_encoder(context_encoding[0], context_attn_mask)

        prefix_attn = torch.ones(batch_size, self.spell_length).long().to(self.args.device)
        attention_mask = torch.cat((prefix_attn, att_mask), 1)


        # attention_mask = torch.cat([torch.ones([att_mask.shape[0], self.spell_length]).long().to(self.args.device), att_mask], dim=1)
        
        # token_type_ids
        # content_id = self.tokenizer.convert_tokens_to_ids("[Content]")
        # prompt_type_ids = torch.LongTensor([[content_id]*self.spell_length for _ in range(att_mask.shape[0])])
        # token_type_ids = torch.cat([prompt_type_ids.to(self.args.device), token_type_ids], dim=1)

        # # position_ids
        # position_ids = attention_mask.long().cumsum(-1)- 1
        # position_ids.masked_fill_(attention_mask == 0, 0)

        # labels = torch.clone(input_ids)
        # labels.masked_fill_(attention_mask==0, -100)

        # context_attn_mask = att_mask.clone()
        # context_attn_mask[labels>0]=0
        # context_encoding = self.model.transformer(input_ids, None, context_attn_mask)
        # past_key_values_prompt = self.prompt_encoder(context_encoding[0], context_attn_mask)

        # labels = torch.clone(queries)
        # labels.masked_fill_(attention_mask==0, -100)
        # labels.masked_fill_(queries == self.pseudo_token_id, -100)

        # prefix_attn = torch.ones(batch_size, self.spell_length).long().to(self.args.device)
        # attention_mask = torch.cat((prefix_attn, attention_mask), 1)

        
        transformer_outputs = self.model.transformer(input_ids=input_ids,
                past_key_values = past_key_values_prompt,
                attention_mask=attention_mask,
                position_ids=position_ids, 
                token_type_ids=token_type_ids)

        hidden_states = transformer_outputs[0]
        lm_logits = self.model.lm_head(hidden_states)
        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            if title_id is None or token_type_ids is None:
                raise Exception("当labels不为None时， title_id和token_type_ids均不可以为None。")
            mask = (token_type_ids == title_id).long()
            labels = labels * mask
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(ignore_index=0, reduction="sum")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # 获取title部分的真实长度，并计算真实loss
            num = shift_labels.ne(0).long().sum().item()
            loss = loss / num
            outputs = (loss,) + outputs
        return outputs    