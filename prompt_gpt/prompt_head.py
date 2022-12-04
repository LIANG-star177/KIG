import torch
from torch.nn.utils.rnn import pad_sequence
from os.path import join
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import GPT2Config 
import numpy as np

import re
import datetime
import sys 
sys.path.append("new_gpt") 

from transformers import BertTokenizer

from prompt_encoder import PromptEncoder
from model import GPT2LMHeadModel

import torch
import torch.nn as nn
import torch.nn.functional as F

class PromptHead(torch.nn.Module):

    def __init__(self, args, model, prompt_encoder, premodel=None):
        super().__init__()
        self.args = args
        self.spell_length = self.args.template_len
        self.model = model
        self.prompt_encoder = prompt_encoder

        if self.args.context_aware:
            self.premodel = premodel
            for param in self.premodel.parameters():
                param.requires_grad = False
        # load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True)
            
        # get model's embeddings
        self.embeddings = self.model.get_input_embeddings()
        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        
        self.pseudo_token_id = self.tokenizer.convert_tokens_to_ids(self.args.pseudo_token)

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

    def embed_input_head(self, queries, claim_label2=None, pre_embeds=None):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding)

        blocked_indices = (queries == self.pseudo_token_id).type(torch.uint8).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1]  # bz
        
        
        if self.args.multi_prompt:
            if self.args.context_aware:
                replace_embeds = [[] for _ in range(bz)]
                for i, prompt_enc in enumerate(self.prompt_encoder):
                    for k in range(bz):
                        if bz!=1:
                            replace_embeds[k].append(prompt_enc(pre_embeds)[k])
                        else:
                            replace_embeds[k].append(prompt_enc(pre_embeds))
                for i in range(bz):
                    replace_embeds[i] = torch.stack(replace_embeds[i], dim=0).transpose(0,2)
            else:
                replace_embeds = []
                for i, prompt_enc in enumerate(self.prompt_encoder):
                    replace_embeds.append(prompt_enc())
                replace_embeds = torch.stack(replace_embeds, dim=0)
                replace_embeds = replace_embeds.transpose(0,2)

            for k,bidx in enumerate(range(bz)):
                mask = claim_label2[k]
                if self.args.context_aware:
                    new_replace_embeds = torch.sum(replace_embeds[k] * mask, dim=-1)
                else:
                    new_replace_embeds = torch.sum(replace_embeds * mask, dim=-1)
                # new_replace_embeds = new_replace_embeds/len(np.nonzero(claim_label2[k]))
                new_replace_embeds = new_replace_embeds.transpose(0,1)
                for i in range(self.spell_length):
                    raw_embeds[bidx, blocked_indices[bidx, i], :] = new_replace_embeds[i, :]
        else:
            replace_embeds = self.prompt_encoder()
            for bidx in range(bz):
                for i in range(self.spell_length):
                    raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        return raw_embeds
    
    def get_att_mask(self, src):
        src_mask = (src != self.tokenizer.pad_token_id)
        return src_mask
    

    def forward(self, input_ids=None, past=None, token_type_ids=None, labels=None, title_id=None, claim_label2 = None):

        prompt_tokens = [self.pseudo_token_id]
        queries = self.get_query_head(input_ids, prompt_tokens)

        # mask
        att_mask = self.get_att_mask(input_ids)
        attention_mask = torch.cat([torch.ones([att_mask.shape[0], self.spell_length]).long().to(self.args.device), att_mask], dim=1)
        
        # token_type_ids
        content_id = self.tokenizer.convert_tokens_to_ids("[Content]")
        prompt_type_ids = torch.LongTensor([[content_id]*self.spell_length for _ in range(att_mask.shape[0])])
        token_type_ids = torch.cat([prompt_type_ids.to(self.args.device), token_type_ids], dim=1)

        # position_ids
        position_ids = attention_mask.long().cumsum(-1)- 1
        position_ids.masked_fill_(attention_mask == 0, 0)
                
        # get embedded input
        if self.args.multi_prompt:
            if self.args.context_aware:
                pre_queries, pre_token_type_ids = self.get_query_head(input_ids, prompt_tokens, flag="pre")
                transformer_outputs = self.model.transformer(pre_queries, token_type_ids=pre_token_type_ids)
                hidden_states = transformer_outputs[0]
                pre_embeds = hidden_states[:,-self.spell_length:] #取上下文aware的pseudo token embeds
                inputs_embeds = self.embed_input_head(queries, claim_label2, pre_embeds)
            else:
                inputs_embeds = self.embed_input_head(queries, claim_label2)
        else:
            inputs_embeds = self.embed_input_head(queries)
        
        return inputs_embeds, attention_mask, position_ids, token_type_ids  



class CaPromptHead(torch.nn.Module):

    def __init__(self, args, model, prompt_encoder, premodel=None):
        super().__init__()
        self.args = args
        self.spell_length = self.args.template_len
        self.model = model
        self.prompt_encoder = prompt_encoder
        # load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(args.vocab_path, do_lower_case=True)
            
        # get model's embeddings
        self.embeddings = self.model.get_input_embeddings()
        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        
        self.pseudo_token_id = self.tokenizer.convert_tokens_to_ids(self.args.pseudo_token)

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
        batch_size, max_seq_len = input_ids.size()
        prompt_tokens = [self.pseudo_token_id]
        queries = self.get_query_head(input_ids, prompt_tokens)

        # mask
        att_mask = self.get_att_mask(input_ids)
        attention_mask = torch.cat([torch.ones([att_mask.shape[0], self.spell_length]).long().to(self.args.device), att_mask], dim=1)
        
        # token_type_ids
        content_id = self.tokenizer.convert_tokens_to_ids("[Content]")
        prompt_type_ids = torch.LongTensor([[content_id]*self.spell_length for _ in range(att_mask.shape[0])])
        token_type_ids = torch.cat([prompt_type_ids.to(self.args.device), token_type_ids], dim=1)

        # position_ids
        position_ids = attention_mask.long().cumsum(-1)- 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        context_attn_mask = attention_mask.clone()
        context_encoding = self.model.transformer(queries, None, context_attn_mask)
        past_key_values_prompt = self.prompt_encoder(context_encoding[0], context_attn_mask)

        prefix_attn = torch.ones(batch_size, self.spell_length).long().to(self.args.device)
        attention_mask = torch.cat((prefix_attn, attention_mask), 1)
        
        return queries, past_key_values_prompt, attention_mask, position_ids, token_type_ids  