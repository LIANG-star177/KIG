import torch
import torch.nn as nn


class PromptEncoder(torch.nn.Module):
    def __init__(self, template_len, hidden_size, tokenizer, args):
        super().__init__()
        self.spell_length = template_len
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer
        self.args = args
        # ent embedding
        self.cloze_length = template_len
        self.cloze_mask = [[1] * self.cloze_length]
        self.cloze_mask = torch.LongTensor(self.cloze_mask).bool().to(args.device)

        self.seq_indices = torch.LongTensor(list(range(len(self.cloze_mask[0])))).to(args.device)
        # embedding
        self.embedding = torch.nn.Embedding(len(self.cloze_mask[0]), self.hidden_size).to(args.device)
        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=self.args.lstm_dropout,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        print("init prompt encoder...")

    def forward(self, embeddings=None):
        # if self.args.multi_prompt:
        #     input_embeds = self.embedding(self.seq_indices).unsqueeze(0).repeat(5,1,1)
        # else:
        if embeddings is None:
            input_embeds = self.embedding(self.seq_indices).unsqueeze(0)
        else:
            input_embeds = embeddings
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
        return output_embeds
