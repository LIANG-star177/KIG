import torch
import torch.nn as nn


class MaskableLSTM(nn.Module):
    '''
    pytorch rewrite lstm use mask
    '''
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional=True, batch_first=True):
        super(MaskableLSTM, self).__init__()   
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.dropout_in = 0.1
        self.dropout_out = 0.1
        self.bidirectional = bidirectional
        self.batch_first = batch_first

        self.fcells = nn.ModuleList([
            nn.LSTMCell(input_size, hidden_size) if l==0 else nn.LSTMCell(hidden_size, hidden_size)
            for l in range(num_layers)])
        self.bcells = nn.ModuleList([
            nn.LSTMCell(input_size, hidden_size) if l==0 else nn.LSTMCell(hidden_size, hidden_size)
            for l in range(num_layers)])

    @staticmethod
    def _forward_rnn(cell, input, masks, init_h, drop_masks):
        max_time = input.size(0) # seq_len:41
        output = []
        hx = init_h # ([32,200], [32,200]) initialized 0s
        for t in range(max_time):
            h_next, c_next = cell(input=input[t], hx = hx) 
            # input[t] to [32,100] (The so word position batch inside on), after h_n and c_n a lstmcell output are both (32,200)
            h_next = h_next*masks[t] + init_h[0]*(1-masks[t]) # masks (41,32,200), masks [t] is a (32,200) is 0 or 1,
            c_next = c_next*masks[t] + init_h[1]*(1-masks[t]) # here the last part should not have been behind it all 0?
            output.append(h_next) # 0-40 each position output
            if drop_masks is not None: h_next = h_next * drop_masks
            hx = (h_next, c_next) # put a h_n, c_n as arguments next lstmcell
        output = torch.stack (output, 0) # connection to the list (41,32,200)
        return output, hx # each output of the last column is connected to the h_n lstmcell output (41,32,200), and hx of (h_next, c_next)
 
    @staticmethod
    def _forward_brnn(cell, input, masks, init_h, drop_masks):
        max_time = input.size(0)
        output = []
        hx = init_h
        for t in reversed(range(max_time)):
            h_next, c_next = cell(input=input[t], hx=hx)
            h_next = h_next*masks[t] + init_h[0]*(1-masks[t])
            c_next = c_next*masks[t] + init_h[1]*(1-masks[t])
            output.append(h_next)
            if drop_masks is not None: h_next = h_next * drop_masks
            hx = (h_next, c_next)
        output.reverse()
        output = torch.stack(output, 0)
        return output, hx
 
    def forward(self, input, masks, init_h=None):
        if self.batch_first:
            input = input.transpose(0, 1)
            masks = masks.transpose(0, 1)
        max_time, batch_size, _ = input.size() # (41,32,100)
        masks = masks[:,:,None].expand(-1, -1, self.hidden_size)  # (41,32,200)
 
        if init_h is None:
            init_h = torch.zeros((batch_size, self.hidden_size), device=input.device)
            torch.nn.init.xavier_normal_(init_h)
            init_h = (init_h, init_h) # h_n, c_n initial; initial: (32,200)
        h_ns, c_ns = [], []
        hiddens_all_layers = []
        output_f, output_b, output = input, input, input
        for layer in range(self.num_layers):
            hidden_mask = None
            if self.training: # dropout
                max_time, batch_size, input_size = output_f.size()
                input_mask = output_f.new(batch_size, input_size).fill_(1-self.dropout_in)#new a matrix(32,100),filled with 0.7
                with torch.no_grad():
                    input_mask = torch.bernoulli(input_mask)#,requires_grad=False) # fill the Bernoulli 0 and 1
                input_mask = input_mask/(1-self.dropout_in) # 1 is changed from the fill value 1.4286
                input_mask = input_mask[:,:,None].expand(-1,-1,max_time).permute(2,0,1)#2 position in one dimension increasing, and MAX_TIME succession, then the result is sequentially switched (41,32,100)
                output_f = output_f*input_mask # (41,32,100)*(41,32,100) manually dropout of the input matrix
                output_b = output_b*input_mask # (41,32,100)*(41,32,100) manually dropout of the input matrix
                hidden_mask = output_f.new(batch_size, self.hidden_size).fill_(1-self.dropout_out)  # （32,200）
                with torch.no_grad():
                    hidden_mask = torch.bernoulli(hidden_mask)#, requires_grad=False)
                hidden_mask = hidden_mask/(1-self.dropout_out)
                
            output_f, (h_n_f, c_n_f) = MaskableLSTM._forward_rnn(cell=self.fcells[layer], \
                 input = output_f, masks = masks, init_h = init_h, drop_masks = hidden_mask) 
            if self.bidirectional:
                output_b, (h_n_b, c_n_b) = MaskableLSTM._forward_brnn(cell=self.bcells[layer], \
                    input=output_b, masks=masks, init_h=init_h, drop_masks=hidden_mask)
            h_n = torch.cat([h_n_f, h_n_b], 1) if self.bidirectional else h_n_f
            c_n = torch.cat([c_n_f, c_n_b], 1) if self.bidirectional else c_n_f
            output = torch.cat([output_f, output_b], 2) if self.bidirectional else output_f 
            h_ns.append (h_n) 
            c_ns.append (c_n) 
            hiddens_all_layers.append(output)
        h_n = torch.stack(h_ns, 0) # (#L,32,400)
        c_n = torch.stack(c_ns, 0)
        if self.batch_first:
            output=output.transpose(1,0)
            hiddens_all_layers = [h.transpose(1,0) for h in hiddens_all_layers]
        return output, (h_n, c_n, hiddens_all_layers) 