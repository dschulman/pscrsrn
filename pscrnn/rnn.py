import torch
import torch.nn as nn
import torch.nn.utils.rnn as tnur

class Classify(nn.Module):
    def __init__(self,
            features, classes,
            cell_type='lstm', hidden=64, layers=2, bidirectional=True,
            outproj_size=64, outproj_leak=0.0,
            dropout=0.0):
        super().__init__()
        self.cell_type = cell_type
        self.hidden = hidden
        self.bidirectional = bidirectional
        if cell_type == 'lstm':
            rnn = nn.LSTM
        elif cell_type == 'gru':
            rnn = nn.GRU
        elif cell_type == 'plain':
            rnn = nn.RNN
        else:
            raise ValueError('unknown cell_type: ' + cell_type)
        self.rnn = rnn(
            input_size = features,
            hidden_size = hidden,
            num_layers = layers,
            bidirectional = bidirectional,
            dropout = dropout)
        ## pytorch doesn't do LSTM forget gate bias init, handle it manually
        if cell_type == 'lstm':
            for name, param in self.rnn.named_parameters():
                if name.startswith('bias'):
                    l = param.shape[0]
                    param.data[l // 4 : l // 2].fill_(1.0)
        self.outproj_lin1 = nn.Linear(
            in_features = hidden * (6 if bidirectional else 3),
            out_features = outproj_size)
        self.outproj_act = nn.LeakyReLU(outproj_leak) if outproj_leak>0.0 else nn.ReLU()
        self.outproj_lin2 = nn.Linear(
            in_features = outproj_size,
            out_features = classes)

    def forward(self, xs):
        N = torch.tensor([x.shape[0] for x in xs], device=xs[0].device)
        x = tnur.pad_sequence(xs)
        h, _ = self.rnn(x)
        h = h.transpose(0, 1)
        hmean = torch.sum(h, dim=1) / N.unsqueeze(1)
        hmax = torch.max(h, dim=1)[0]  ## TODO handle masking
        if self.bidirectional:
            nh = self.hidden
            hfirst = h[:,0,nh:]
            Nexpand = N.reshape(-1, 1, 1).expand(h.shape[0], 1, nh)
            hlast = torch.gather(h[:,:,:nh], 1, (Nexpand-1)).squeeze(1)
            h = torch.cat([hmean,hmax,hlast,hfirst], 1)
        else:
            Nexpand = N.reshape(-1, 1, 1).expand(h.shape[0], 1, h.shape[2])
            hlast = torch.gather(h, 1, (Nexpand-1)).squeeze(1)
            h = torch.cat([hmean,hmax,hlast], 1)
        h = self.outproj_lin1(h)
        h = self.outproj_act(h)
        return self.outproj_lin2(h)
