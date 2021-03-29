import torch
import torch.nn as nn
import torch.nn.utils.rnn as tnur

class Classify(nn.Module):
    def __init__(self,
            features, classes,
            inproj_features=64, inproj_size=7, inproj_stride=4, inproj_norm=True,
            cell_type='lstm', hidden=64, layers=2, bidirectional=True,
            outproj_size=64,
            leak=0.0, dropout=0.2):
        super().__init__()
        self.cell_type = cell_type
        self.hidden = hidden
        self.bidirectional = bidirectional
        self.inproj_conv = seq.Conv(
            in_channels = features,
            out_channels = inproj_features,
            kernel_size = inproj_size,
            stride = inproj_stride,
            pad_delta = 1)
        self.inproj_norm = seq.BatchNorm(inproj_features) if inproj_norm else None
        self.inproj_act = nn.LeakyReLU(leak) if leak>0.0 else nn.ReLU()
        if cell_type == 'lstm':
            rnn = nn.LSTM
        elif cell_type == 'gru':
            rnn = nn.GRU
        elif cell_type == 'plain':
            rnn = nn.RNN
        else:
            raise ValueError('unknown cell_type: ' + cell_type)
        self.rnn = rnn(
            input_size = inproj_features,
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
        self.outproj_act = nn.LeakyReLU(leak) if leak>0.0 else nn.ReLU()
        self.outproj_lin2 = nn.Linear(
            in_features = outproj_size,
            out_features = classes)

    def forward(self, xs):
        N = torch.tensor([x.shape[0] for x in xs], device=xs[0].device)
        x = tnur.pad_sequence(xs, batch_first=True).transpose(1,2)
        h, N = self.inproj_conv(x, N)
        if self.inproj_norm is not None:
            h = self.inproj_norm(h, N)
        h = self.inproj_act(h)
        h = tnur.pack_padded_sequence(h.transpose(1,2), N, batch_first=True, enforce_sorted=False)
        h, _ = self.rnn(h)
        h, _ = tnur.pad_packed_sequence(h, batch_first=True)
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
