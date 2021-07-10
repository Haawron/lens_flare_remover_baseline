from model.baseline import  DaconBaseline
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.precision = args.precision
        self.cpu = args.cpu
        self.device = torch.device('cpu' if args.cpu else 'cuda')
        if args.model == 'baseline':
            self.model = DaconBaseline(args)
        if self.device == torch.device('cuda'):
            self.model = self.model.cuda()

    def forward(self, x):
        x = self.model(x)
        return x

