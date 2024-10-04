from enum import Enum
import torch
import torch.nn as nn

class ForwardType(Enum):
    SIMPLE = 0
    STACKED = 1
    CASCADE = 2
    GRADIENT = 3

class DynamicNet(object):
    def __init__(self, c0, lr, device=None):
        self.models = []
        self.c0 = c0
        self.lr = lr
        # Si device n'est pas spécifié, choisir 'cuda' si disponible, sinon 'cpu'
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.boost_rate = nn.Parameter(torch.tensor(lr, requires_grad=True, device=self.device))

    def add(self, model):
        self.models.append(model.to(self.device))

    def parameters(self):
        params = []
        for m in self.models:
            params.extend(m.parameters())

        params.append(self.boost_rate)
        return params

    def zero_grad(self):
        for m in self.models:
            m.zero_grad()

    def to_device(self, device=None):
        """Déplacer les modèles et les paramètres sur le device spécifié"""
        device = device if device else self.device
        self.boost_rate = self.boost_rate.to(device)
        for m in self.models:
            m.to(device)

    def to_eval(self):
        for m in self.models:
            m.eval()

    def to_train(self):
        for m in self.models:
            m.train(True)

    def forward(self, x):
        if len(self.models) == 0:
            return None, self.c0
        middle_feat_cum = None
        prediction = None
        with torch.no_grad():
            for m in self.models:
                if middle_feat_cum is None:
                    middle_feat_cum, prediction = m(x, middle_feat_cum)
                else:
                    middle_feat_cum, pred = m(x, middle_feat_cum)
                    prediction += pred
        return middle_feat_cum, self.c0 + self.boost_rate * prediction

    def forward_grad(self, x):
        if len(self.models) == 0:
            return None, self.c0
        middle_feat_cum = None
        prediction = None
        for m in self.models:
            if middle_feat_cum is None:
                middle_feat_cum, prediction = m(x, middle_feat_cum)
            else:
                middle_feat_cum, pred = m(x, middle_feat_cum)
                prediction += pred
        return middle_feat_cum, self.c0 + self.boost_rate * prediction

    @classmethod
    def from_file(cls, path, builder, device=None):
        d = torch.load(path, map_location=device)
        device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        net = DynamicNet(d['c0'], d['lr'], device=device)
        net.boost_rate = d['boost_rate'].to(device)
        for stage, m in enumerate(d['models']):
            submod = builder(stage)
            submod.load_state_dict(m)
            net.add(submod.to(device))
        return net

    def to_file(self, path):
        models = [m.state_dict() for m in self.models]
        d = {'models': models, 'c0': self.c0, 'lr': self.lr, 'boost_rate': self.boost_rate}
        torch.save(d, path)
