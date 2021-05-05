import torch.nn as nn
from torch.nn import functional as F
import torch
#from torch_geometric.nn import DenseGCNConv
import numpy as np
import PGBN_sampler
from torch.nn import Parameter


class PGBN(nn.Module):
    def __init__(self, voc_size, hDim, numTopic, device, Iterationall):
        super(PGBN, self).__init__()
        self.voc_size = voc_size
        self.hDim = hDim
        self.numTopic = numTopic
        self.device = device

        self.f1 = nn.Linear(voc_size, hDim)
        self.shape = nn.Linear(hDim, 1)
        self.scale = nn.Linear(hDim, numTopic)

        realmin = 2.2e-308
        Phi = 0.2 + 0.8 * np.random.rand(voc_size, self.numTopic)
        Phi = Phi / np.maximum(realmin, Phi.sum(0))
        self.Phi = torch.from_numpy(Phi).float().to(device)

        self.train_PGBN_step = -1

        self.pgbn_eta = 0.01

        Setting = {}
        Setting['Iterall'] = Iterationall
        Setting['tao0FR'] = 0
        Setting['kappa0FR'] = 0.9
        Setting['tao0'] = 20
        Setting['kappa0'] = 0.7
        Setting['epsi0'] = 1

        self.ForgetRate = np.power((Setting['tao0FR'] + np.linspace(1, Setting['Iterall'], Setting['Iterall'])),
                                   -Setting['kappa0FR'])
        epsit = np.power((Setting['tao0'] + np.linspace(1, Setting['Iterall'], Setting['Iterall'])), -Setting['kappa0'])
        self.epsit = Setting['epsi0'] * epsit / epsit[0]

        self.classifier = nn.Linear(numTopic, 2)




    def forward(self, x):
        h = F.softplus(self.f1(torch.log(1+x)))
        Theta_shape = torch.exp(self.shape(h))
        #Theta_shape = torch.clamp(Theta_shape, min=1e-3, max=1e3)
        Theta_scale = torch.exp(self.scale(h))

        Theta_shape.repeat([1, self.numTopic])

        Theta_e = torch.Tensor(x.shape[0], self.numTopic).uniform_(0,1).to(self.device)
        Theta = (Theta_scale * ((-torch.log(1 - Theta_e)) ** (1 / Theta_shape)))

        #Theta_mean = Theta_scale * torch.exp(torch.lgamma(1 + 1 / Theta_shape))


        Theta_norm = Theta / (torch.max(Theta, dim=1)[0]).unsqueeze(1)
        #prob = torch.nn.functional.softmax(self.classifier(Theta_norm))
        prob = self.classifier(Theta_norm)

        return Theta, prob


    def updatePhi(self, MBratio, MBObserved, X, Theta):
        Theta_np = Theta.t().to('cpu').detach().numpy()

        Xt = X.t().to('cpu').numpy()
        phi = self.Phi.to('cpu').numpy()
        Xt_to_t1, WSZS = PGBN_sampler.Multrnd_Matrix(Xt.astype('double'), phi.astype('double'), Theta_np.astype('double'))

        EWSZS = WSZS
        EWSZS = MBratio * EWSZS

        if (MBObserved == 0):
            self.NDot = EWSZS.sum(0)
        else:
            self.NDot = (1 - self.ForgetRate[MBObserved]) * self.NDot + self.ForgetRate[MBObserved] * EWSZS.sum(0)
        tmp = EWSZS + self.pgbn_eta
        tmp = (1 / self.NDot) * (tmp - tmp.sum(0) * phi)
        tmp1 = (2 / self.NDot) * phi

        tmp = phi + self.epsit[MBObserved] * tmp + np.sqrt(self.epsit[MBObserved] * tmp1) * np.random.randn(
            phi.shape[0], phi.shape[1])
        phi = PGBN_sampler.ProjSimplexSpecial(tmp, phi, 0)
        self.Phi = torch.from_numpy(phi).float().to(self.device)

def build_pgbn_models(config, Iterationall):

    PGBN_model = PGBN(voc_size = config['data']['vocab_size'],
                      hDim = config['model']['PGBN_h1'],
                      numTopic = config['model']['PGBN_numTopic1'],
                      device = config['gpu']['device'],
                      Iterationall = Iterationall)
    return PGBN_model
