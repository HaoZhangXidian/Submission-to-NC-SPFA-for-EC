# coding: utf-8

from torch.nn import functional as F
import torch
import numpy as np
loss = torch.nn.CrossEntropyLoss()

class pgbn_Trainer(object):
    def __init__(self, PGBN_encoder,  PGBN_optimizer):
        self.PGBN_encoder = PGBN_encoder
        self.PGBN_optimizer = PGBN_optimizer
        self.Num_noupdate = 0

    def pgbn_model_trainstep(self, BOW_minibatch, y_minibatch_T, MBratio, MBObserved):
        toogle_grad(self.PGBN_encoder, True)
        self.PGBN_encoder.train()
        self.PGBN_optimizer.zero_grad()


        Theta, prob = self.PGBN_encoder(BOW_minibatch)
        Likelihood = torch.sum(BOW_minibatch * torch.log(torch.mm(Theta, self.PGBN_encoder.Phi.t())) - torch.mm(Theta, self.PGBN_encoder.Phi.t()) - torch.lgamma(BOW_minibatch + 1))
        classification_loss = loss(prob, y_minibatch_T)

        Loss = -0.001*Likelihood + classification_loss
        Loss.backward()

        nan_num = 0
        for p in self.PGBN_encoder.parameters():
            nan_num += torch.sum(torch.isnan(p.grad))
            if nan_num > 0:
                break

        if nan_num == 0:
            self.PGBN_optimizer.step()
        else:
            print('Not update')
            self.Num_noupdate += 1

        self.PGBN_encoder.updatePhi(MBratio, MBObserved, BOW_minibatch, Theta)

        return Likelihood.item(), classification_loss.item()

    def test(self, BOW_minibatch):
        Theta, prob = self.PGBN_encoder(BOW_minibatch)

        Theta = Theta.to('cpu').detach().numpy()
        prob = prob.to('cpu').detach().numpy()

        return Theta, prob

# Utility functions
def toogle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)







