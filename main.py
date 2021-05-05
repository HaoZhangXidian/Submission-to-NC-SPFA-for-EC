# check by Chaos 2019/10/29
import time
import pickle
import yaml
import torch
from build_model import build_pgbn_models
from train import pgbn_Trainer
from Optimizer import build_optimizers
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, roc_auc_score
from random import shuffle
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
import scipy.io as sio

df_data = pd.read_csv(open('./data_AD.csv'), index_col=0) # The data path
df_data.fillna(0, inplace=True)
X = df_data.iloc[:, 22:1944].values # X is a N by D matrix, where N is the number of patients, and the D is the number of features (codes)
Y = df_data.iloc[:, 1].values # Y is a vector, representing the SAE

train_num = X.shape[0]

# config
config_path = 'pgbn.yaml'
with open(config_path, 'r') as f:
    config = yaml.load(f)

# cuda set
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = config['gpu']['device']
else:
    device = "cpu"

X_tensor = torch.from_numpy(X).float().to(device)

Iterationall = config['training']['epoch'] * (train_num // config['training']['batch_size'])
PGBN_encoder = build_pgbn_models(config, Iterationall)

PGBN_encoder = PGBN_encoder.to(device)

PGBN_optimizer = build_optimizers(PGBN_encoder, config)
trainer = pgbn_Trainer(PGBN_encoder, PGBN_optimizer)

Likelihood_all = []

MBratio = np.floor(train_num / config['training']['batch_size']).astype('int')

for epoch in range(config['training']['epoch']):
    print('Start epoch %d...' % epoch)
    MBt = 0
    Likelihood_epoch = 0
    class_epoch = 0

    indices = np.arange(X.shape[0])  # gets the number of rows
    shuffle(indices)
    shuffled_matrix = X[list(indices)]
    shuffled_Y = Y[list(indices)]

    tstart = time.time()

    for iteration in range(train_num // config['training']['batch_size']):

        bow_minibatch = shuffled_matrix[iteration * config['training']['batch_size']: (iteration + 1) * config['training']['batch_size']]
        y_minibatch = shuffled_Y[iteration * config['training']['batch_size']: (iteration + 1) * config['training']['batch_size']]

        MBObserved = (epoch * MBratio + MBt).astype('int')
        bow_minibatch_T = torch.from_numpy(bow_minibatch).float().to(device)
        y_minibatch_T = torch.from_numpy(y_minibatch).type(torch.LongTensor).to(device)

        Likelihood, classification_loss = trainer.pgbn_model_trainstep(bow_minibatch_T, y_minibatch_T, MBratio, MBObserved)
        Likelihood_epoch += Likelihood/config['training']['batch_size']
        class_epoch += classification_loss
        MBt += 1

    tend = time.time()

    print('[epoch %0d time %4f]  Likelihood_dataset = %.4f class_loss_dataset = %.4f' % (epoch, tend - tstart,  Likelihood_epoch/MBt, class_epoch))

    Theta, prob = trainer.test(X_tensor)
    pred = np.argmax(prob, axis=1)

    index = np.where(Y==0)[0]
    predict = pred[index]
    negative_acc = len(np.where(predict==0)[0])/len(index)

    index = np.where(Y == 1)[0]
    predict = pred[index]
    positive_acc = len(np.where(predict == 1)[0]) / len(index)

    all_acc = (pred==Y).sum()/len(Y)
    auc = roc_auc_score(Y, prob[:, 1])

    print('[epoch %0d]  negative_acc = %.4f positive_acc = %.4f all_acc = %.4f auc = %.4f' % (epoch,  negative_acc, positive_acc, all_acc, auc))











