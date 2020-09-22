#! /usr/bin/env python3

'''
1. Download and install 'anaconda'
2. Install pytorch 'conda install pytorch torchvision cudatoolkit=10.2 -c pytorch'  or goto 'https://pytorch.org/get-started/locally/' for more details.
3. Run this code
Purpose: The easiest GAN for beginners

'''

import numpy as np; import matplotlib.pyplot as plt
import torch;from torch.autograd import Variable; import torch.nn as nn; import torch.nn.functional as F
Tensor = torch.FloatTensor

numSample=1000; numEpoch=10000; num_DG=3; dimSample = 32; dimLatent=10; batchSize=128; bceLoss = torch.nn.BCELoss()


G = nn.Sequential(nn.Linear(dimLatent, 128),nn.LeakyReLU(0.2, inplace=True),nn.Linear(128, 256),nn.LeakyReLU(0.2, inplace=True),
  nn.Linear(256, 512),nn.LeakyReLU(0.2, inplace=True),nn.Linear(512, 1024),nn.LeakyReLU(0.2, inplace=True),nn.Linear(1024, dimSample),nn.Tanh())

D = nn.Sequential(nn.Linear(dimSample, 512),nn.LeakyReLU(0.2, inplace=True),nn.Linear(512, 256),nn.LeakyReLU(0.2, inplace=True),nn.Linear(256, 1),nn.Sigmoid())


dataSet=Variable(torch.sin(torch.mm(torch.linspace(1,1.2,numSample).unsqueeze(1),torch.linspace(-2.5,3.5,dimSample).unsqueeze(0))),requires_grad=False)
# dataSet=(dataSet-dataSet.min())/(dataSet.max()-dataSet.min())*2.-1.



optimG = torch.optim.Adam(G.parameters(),lr=0.0001)
# optimD = torch.optim.SGD(D.parameters(),lr=0.001)
optimD = torch.optim.Adam(D.parameters(),lr=0.0001)

lossDList=[]; lossGList=[]
for epoch in range(numEpoch):
    xReal=dataSet[torch.randint(0,numSample,(batchSize,)),:]    # xReal
    z = torch.normal(0, 1, size=(batchSize, dimLatent), requires_grad=False, out=None)   # latent noise
    yReal= Variable(Tensor(batchSize, 1).fill_(1.0), requires_grad=False)   # real labels
    yFake= Variable(Tensor(batchSize, 1).fill_(0.0), requires_grad=False)   # fake labels


    ## Training Discriminator
    optimD.zero_grad()
    lossD=(bceLoss(D(G(z)),yFake)+bceLoss(D(xReal),yReal))/2.    # lossG = BCE Loss
    lossD.backward(); optimD.step()

    ## Training Generator
    if epoch%num_DG==0:   # Repeat num_DG times training D and then run once training G
        optimG.zero_grad()
        lossG=-bceLoss(D(G(z)),yFake)    # lossG = -BCE Loss
        lossG.backward(); optimG.step()



    lossDList.append(lossD.item()); lossGList.append(lossG.item())
    if epoch%50==0:
        plt.figure('GAN'); plt.clf(); plt.subplot(2,1,1); plt.plot(lossDList,'g',label='lossD'); plt.plot(lossGList,'r',label='lossG'); plt.title('loss'); plt.legend()
        plt.subplot(2,2,3); plt.plot(xReal.numpy().T); plt.title('xReal')
        plt.subplot(2,2,4); plt.plot(G(z).detach().numpy().T); plt.title('xFake'); plt.pause(0.01)



plt.show()


#
