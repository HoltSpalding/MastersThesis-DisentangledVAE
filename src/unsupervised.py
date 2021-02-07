import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
from torch.nn import Parameter
import sys
sys.path.append('../')
import probtorch
from probtorch.util import expand_inputs
print('probtorch:', probtorch.__version__, 
      'torch:', torch.__version__, 
      'cuda:', torch.cuda.is_available())



# model parameters
#32 X 32 images
NUM_PIXELS_H = 32
NUM_PIXELS_W = 32
NUM_PIXELS = NUM_PIXELS_H * NUM_PIXELS_W #1024
NUM_HIDDEN = 256
#one class of image
NUM_DIGITS = 1
#TODO lower number of latent variables we learn
NUM_LATENT = 50   

# training parameters
NUM_SAMPLES = 8 
NUM_BATCH = 5
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
BETA1 = 0.90
EPS = 1e-9
CUDA = torch.cuda.is_available()

# path parameters
MODEL_NAME = 'bricks-%02ddim' % NUM_LATENT
#location of the images
DATA_PATH = '../us/'
WEIGHTS_PATH = '../models/upsupervisedweights'
RESTORE = False



class Encoder(nn.Module):
    def __init__(self, num_pixels=NUM_PIXELS, 
                       num_hidden=NUM_HIDDEN,
                       num_latent=NUM_LATENT,
                       num_batch=NUM_BATCH):
        super(self.__class__, self).__init__()
        self.enc_hidden = nn.Sequential( 
                            nn.Linear(num_pixels, num_hidden),
                            nn.ReLU())
        self.z_mean = nn.Linear(num_hidden, num_latent)
        self.z_log_std = nn.Linear(num_hidden, num_latent)
    
    @expand_inputs
    def forward(self, images, labels=None, num_samples=None):
        q = probtorch.Trace()
        hiddens = self.enc_hidden(images)
        q.normal(self.z_mean(hiddens),
                 self.z_log_std(hiddens).exp(),
                 name='z')
        return q

class Decoder(nn.Module):
    def __init__(self, num_pixels=NUM_PIXELS, 
                       num_hidden=NUM_HIDDEN,
                       num_latent=NUM_LATENT):
        super(self.__class__, self).__init__()
        self.z_mean = torch.zeros(num_latent)
        self.z_std = torch.ones(num_latent)
        self.dec_image = nn.Sequential(
                           nn.Linear(num_latent, num_hidden),
                           nn.ReLU(),
                           nn.Linear(num_hidden, num_pixels),
                           nn.Sigmoid())

    def forward(self, images, q=None, num_samples=None):
        p = probtorch.Trace()
        z = p.normal(self.z_mean, 
                     self.z_std,
                     value=q['z'],
                     name='z')
        images_mean = self.dec_image(z)
        p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x + 
                                  torch.log(1 - x_hat + EPS) * (1-x)).sum(-1),
               images_mean, images, name='x')
        return p



def elbo(q, p, alpha=0.1):
    if NUM_SAMPLES is None:
        return probtorch.objectives.montecarlo.elbo(q, p, sample_dim=None, batch_dim=0, alpha=alpha)
    else:
        return probtorch.objectives.montecarlo.elbo(q, p, sample_dim=0, batch_dim=1, alpha=alpha)


#######################################################
#TODO LOAD DATA 
from torchvision import datasets, transforms
import os 
import PIL

def PIL_Loader(filename):
    return PIL.Image.open(filename).convert("L")

if not os.path.isdir(DATA_PATH):
    os.makedirs(DATA_PATH)

IMG_FOLDER = datasets.ImageFolder(DATA_PATH, transform=transforms.ToTensor(), loader=PIL_Loader)

train_data = torch.utils.data.DataLoader(IMG_FOLDER,batch_size=NUM_BATCH, shuffle=False) 
test_data = torch.utils.data.DataLoader(IMG_FOLDER, batch_size=NUM_BATCH, shuffle=False) 


# train_data = torch.utils.data.DataLoader(

#                 datasets.MNIST(DATA_PATH, train=True, download=True,
#                                transform=transforms.ToTensor()),
#                 batch_size=NUM_BATCH, shuffle=True) 


# test_data = torch.utils.data.DataLoader(
#                 datasets.MNIST(DATA_PATH, train=False, download=True,
#                                transform=transforms.ToTensor()),
#                 batch_size=NUM_BATCH, shuffle=True)     




############################################################
def cuda_tensors(obj): 
    for attr in dir(obj):
        value = getattr(obj, attr)
        if isinstance(value, torch.Tensor):
            setattr(obj, attr, value.cuda())

enc = Encoder()
dec = Decoder()
if CUDA:
    enc.cuda()
    dec.cuda()
    cuda_tensors(enc)
    cuda_tensors(dec)

optimizer =  torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),
                              lr=LEARNING_RATE,
                              betas=(BETA1, 0.999))



def train(data, enc, dec, optimizer):
    epoch_elbo = 0.0
    enc.train()
    dec.train()
    N = 0
    for b, (images, labels) in enumerate(data):
        if images.size()[0] == NUM_BATCH:
            N += NUM_BATCH
            images = images.view(-1, NUM_PIXELS)
            if CUDA:
                images = images.cuda()
            optimizer.zero_grad()
            q = enc(images, num_samples=NUM_SAMPLES)
            p = dec(images, q, num_samples=NUM_SAMPLES)
            loss = -elbo(q, p)
            loss.backward()
            optimizer.step()
            if CUDA:
                loss = loss.cpu()
            epoch_elbo -= float(loss.item())
    return epoch_elbo / N

def test(data, enc, dec):
    enc.eval()
    dec.eval()
    epoch_elbo = 0.0
    N = 0
    for b, (images, labels) in enumerate(data):
        if images.size()[0] == NUM_BATCH:
            N += NUM_BATCH
            images = images.view(-1, NUM_PIXELS)
            if CUDA:
                images = images.cuda()
            q = enc(images, num_samples=NUM_SAMPLES)
            p = dec(images, q, num_samples=NUM_SAMPLES)
            batch_elbo = elbo(q, p)
            if CUDA:
                batch_elbo = batch_elbo.cpu()
            epoch_elbo += float(batch_elbo.item())
    return epoch_elbo / N

ELBO_vals = []

import time
from random import random
if not RESTORE:
    mask = {}
    for e in range(NUM_EPOCHS):
        train_start = time.time()
        train_elbo = train(train_data, enc, dec, optimizer)
        train_end = time.time()
        test_start = time.time()
        test_elbo = test(test_data, enc, dec)
        test_end = time.time()
        ELBO_vals.append(train_elbo)
        print('[Epoch %d] Train: ELBO %.4e (%ds) Test: ELBO %.4e (%ds)' % (
                e, train_elbo, train_end - train_start, 
                test_elbo, test_end - test_start))

    if not os.path.isdir(WEIGHTS_PATH):
        os.mkdir(WEIGHTS_PATH)
    torch.save(enc.state_dict(),
               '%s/%s-%s-%s-enc.rar' % (WEIGHTS_PATH, MODEL_NAME, probtorch.__version__, torch.__version__))
    torch.save(dec.state_dict(),
               '%s/%s-%s-%s-dec.rar' % (WEIGHTS_PATH, MODEL_NAME, probtorch.__version__, torch.__version__))

if RESTORE:
    enc.load_state_dict(torch.load('%s/%s-%s-%s-enc.rar' % (WEIGHTS_PATH, MODEL_NAME, probtorch.__version__, torch.__version__)))
    dec.load_state_dict(torch.load('%s/%s-%s-%s-dec.rar' % (WEIGHTS_PATH, MODEL_NAME, probtorch.__version__, torch.__version__)))


import numpy as np
ys = []
zs = []
for (x, y) in test_data:
    if len(x) == NUM_BATCH:
        images = x.view(-1, NUM_PIXELS)
        if CUDA:
            q = enc(images.cuda())
            z = q['z'].value.cpu().detach().numpy()
        else:
            q = enc(images)
            z = q['z'].value.data.detach().numpy()
        zs.append(z)
        ys.append(y.numpy())
ys = np.concatenate(ys,0)
zs = np.concatenate(zs,0)


# run TSNE when number of latent dims exceeds 2
if NUM_LATENT > 2:
    from sklearn.manifold import TSNE
    zs2 = TSNE().fit_transform(zs)
    zs2_mean = zs2.mean(0)
    zs2_std = zs2.std(0)
else:
    zs2 = zs


x,_ = next(iter(train_data))
x_var = x.view(-1, NUM_PIXELS)
if CUDA:
    q = enc(x_var.cuda())
    p = dec(x_var.cuda(), q)
    x_mean = p['x'].value.view(NUM_BATCH, NUM_PIXELS_W, NUM_PIXELS_H).data.cpu().numpy()
else:
    q = enc(x_var)
    p = dec(x_var, q)
    x_mean = p['x'].value.view(NUM_BATCH, NUM_PIXELS_W, NUM_PIXELS_H).data.numpy().squeeze()
    
# fig = plt.figure(figsize=(12,5.25))
fig = plt.figure(figsize=(12,5.25))

for k in range(5):
    ax = plt.subplot(2, 5, k+1)

    ax.imshow(x[k].squeeze(),cmap="gray")
    if k == 0:
        ax.set_title("Original", y = 0.4, x = -0.1, rotation = 90)
    plt.axis("off")

    ax = plt.subplot(2, 5, k+6)
    if k == 0:
        ax.set_title("Reconstruction", y = 0.3, x = -0.1, rotation = 90)
    ax.imshow(x_mean[k].squeeze(), cmap="gray")
    plt.axis("off")

    # ax = plt.subplot(3, 5, k+11)
    # if k == 0:
    #     ax.set_title("StdDev", y = 0.4, x = -0.1, rotation = 90)
    # # print(x[k])
    # # print(x[k].shape)
    # ax.imshow(x[k].squeeze())
    # plt.axis("off")



fig.tight_layout()
plt.suptitle("Image Reconstruction, %d Epochs, %d Latent Variables" % (NUM_EPOCHS,NUM_LATENT), fontsize=18, y = 1)
plt.savefig("recon_ep%d_lv%d.png" % (NUM_EPOCHS,NUM_LATENT))
plt.clf()
plt.figure()
plt.plot(range(NUM_EPOCHS),ELBO_vals)
plt.title("VAE ELBO, %d Epochs, %d Latent Variables" % (NUM_EPOCHS,NUM_LATENT))
plt.xlabel("Epoch")
plt.ylabel("ELBO")
plt.savefig("elbo_ep%d_lv%d.png" % (NUM_EPOCHS,NUM_LATENT))

