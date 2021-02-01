import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
    
class VAE(nn.Module):
    def __init__(self, input_dim=136, z_dim=4):
        super().__init__()
        self.z_dim = z_dim
        self.efc1 = nn.Linear(input_dim, 512)
        self.efc2 = nn.Linear(512, 256)
        self.efc3 = nn.Linear(256, 128)
        self.fc21 = nn.Linear(128, z_dim)
        self.fc22 = nn.Linear(128, z_dim)
        self.dfc3 = nn.Linear(z_dim, 128)
        self.dfc4 = nn.Linear(128, 256)
        self.dfc5 = nn.Linear(256, 512)
        self.dfc6 = nn.Linear(512, input_dim)

    def forward(self, x):
        # encoder
        x, _ = self.encoder(x)
        
        # decoder                                                                    
        x = self.decoder(x)
        return x

    def encoder(self, x):
        # encoder                                                                                                        
        x_ = F.relu(self.efc1(x))
        x_ = F.relu(self.efc2(x_))
        x_ = F.relu(self.efc3(x_))

        # sampler ここではreluしないこと！！                                              
        self.mu = self.fc21(x_)
        self.logvar = self.fc22(x_)
        std = torch.exp(0.5*self.logvar)
        eps = torch.randn_like(std)
        if self.training:
            x = self.mu + std*eps
        else:
            x = self.mu
            
        # x_はエンコーダ直前の層の出力
        return x, x_

    def decoder(self, x):
        x = F.relu(self.dfc3(x))
        x = F.relu(self.dfc4(x))
        x = F.relu(self.dfc5(x))
        x = torch.tanh(self.dfc6(x))
        return x

    def loss(self, x_, x, w):
        mu, logvar = self.mu, self.logvar                                   
        mse = F.mse_loss(x_, x, reduction='sum')
        kld = 0.5 * torch.sum(mu**2 + logvar.exp() - logvar - 1)
        # return bce + kld
        return mse + w*kld
    
    def load(self, model_name='VAE.pth'):
        model_path = os.path.dirname(__file__) + '/' + model_name
        self.load_state_dict(torch.load(model_path))
        
    def save(self, model_name='VAE.pth'):
        model_path = os.path.dirname(__file__) + '/' + model_name
        torch.save(self.state_dict(), model_path)
    
    def train_model(self, epoch, optimizer, train_dataloader, weight=0.005):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)
        self = self.to(device)
        epoch_losses = []
        for i in range(epoch):
            losses = 0
            for x in train_dataloader:
                x = x.to(device)
                x_ = self.forward(x)
                loss = self.loss(x_, x, weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.cpu().detach().numpy()
            epoch_losses += [losses/len(train_dataloader)]
            print("EPOCH: {} loss: {}".format(i, losses/len(train_dataloader)))
        plt.plot(epoch_losses)
        plt.show()