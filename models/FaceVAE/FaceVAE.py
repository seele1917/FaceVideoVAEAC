import sys
import os
sys.path.append('../../')
from utils import separate_mouth_contour, combine_mouth_contour

import torch
import torch.nn as nn

class FaceVAE(nn.Module):
    def __init__(self, vae_mouth, vae_contour, output_dim=136):
        super().__init__()
        self.vae_mouth = vae_mouth
        self.vae_mouth.eval()
        self.vae_contour = vae_contour
        self.vae_contour.eval()
        
        z_dim_all = vae_mouth.z_dim + vae_contour.z_dim
        self.decoder_layer = nn.Sequential(
            nn.Linear(z_dim_all, 4),
            nn.ReLU(),
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, z_dim_all),
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def encoder(self, x):
        mouth_x, contour_x = separate_mouth_contour(x)
        _, mouth_x_ = self.vae_mouth.encoder(mouth_x)
        mouth_mu, mouth_logvar = self.vae_mouth.mu, self.vae_mouth.logvar

        _, contour_x_ = self.vae_contour.encoder(contour_x)
        contour_mu, contour_logvar = self.vae_contour.mu, self.vae_contour.logvar
        
        # sampler
        self.mu = torch.cat((mouth_mu, contour_mu), dim=1)
        self.logvar = torch.cat((mouth_logvar, contour_logvar), dim=1)
        
        std = torch.exp(0.5*self.logvar)
        eps = torch.randn_like(std)
        if self.training:
            x = self.mu + std*eps
        else:
            x = self.mu
        return x
    
    def decoder(self, z):        
        z = self.decoder_layer(z)
        mouth_z, contour_z = z[:, :self.vae_mouth.z_dim], z[:, self.vae_mouth.z_dim:]
        mouth = self.vae_mouth.decoder(mouth_z)
        contour = self.vae_contour.decoder(contour_z)
        face = combine_mouth_contour(mouth, contour)
        return face
    
    def loss(self, x, y, w=0.85):
        mouth_x, contour_x = separate_mouth_contour(x)
        mouth_y, contour_y = separate_mouth_contour(y)
        
        mse_mouth = F.mse_loss(mouth_x, mouth_y, reduction='sum')
        mse_contour = F.mse_loss(contour_x, contour_y, reduction='sum')
        
        return w*mse_mouth + (1-w)*mse_contour

    def load(self, model_name='FaceVAE.pth'):
        model_path = os.path.dirname(__file__) + '/' + model_name
        self.load_state_dict(torch.load(model_path))
        
    def save(self, model_name='FaceVAE.pth'):
        model_path = os.path.dirname(__file__) + '/' + model_name
        torch.save(self.state_dict(), model_path)