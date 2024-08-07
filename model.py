import numpy as np
import torch as T
import torch.nn as nn

class BraninNet(nn.Module):
    def __init__(self, lr, device):
        super(BraninNet, self).__init__()

        self.device = device
        self.learning_rate = lr
        self.input_dims = 2
        self.hidden1_dims = 200
        self.hidden2_dims = 500
        self.hidden3_dims = 300
        self.output_dims = 2
        
        self.input_block = nn.Sequential(nn.Linear(self.input_dims, self.hidden1_dims),
                                    nn.Tanh(),
                                    nn.BatchNorm1d(self.hidden1_dims))
        self.h_block1    = nn.Sequential(nn.Linear(self.hidden1_dims, self.hidden2_dims),
                                    nn.Tanh(),
                                    nn.BatchNorm1d(self.hidden2_dims))
        self.h_block2    = nn.Sequential(nn.Linear(self.hidden2_dims, self.hidden3_dims),
                                    nn.Tanh(),
                                    nn.BatchNorm1d(self.hidden3_dims))
        self.out_block   = nn.Sequential(nn.Linear(self.hidden3_dims, self.output_dims),
                                    nn.LeakyReLU())

        self.optimizer = T.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.L1Loss()
        self.to(self.device)
        
    def forward(self, input):
        output = self.input_block(input)
        output = self.h_block1(output)
        output = self.h_block2(output)
        output = self.out_block(output)
        return output 

    def train_batch(self, X, y):
        self.optimizer.zero_grad()
        self.train()
        pred = self.forward(X)
        loss = self.criterion(y, pred)
        loss.backward()
        self.optimizer.step()
        self.eval()
        return loss.item()

    def predict(self, X):
        self.eval()
        y = self.forward(X)
        y_hf = y[:, 0].cpu().detach().numpy()
        y_lf = y[:, 1].cpu().detach().numpy()
        return y_hf, y_lf