import numpy as np
import torch as T

class BraninDataset:
    def __init__(self, y_func, device, train_size=1000000, validation_size=10000, batch_size=10):
        self.train_size = train_size 
        self.validation_size = validation_size
        self.batch_size = batch_size
        self.y_func = y_func
        self.device = device
        
        # x1 and x2 range -> [low, high]
        self.training_set_range = ([-5, 10], [0, 15]) 
        self.validation_set_range = ([-5, 10], [0, 15])

        self.X_train, self.y_train = self.__generate_data_with_range(self.train_size, self.training_set_range)
        self.X_val, self.y_val = self.__generate_data_with_range(self.validation_size, self.validation_set_range)

    def __iter__(self):
        self.index = np.arange(self.train_size)
        np.random.shuffle(self.index)
        self.index_start = 0
        self.index_end = self.index.shape[0]
        return self
    
    def __next__(self):
        if self.index_start == self.index_end: raise StopIteration
        batch_index_end = min(self.index_start + self.batch_size, self.index_end)
        batch = self.index[self.index_start:batch_index_end]
        self.index_start = batch_index_end
        return self.X_train[batch,:], self.y_train[batch,:]

    def __generate_data_with_range(self, size, X_ranges):
        X = T.zeros(size, len(X_ranges))
        for i, (x_l, x_h) in enumerate(X_ranges):
            X[:,i] = T.FloatTensor(size).uniform_(x_l, x_h)
        y_hf, y_lf = self.y_func(X[:, 0], X[:, 1])
        y = T.zeros(size, 2)
        y[:, 0] = y_hf
        y[:, 1] = y_lf
        return X.to(self.device), y.to(self.device)

    def __sample_batch(self, X, y, size, batch_size):
        if batch_size == -1: return X, y
        index = T.randint(0, size, (batch_size,))
        return X[index], y[index]

    def sample_train_batch(self, batch_size=-1):
        return self.__sample_batch(self.X_train, self.y_train, self.train_size, batch_size)

    def sample_val_batch(self, batch_size=-1):
        return self.__sample_batch(self.X_val, self.y_val, self.validation_size, batch_size)
