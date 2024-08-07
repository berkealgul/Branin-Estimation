import matplotlib.pyplot as plt
import numpy as np
import torch as T
from model import BraninNet
from dataset import BraninDataset

TRAIN = False

def braninFunc(x1, x2):
    def BR(x1_, x2_):
        return 10 + (x2_ - 5.1*(x1_**2/(4*np.pi**2)) + 5*x1_/np.pi - 6)**2 + 10*np.cos(x1_)*(1-1/(8*np.pi))
    y_hf = BR(x1, x2) - 22.5*x2
    y_lf = BR(0.7*x1, 0.7*x2) - 15.75*x2 + 20*(0.9+x1)**2 - 50
    return y_hf, y_lf

dataset_train_size = 3200000
dataset_val_size = 3200

num_epoch = 15
batch_size = 128
learning_rate = 0.002
device =  T.device("cuda:0" if T.cuda.is_available() else "cpu")

# TRAINING (or load pretrain weights if we want to)

dataset = BraninDataset(train_size=dataset_train_size, validation_size=dataset_val_size, 
                        batch_size=batch_size, y_func=braninFunc, device=device)
net = BraninNet(learning_rate, device)
losses = []

if TRAIN:
    print("Training Started")
    for i in range(num_epoch):
        running_loss = 0.0
        for _, (X_batch, y_batch) in enumerate(dataset):
            loss = net.train_batch(X_batch, y_batch)
            losses.append(loss)
            running_loss += loss
        print("Epoch: %d/%d  Running Loss: %f" %(i+1, num_epoch, running_loss/1000))
    T.save(net.state_dict(), 'model_weights.pth')
else:
    print("Pre-Trained Model Loaded")
    net.load_state_dict(T.load('model_weights.pth'))

# VALIDATION

# validation_treshold = 10
# X_val, y_val = dataset.sample_val_batch()
# y_pred = net.forward(X_val)
# val_acc = y_val[(y_val - y_pred).abs() < validation_treshold].numel() / y_val.numel()

# print("Validation Accuracy", val_acc)

# PLOTTING

plot_surface_len = 20
x1 = np.linspace(-5, 10, plot_surface_len)
x2 = np.linspace(0, 15, plot_surface_len)
X1, X2 = np.meshgrid(x1, x2)
Xp = T.empty(x1.shape[0] * x2.shape[0], 2)
Xp[:, 0] = T.tensor(X1.flatten())
for i in range(plot_surface_len):
    a = i*plot_surface_len
    b = (i+1)*plot_surface_len
    Xp[a:b,1] = x2[i]

y_hf, y_lf = braninFunc(X1, X2)
y_hf_pred, y_lf_pred = net.predict(Xp.to(device))
y_hf_pred = np.reshape(y_hf_pred, (plot_surface_len, plot_surface_len))
y_lf_pred = np.reshape(y_lf_pred, (plot_surface_len, plot_surface_len))

fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection='3d'))
fig.suptitle('Bramin')
for ax in axs.flatten():
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
axs[0, 0].set_title("y_hf")
axs[0, 1].set_title("y_lf")
axs[1, 0].set_title("Predicted y_hf")
axs[1, 1].set_title("Predicted y_lf")
axs[0, 0].plot_surface(X1, X2, y_hf)
axs[0, 1].plot_surface(X1, X2, y_lf)
axs[1, 0].plot_surface(X1, X2, y_hf_pred)
axs[1, 1].plot_surface(X1, X2, y_lf_pred)

if TRAIN:
    plt.figure(2)
    plt.title("Loss per Batch")
    plt.plot(np.arange(len(losses)), losses)
    fig.savefig("Bramin.png")
    plt.savefig("loss.png")

plt.show()