from numpy.random import poisson, normal, choice
import torch
import matplotlib.pyplot as plt
from network import AttentiveNeuralProcess
from train import generate_batch

anp = AttentiveNeuralProcess(x_dim=1, y_dim=1, num_hidden=128)
anp.load_state_dict(torch.load('checkpoint_1.pt')['model_state_dict'])

print("standard deviation:", anp.std)

n_rows = 5
n_cols = 6
n_plots = n_rows*n_cols
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(16, 3*n_rows))
axes = axes.reshape(-1)
Xcontext, Ycontext, _, _ = generate_batch(n_plots)
Xtest = torch.arange(-10., 10., 0.1).view(1, -1, 1).expand(len(Xcontext), -1, -1)
Ypred, _, _ = anp.forward(Xcontext, Ycontext, Xtest, None)
for i, (x, y) in enumerate(zip(Xcontext, Ycontext)):
    x = x.view(-1).numpy()
    y = y.view(-1).numpy()
    axes[i].scatter(x, y, color='k')
for i, (xt, yt) in enumerate(zip(Xtest, Ypred)):
    xt = xt.view(-1).numpy()
    yt = yt.view(-1).detach().numpy()
    axes[i].plot(xt, yt, color='r')
    axes[i].set_xlim(-10, 10)
    axes[i].set_ylim(-2, 2)
plt.show()