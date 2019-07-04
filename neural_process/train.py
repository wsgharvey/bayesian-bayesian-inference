from numpy.random import poisson, normal, choice
import torch
import matplotlib.pyplot as plt
from network import AttentiveNeuralProcess

def sample_polynomial(lengthscale):
    order = poisson(2)
    bias = normal(0, 1)
    return [normal(0, lengthscale**(-d)) for d in range(0, order+1)]

def eval_polynomial(weights, X):
    return sum(w*X**d for d, w in enumerate(weights))

def add_noise(sigma, Y):
    return Y + torch.randn_like(Y)*sigma

def generate_data():
    lengthscale = 10
    max_context = 30
    max_target = 50
    noise_std = 0.01
    def unit_to_interval(X):
        return (X-0.5)*2*lengthscale
    polynomial = sample_polynomial(lengthscale)
    def get_samples(polynomial, max_n):
        n = choice(range(1, max_n))
        X = unit_to_interval(torch.rand(n))
        Y = add_noise(noise_std, eval_polynomial(polynomial, X))
        return X, Y
    X_context, Y_context = get_samples(polynomial, max_context)
    X_target, Y_target = get_samples(polynomial, max_target)
    return [xy.unsqueeze(-1) for xy in [X_context, Y_context, X_target, Y_target]]

def generate_batch(batch_size):
    batches = [], [], [], []
    for _ in range(batch_size):
        result = generate_data()
        for T, t in zip(batches, result):
            T.append(t)
    return batches  # we do not stack into tensor as they have variable length

anp = AttentiveNeuralProcess(x_dim=1, y_dim=1, num_hidden=64)
optim = torch.optim.Adam(anp.parameters(), lr=1e-4)

for iteration in range(50000):
    y_pred, kl, loss = anp.forward(*generate_batch(64))
    
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    if iteration % 50 == 0:
        print(iteration, loss)
        torch.save({
                    'iteration': iteration,
                    'model_state_dict': anp.state_dict(),
                    'optimizer_state_dict': optim.state_dict()
                    }, f"checkpoint_{0 if iteration%100 == 0 else 1}.pt")