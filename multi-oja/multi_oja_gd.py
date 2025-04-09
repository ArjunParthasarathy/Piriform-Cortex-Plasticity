import matplotlib.pyplot as plt
import torch
import numpy as np

gpu = torch.device("cuda:0")

# Neurons in input layer
N_x = 16
# Neurons in output layer - leq to input layer neurons
N_y = 4
M = torch.normal(torch.zeros((N_x, N_x)), torch.ones((N_x, N_x)))
Q, _ = torch.linalg.qr(M)
Q = Q.to(gpu)
D = torch.diag(torch.exp(-torch.arange(N_x))).to(gpu)
Sigma = Q @ D @ Q.t()
r_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(N_x, device=gpu), Sigma)

W_FF = torch.normal(torch.zeros(N_y, N_x), torch.ones(N_y, N_x))
W_FF = W_FF.to(gpu)
W_FF.requires_grad_(True)
sigma_R = 0.1
W_R = torch.normal(torch.zeros(N_y, N_y), torch.ones(N_y, N_y) * ((sigma_R ** 2) / N_y))
W_R = W_R.to(gpu)
W_R.requires_grad_(True)

lr = 1e-1
n_train = 3000
alpha_start = 1e-2
alpha_end = 1e-2
# Exponential decay lr starting at alpha_start and ending at alpha_end
alphas = alpha_start*torch.exp(torch.arange(n_train) * (np.log(alpha_end / alpha_start) / (n_train-1)))
#alphas = ((alpha_end - alpha_start) / n_train) * torch.arange(n_train) + alpha_start
optim_FF = torch.optim.Adam([W_FF], lr=lr)
optim_R = torch.optim.Adam([W_R], lr=lr)

losses = torch.empty((n_train,))
W_ff = torch.empty(n_train, N_y, N_x)
W_r = torch.empty(n_train, N_y, N_y)
pre = torch.empty((n_train, N_x))
post = torch.empty((n_train, N_y))

# Jointly train W_FF and W_R
for i in range(n_train):
    X = r_dist.sample()
    # Convergent dynamics matrix for output layer
    W_tilde = torch.linalg.inv((torch.eye(N_y, device=gpu) - W_R)) @ W_FF
    # output neurons predicted with W_FF
    Y_hat = W_tilde @ X
    
    B = 512
    c = torch.cov(r_dist.sample((B,)).t())
    eigvals, eigvecs = torch.linalg.eigh(c)
    # last neuron has PC1
    # TODO does it matter whether we flip? NO
    #pc_i = torch.flip(eigvecs[:, -N_y:].t(), dims=(0,))
    pc_i = eigvecs[:, -N_y:].t().flip(dims=(0,))
    
    # output neurons predicted with the first N_Y PCs (the last output neuron has first PC)
    Y = pc_i @ X
    
    # norm_pci = pc_i / torch.linalg.vector_norm(pc_i, dim=1).unsqueeze(1)
    # norm_W = W_FF / torch.linalg.vector_norm(W_FF, dim=1).unsqueeze(1)

    # # transposing and unsqueezing to match required bmm format
    # a = torch.bmm(norm_W.unsqueeze(1), norm_pci.unsqueeze(2)).squeeze(1, 2)

    lambda_ff = 1
    lambda_r = 5
    l1reg_ff = torch.mean(torch.sum(W_FF ** 2) / (N_y*N_x))
    l1reg_r = torch.mean(torch.sum(W_R ** 2) / (N_x*N_x))
    # loss = torch.mean((Y-Y_hat) ** 2) + lambda_ff*l1reg_ff + lambda_r*l1reg_r
    # for each PC, find the best-aligned post neuron
    PCs2output = torch.zeros((pc_i.shape[0]), dtype=torch.int, requires_grad=False)
    for ii in range(pc_i.shape[0]):
        with torch.no_grad():
            PCs2output[ii] = torch.argmax(W_tilde @ pc_i[ii, :] * 1 / (torch.linalg.vector_norm(W_tilde, dim=1)))
    overlaps = torch.diag((W_tilde[PCs2output, :] @ pc_i.t()) * 1 / torch.linalg.vector_norm(W_tilde[PCs2output, :], dim=1))
    loss = 1 - torch.mean(torch.abs(overlaps))
    print(f"Iter {i}: {loss.item()}, overlaps: {overlaps}")
    losses[i] = loss.item()
    W_ff[i] = W_FF
    W_r[i] = W_R
    pre[i] = X
    post[i] = Y_hat

    optim_FF.zero_grad()
    optim_R.zero_grad()
    
    loss.backward()

    optim_FF.step()
    optim_R.step()


plt.plot(losses)
plt.savefig("losses_gd.png")
torch.save(losses, "losses_gd.pt")
torch.save(pre, "r_pre_gd.pt")
torch.save(post, "r_post_gd.pt")
torch.save(W_ff, "w_ff_gd.pt")
torch.save(W_r, "w_r_gd.pt")
plt.show()