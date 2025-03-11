import matplotlib.pyplot as plt
import torch
import numpy as np

gpu = torch.device("cuda:0")

N = 16
M = torch.normal(torch.zeros((N, N)), torch.ones((N, N)))
Q, _ = torch.linalg.qr(M)
Q = Q.to(gpu)
D = torch.diag(torch.exp(-torch.arange(N))).to(gpu)
Sigma = Q @ D @ Q.t()
r_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(N, device=gpu), Sigma)

W = torch.normal(torch.zeros(N), torch.ones(N))
W = W.to(gpu)
W.requires_grad_(True)

lr = 1e-3
n_train = 2000
alpha_start = 1e-2
alpha_end = 1e-3
# Exponential decay lr starting at alpha_start and ending at alpha_end
alphas = alpha_start*torch.exp(torch.arange(n_train) * (np.log(alpha_end / alpha_start) / (n_train-1)))
#alphas = ((alpha_end - alpha_start) / n_train) * torch.arange(n_train) + alpha_start
optim = torch.optim.Adam([W], lr=lr)
# Batches to draw from normal dist to (noisily) estimate PC1
B = 128
c = torch.cov(r_dist.sample((B,)).t())
eigvals, eigvecs = torch.linalg.eigh(c)
# eigvals sorted in ascending order so take last one
pc1 = eigvecs[:, -1]
norm_pc1 = pc1 / torch.linalg.vector_norm(pc1)
losses = torch.empty((n_train,))
W_val = torch.empty(n_train, N)
pre = torch.empty((n_train, N))
post = torch.empty((n_train, 1))
for i in range(n_train):
    r_pre = r_dist.sample()
    r_post = W.t() @ r_pre
    
    loss = 1-torch.abs(W.t() @ norm_pc1 / torch.linalg.vector_norm(W))
    print(f"Iter {i}: {loss.item()}")
    losses[i] = loss.item()
    W_val[i] = W
    pre[i] = r_pre
    post[i] = r_post
    #delta_W = alphas[i] * r_post*(r_pre - r_post*W)
    #W += delta_W
    optim.zero_grad()
    loss.backward()
    optim.step()


plt.plot(losses)
plt.savefig("losses_gd.png")
torch.save(losses, "losses_gd.pt")
torch.save(pre, "r_pre_gd.pt")
torch.save(post, "r_post_gd.pt")
torch.save(W_val, "weights_gd.pt")