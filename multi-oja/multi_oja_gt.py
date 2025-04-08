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
sigma_R = 0.0
W_R = torch.normal(torch.zeros(N_y, N_y), torch.ones(N_y, N_y) * ((sigma_R ** 2) / N_y))
W_R = W_R.to(gpu)

n_train = 2000
alpha_start = 1e-2
alpha_end = 1e-3
alphas_hebb = 0
# Exponential decay lr starting at alpha_start and ending at alpha_end
alphas = alpha_start*torch.exp(torch.arange(n_train) * (np.log(alpha_end / alpha_start) / (n_train-1)))
#alphas = ((alpha_end - alpha_start) / n_train) * torch.arange(n_train) + alpha_start
# Batches to draw from normal dist to (noisily) estimate PC i

losses = torch.empty((n_train,))
W_ff = torch.empty(n_train, N_y, N_x)
W_r = torch.empty(n_train, N_y, N_y)
pre = torch.empty((n_train, N_x))
post = torch.empty((n_train, N_y))

B = 1024
c = torch.cov(r_dist.sample((B,)).t())
eigvals, eigvecs = torch.linalg.eigh(c)
# Get first N_y PCs
pc_i = eigvecs[:, -N_y:].t()

for i in range(n_train):

    X = r_dist.sample()
    # Convergent dynamics matrix for output layer
    W_tilde = (torch.linalg.inv((torch.eye(N_y, device=gpu) - W_R)) @ W_FF)
    # output neurons predicted with W_FF
    Y_hat = W_tilde @ X
    Y = pc_i @ X
    
    norm_pci = pc_i / torch.linalg.vector_norm(pc_i, dim=1).unsqueeze(1)
    norm_W = W_tilde / torch.linalg.vector_norm(W_tilde, dim=1).unsqueeze(1)

    # unsqueezing to match required bmm format (bmm is just doing a dot product here)
    a = torch.bmm(norm_W.unsqueeze(1), norm_pci.unsqueeze(2))

    loss = 1-torch.mean(torch.abs(a))

    print(f"Iter {i}: {loss.item()}")
    losses[i] = loss.item()

    W_ff[i] = W_FF
    W_r[i] = W_R
    pre[i] = X
    post[i] = Y_hat

    delta_W_FF = alphas[i] * Y_hat.unsqueeze(1) * (X.unsqueeze(0) - Y_hat.unsqueeze(1) * W_FF)
    W_FF += delta_W_FF
    delta_W_R = -alphas_hebb * torch.diag(Y_hat * Y_hat)
    W_R += delta_W_R


plt.plot(losses)
plt.savefig("losses_gt.png")
plt.close()
overlaps = torch.einsum('bij, j -> bi', W_ff, pc_i[0, :].cpu())
plt.plot(overlaps)
plt.savefig("overlaps.png")
plt.close()
torch.save(losses, "losses_gt.pt")
torch.save(pre, "r_pre_gt.pt")
torch.save(post, "r_post_gt.pt")
torch.save(W_ff, "w_ff_gt.pt")
torch.save(W_r, "w_r_gt.pt")