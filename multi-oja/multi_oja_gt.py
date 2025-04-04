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
W_R = torch.normal(torch.zeros(N_y, N_y), torch.ones(N_y, N_y))
W_R = W_R.to(gpu)

lr = 1e-3
n_train = 2000
alpha_start = 1e-1
alpha_end = 1e-3
# Exponential decay lr starting at alpha_start and ending at alpha_end
alphas = alpha_start*torch.exp(torch.arange(n_train) * (np.log(alpha_end / alpha_start) / (n_train-1)))
#alphas = ((alpha_end - alpha_start) / n_train) * torch.arange(n_train) + alpha_start
optim = torch.optim.Adam([W], lr=lr)
# Batches to draw from normal dist to (noisily) estimate PC i
losses = torch.empty((n_train,))
#W_val = torch.empty(n_train, N)
#pre = torch.empty((n_train, N))
#post = torch.empty((n_train, 1))
for i in range(n_train):
    r_pre = r_dist.sample().unsqueeze(1)
    r_post = W @ r_pre
    print(r_post.shape)

    B = 512
    c = torch.cov(r_dist.sample((B,)).t())
    eigvals, eigvecs = torch.linalg.eigh(c)
    # Get first N_y PCs
    pc_i = eigvecs[:, -N_y:].t().flip(dims=(0,))
    #print(pc_i.shape)
    #print(torch.linalg.vector_norm(pc_i, dim=1).unsqueeze(1).shape)

    print(eigvals[-N_y:], torch.exp(-torch.arange(N_x))[:N_y])

    norm_pci = pc_i / torch.linalg.vector_norm(pc_i, dim=1).unsqueeze(1)
    norm_W = W_FF / torch.linalg.vector_norm(W_FF, dim=1).unsqueeze(1)

    # unsqueezing to match required bmm format (bmm is just doing a dot product here)
    a = torch.bmm(norm_W.unsqueeze(1), norm_pci.unsqueeze(2))

    loss = 1-torch.mean(torch.abs(a))

    print(f"Iter {i}: {loss.item()}")
    losses[i] = loss.item()
    # W_val[i] = W
    # pre[i] = r_pre
    # post[i] = r_post
    delta_W = alphas[i] * r_post*(r_pre.t() - r_post*W_FF)
    W_FF += delta_W
    # TODO GT anti-hebbian for recurrent weights    
        
    # optim.zero_grad()
    # loss.backward()
    # optim.step()


plt.plot(losses)
plt.savefig("losses_gt.png")
torch.save(losses, "losses_gt.pt")
# torch.save(pre, "r_pre_gd.pt")
# torch.save(post, "r_post_gd.pt")
# torch.save(W_val, "weights_gd.pt")