import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, RidgeCV
from sklearn.preprocessing import PolynomialFeatures

gpu = torch.device("cuda:0")
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

eps = 1e-7
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

sigma_R = 0.1
W_R = torch.normal(torch.zeros(N_y, N_y), torch.ones(N_y, N_y) * ((sigma_R ** 2) / N_y))
W_R = W_R.to(gpu)

n_train = 3000
alpha_start = 1e-2
alpha_end = 1e-3
# Exponential decay lr starting at alpha_start and ending at alpha_end
alphas = alpha_start*torch.exp(torch.arange(n_train) * (np.log(alpha_end / alpha_start) / (n_train-1)))
#alphas = ((alpha_end - alpha_start) / n_train) * torch.arange(n_train) + alpha_start

degree = 2

def load_features(start, end):
    pre = torch.load("r_pre_gt.pt").detach().cpu()
    pre_start = pre[start, :]
    post = torch.load("r_post_gt.pt").detach().cpu()
    post_start = post[start, :]
    W_FF = torch.load("w_ff_gt.pt").detach().cpu()
    Wff_0, Wff_f = W_FF[start, :], W_FF[end, :]
    W_R = torch.load("w_r_gt.pt").detach().cpu()
    WR_0, WR_f = W_R[start, :], W_R[end, :]
    return pre_start, post_start, Wff_0, Wff_f, WR_0, WR_f

def prepare_features(pre, post, W):
    # Num vars in polynomial (presyn, postsyn, weight)
    numvars = 3
    # Num terms in polynomial is (degree + 1) ** num_vars
    numterms = (degree + 1) ** numvars

    pre = pre.unsqueeze(0).broadcast_to(W.shape).flatten()
    post = post.unsqueeze(1).broadcast_to(W.shape).flatten()
    W = W.flatten()
    features = torch.stack((pre, post, W), dim=1)
    
    features = features.unsqueeze(1).expand((-1, numterms, -1))
    # features: (16, 3)
    exps = torch.cartesian_prod(*([torch.arange(degree + 1)] * numvars))
    # exps: (8, 3)
    exps = exps.unsqueeze(0).broadcast_to(features.shape)
    #print(exps.shape)
    # exps: (16, 8, 3)

    transformed_features = torch.pow(features, exps).prod(dim=2, keepdim=False)

    return transformed_features

def prepare_labels(W0, Wf):
    delta_W = Wf.flatten() - W0.flatten()
    return delta_W

# Subdivides loss and returns in 2D tensor of (num_intervals, 2) where the second dim gives (epoch_start, epoch_end)
def load_loss(num_intervals, epoch_bound):
    if epoch_bound == None:
        epoch_bound = (0, n_train)
    losses = torch.load(f"losses_gd.pt").detach().cpu()
    losses_region = losses[epoch_bound[0]: epoch_bound[1]]
    total_reduction = losses_region[-1] - losses_region[0]
    interval = total_reduction / (num_intervals)
    loss_values = losses_region[0] + torch.arange(num_intervals + 1) * interval
    sorted, orig_indices = torch.sort(losses_region)
    inds = torch.bucketize(loss_values, sorted)
    inds = orig_indices[inds]
    epoch_intervals = []
    for i in range(1, num_intervals+1):
        # Round to nearest 100 epochs b/c those are the snapshots we covered
        #lower_bound = round(inds[i-1].item() / float(snapshot_every)) * snapshot_every
        #upper_bound = round(inds[i].item() / float(snapshot_every)) * snapshot_every
        lower_bound = inds[i-1].item()
        upper_bound = inds[i].item()
        loss_bounds = (lower_bound, upper_bound)
        epoch_intervals.append(loss_bounds)
    
    return epoch_intervals

def fit_powerseries(X, Y, alpha=1e-1):
    # L1 ratio - 0.02 is all L2, 1 is all L1
    #reg = ElasticNet(l1_ratio=0.02)
    #reg = RidgeCV(alphas=torch.arange(1, 11) * 0.05)
    #reg = ElasticNet(alpha=alpha, l1_ratio=1.0)
    
    # Already have intercept from power series transform
    #reg = LinearRegression(fit_intercept=False)
    #reg = LinearRegression(fit_intercept=True) # now we have intercept
    reg = Lasso(alpha=alpha, fit_intercept=False)
    reg.fit(X, Y)

    return reg

def prepare_train_data(alphas, num_intervals=5, epoch_subset=None):    
    loss_intervals = load_loss(num_intervals, epoch_subset)
    oja_train_samples = []
    oja_train_labels = []
    ahebb_train_samples = []
    ahebb_train_labels = []
    manual_int = n_train
    b = torch.arange(manual_int)
    num_intervals = manual_int-1
    for i in range(num_intervals):
        start, end = b[i], b[i+1]
        #start, end = loss_intervals[i]
        pre, post, Wff_0, Wff_f, WR_0, WR_f = load_features(start, end)
        features_oja = prepare_features(pre, post, Wff_0)
        features_oja = alphas[start] * features_oja
        labels_oja = prepare_labels(Wff_0, Wff_f)
        oja_train_samples.append(features_oja)
        oja_train_labels.append(labels_oja)
        
        features_ahebb = prepare_features(post, post, WR_0)
        labels_ahebb = prepare_labels(WR_0, WR_f)
        ahebb_train_samples.append(features_ahebb)
        ahebb_train_labels.append(labels_ahebb)
        
        
    oja_all_samples = torch.cat(oja_train_samples, dim=0)
    oja_all_labels = torch.cat(oja_train_labels, dim=0)
    oja_samples_mu = torch.mean(oja_all_samples, dim=0, keepdim=True)
    oja_samples_std = torch.std(oja_all_samples, dim=0, keepdim=True)
    oja_normed_samples = (oja_all_samples - oja_samples_mu) / (oja_samples_std + eps)
    oja_Y_mu = torch.mean(oja_all_labels, dim=0, keepdim=True)
    oja_Y_std = torch.std(oja_all_labels, dim=0, keepdim=True)
    oja_normed_Y = (oja_all_labels - oja_Y_mu) / (oja_Y_std + eps)

    
    ahebb_all_samples = torch.cat(ahebb_train_samples, dim=0)
    ahebb_all_labels = torch.cat(ahebb_train_labels, dim=0)
    ahebb_samples_mu = torch.mean(ahebb_all_samples, dim=0, keepdim=True)
    ahebb_samples_std = torch.std(ahebb_all_samples, dim=0, keepdim=True)
    ahebb_normed_samples = (ahebb_all_samples - ahebb_samples_mu) / (ahebb_samples_std + eps)
    ahebb_Y_mu = torch.mean(ahebb_all_labels, dim=0, keepdim=True)
    ahebb_Y_std = torch.std(ahebb_all_labels, dim=0, keepdim=True)
    ahebb_normed_Y = (ahebb_all_labels - ahebb_Y_mu) / (ahebb_Y_std + eps)
    
    #normed_Y = all_labels

    oja_features_X_stats = (oja_normed_samples.detach().cpu().numpy(), oja_samples_mu, oja_samples_std)
    oja_features_Y_stats = (oja_normed_Y.detach().cpu().numpy(), oja_Y_mu, oja_Y_std)

    ahebb_features_X_stats = (ahebb_normed_samples.detach().cpu().numpy(), ahebb_samples_mu, ahebb_samples_std)
    ahebb_features_Y_stats = (ahebb_normed_Y.detach().cpu().numpy(), ahebb_Y_mu, ahebb_Y_std)

    return loss_intervals, oja_features_X_stats, oja_features_Y_stats, ahebb_features_X_stats, ahebb_features_Y_stats


def accum_rule(reg_oja, reg_ahebb, oja_features_stats, ahebb_features_stats, learning_rate_scale=1.0, num_steps=100, alpha_mode="none", sample_new_cov=True):
    if sample_new_cov:
        M = torch.normal(torch.zeros((N_x, N_x)), torch.ones((N_x, N_x)))
        Q, _ = torch.linalg.qr(M)
        Q = Q.to(gpu)
        D = torch.diag(torch.exp(-torch.arange(N_x))).to(gpu)
        Sigma = Q @ D @ Q.t()
        r_dist_accum = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(N_x, device=gpu), Sigma)
    else:
        r_dist_accum = r_dist

    X_oja_mu, X_oja_std, Y_oja_mu, Y_oja_std = oja_features_stats
    X_ahebb_mu, X_ahebb_std, Y_ahebb_mu, Y_ahebb_std = ahebb_features_stats

    # Batches to draw from normal dist to (noisily) estimate PC1
    B = 512
    c = torch.cov(r_dist_accum.sample((B,)).t())
    _, eigvecs = torch.linalg.eigh(c)
    # eigvals sorted in ascending order so take last one
    pc_i = eigvecs[:, -N_y:].t()

    W_FF = torch.normal(torch.zeros(N_y, N_x), torch.ones(N_y, N_x)).to(gpu)
    Wff_gt = W_FF.clone()

    W_R = torch.normal(torch.zeros(N_y, N_y), torch.ones(N_y, N_y) * ((sigma_R ** 2) / N_y)).to(gpu)
    Wr_gt = W_R.clone()

    alphas_accum = alpha_start*torch.exp(torch.arange(num_steps) * (np.log(alpha_end / alpha_start) / (num_steps-1)))

    losses = torch.empty((num_steps,))
    losses_gt = torch.empty((num_steps,))
    for i in range(num_steps):
        pre = r_dist_accum.sample().to(gpu)
        
        W_tilde = torch.linalg.inv(torch.eye(N_y, device=gpu) - W_R)
        Wtilde_gt = torch.linalg.inv(torch.eye(N_y, device=gpu) - Wr_gt)

        # Post computed with our powerseries' updates to respective weights
        post_hat = W_tilde @ (W_FF @ pre)
        # Post computed with Oja's and anti-Hebbian rules applied to the respective weights
        post_gt = Wtilde_gt @ (Wff_gt @ pre)

        # The actual state both rules should get to
        post_target = pc_i @ pre
        
        features_oja = prepare_features(pre.cpu(), post_hat.cpu(), W_FF.cpu())
        # No alpha rate on anti-hebb lateral plasticity
        features_ahebb = prepare_features(post_hat.cpu(), post_hat.cpu(), W_R.cpu())

        # alphas are same as training so we can z-score after multiplying by alpha
        if alpha_mode == "same":
            features_oja = alphas_accum[i] * features_oja
        # LR doesn't decrease so converges faster than gradient descent, but less stable
        features_oja = (features_oja - X_oja_mu) / (X_oja_std + eps)
        features_ahebb = (features_ahebb - X_ahebb_mu) / (X_ahebb_std + eps)
        
        # Loss from using accumulation rule
        loss = torch.mean(torch.abs(post_hat-post_target))
        # Loss from using actual Oja + anti-Hebbian rule
        loss_gt = torch.mean(torch.abs(post_gt-post_target))
        print(f"Iter {i}: {loss.item()}")
        losses[i] = loss.item()
        losses_gt[i] = loss_gt.item()
        
        delta_W_FF = torch.from_numpy(reg_oja.predict(features_oja.detach().cpu().numpy())) * Y_oja_std + Y_oja_mu
        delta_W_FF = delta_W_FF.view(W_FF.shape).to(gpu)
        delta_W_R = torch.from_numpy(reg_ahebb.predict(features_ahebb.detach().cpu().numpy())) * Y_ahebb_std + Y_ahebb_mu
        delta_W_R = delta_W_R.view(W_R.shape).to(gpu)
        if alpha_mode == "after":
            delta_W_FF *= alphas_accum[i] * delta_W_FF

        delta_Wff_gt = post_gt.unsqueeze(1) * (pre.unsqueeze(0) - post_gt.unsqueeze(1) * Wff_gt)
        delta_Wr_gt = -1 * torch.diag(post_gt * post_gt)
        W_FF += learning_rate_scale * delta_W_FF
        Wff_gt += learning_rate_scale * delta_Wff_gt
        W_R += learning_rate_scale * delta_W_R
        Wr_gt += learning_rate_scale * delta_Wr_gt


    return losses, losses_gt


def compare_coefs(degree, numvars, coef):
    n = degree + 1
    coefs = np.reshape(coef, tuple([n] * numvars))
    fig, ax = plt.subplots(nrows=n, ncols=n, sharex=True, sharey=True, constrained_layout=True)
    for i in range(n):
        for j in range(n):
            ax[i, j].bar(torch.arange(n), coefs[i][j])
            ax[i, j].set_xticks(torch.arange(n))
            if (i, j) == (0, 0):
                ax[i, j].set_ylabel("coef")
            if (i, j) == (n-1, n-1):
                ax[i, j].set_xlabel("degree w_start")

    fig.supxlabel("degree post")
    fig.supylabel("degree pre")
    return fig


loss_intervals, oja_features_X_stats, oja_features_Y_stats, ahebb_features_X_stats, ahebb_features_Y_stats = prepare_train_data(alphas, num_intervals=50, epoch_subset=None)
oja_features, oja_X_mu, oja_X_std = oja_features_X_stats
oja_labels, oja_Y_mu, oja_Y_std = oja_features_Y_stats
ahebb_features, ahebb_X_mu, ahebb_X_std = ahebb_features_X_stats
ahebb_labels, ahebb_Y_mu,ahebb_Y_std = ahebb_features_Y_stats
#print(f"Loss intervals: {loss_intervals}")
reg_oja = fit_powerseries(oja_features, oja_labels, alpha=1e-1)
reg_ahebb = fit_powerseries(ahebb_features, ahebb_labels, alpha=5e-1)
print(f"(Predicted) Oja Coefs: {reg_oja.coef_}")
print(f"(Predicted) Anti-Hebbian Coefs: {reg_ahebb.coef_}")
print(f"Oja R^2: {reg_oja.score(oja_features, oja_labels)}")
print(f"Anti-Hebbian R^2: {reg_ahebb.score(ahebb_features, ahebb_labels)}")
fig = compare_coefs(degree=degree, numvars=3, coef=reg_oja.coef_)
fig.suptitle("Oja's Linear Predictor Coefficients")
plt.savefig("oja_coefs.png")
plt.close()
fig = compare_coefs(degree=degree, numvars=3, coef=reg_ahebb.coef_)
fig.suptitle("Anti-Hebbian Linear Predictor Coefficients")
plt.savefig("anti_hebbian_coefs.png")
plt.close()

oja_features_stats = (oja_X_mu, oja_X_std, oja_Y_mu, oja_Y_std)
ahebb_features_stats = (ahebb_X_mu, ahebb_X_std, ahebb_Y_mu, ahebb_Y_std)
losses, losses_gt = accum_rule(reg_oja, reg_ahebb, oja_features_stats, ahebb_features_stats, 
                                learning_rate_scale=0.1, num_steps=n_train * 1, alpha_mode="none", sample_new_cov=True)
plt.plot(losses)
plt.ylim([0, 1])
plt.title("Loss: Learning Rule Accumulation")
plt.savefig("accum_rule.png")
plt.show()
