import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, RidgeCV
from sklearn.preprocessing import PolynomialFeatures

gpu = torch.device("cuda:0")
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

N = 16
eps = 1e-7


M = torch.normal(torch.zeros((N, N)), torch.ones((N, N)))
Q, _ = torch.linalg.qr(M)
D = torch.diag(torch.exp(-torch.arange(N)))
Sigma = Q @ D @ Q.t()
r_dist = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(N), Sigma)

W = torch.normal(torch.zeros(N), torch.ones(N))

lr = 1e-3
n_train = 2000
alpha_start = 1e-2
alpha_end = 1e-3
# Exponential decay lr starting at alpha_start and ending at alpha_end
alphas = alpha_start*torch.exp(torch.arange(n_train) * (np.log(alpha_end / alpha_start) / (n_train-1)))
#alphas = ((alpha_end - alpha_start) / n_train) * torch.arange(n_train) + alpha_start

degree = 2

def load_features(start, end):
    pre = torch.load("r_pre_gd.pt").detach().cpu()
    pre_start = pre[start, :]
    post = torch.load("r_post_gd.pt").detach().cpu()
    post_start = post[start, :]
    W = torch.load("weights_gd.pt").detach().cpu()
    W0, Wf = W[start, :], W[end, :]
    return pre_start, post_start, W0, Wf

def prepare_features(r_pre, rpost, W):
    # Num vars in polynomial (presyn, postsyn, weight)
    numvars = 3
    # Num terms in polynomial is (degree + 1) ** num_vars
    numterms = (degree + 1) ** numvars

    r_post = rpost.expand((N))
    features = torch.stack((r_pre, r_post, W), dim=1)
    
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
    # # Last diff should be 0
    # delta_W = torch.empty_like(W)
    # delta_W[:-1, :] = torch.diff(W, dim=0)
    # delta_W[-1, :] = torch.zeros(N)
    delta_W = Wf - W0
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

def fit_weights(X, Y):
    # L1 ratio - 0.02 is all L2, 1 is all L1
    #reg = ElasticNet(l1_ratio=0.02)
    #reg = RidgeCV(alphas=torch.arange(1, 11) * 0.05)
    #reg = ElasticNet(alpha=alpha, l1_ratio=1.0)
    
    # Already have intercept from power series transform
    #reg = LinearRegression(fit_intercept=False)
    #reg = LinearRegression(fit_intercept=True) # now we have intercept
    reg = Lasso(alpha=1e-1, fit_intercept=False)
    reg.fit(X, Y)

    return reg

# We never train on random realization
def prepare_train_data(alphas, num_intervals=5, epoch_subset=None):    
    loss_intervals = load_loss(num_intervals, epoch_subset)
    train_samples = []
    train_labels = []
    manual_int = n_train
    b = torch.arange(manual_int)
    num_intervals = manual_int-1
    for i in range(num_intervals):
        start, end = b[i], b[i+1]
        #start, end = loss_intervals[i]
        pre, post, W0, Wf = load_features(start, end)
        features = prepare_features(pre, post, W0)
        features = alphas[start] * features
        labels = prepare_labels(W0, Wf)
        train_samples.append(features)
        train_labels.append(labels)
        
    all_samples = torch.cat(train_samples, dim=0)
    all_labels = torch.cat(train_labels, dim=0)

    #print(all_samples.shape)

    samples_mu = torch.mean(all_samples, dim=0, keepdim=True)
    samples_std = torch.std(all_samples, dim=0, keepdim=True)
    #print(samples_std.shape)
    # print(samples_mu)
    # print(samples_std)
    #print(torch.min(samples_mu), torch.max(samples_mu))
    normed_samples = (all_samples - samples_mu) / (samples_std + eps)

    #print(torch.min(normed_samples), torch.max(normed_samples))

    # Z-scoring labels too so we get correct coefs - these are just used to recover learning rule coefs not prediction
    Y_mu = torch.mean(all_labels, dim=0, keepdim=True)
    Y_std = torch.std(all_labels, dim=0, keepdim=True)
    normed_Y = (all_labels - Y_mu) / (Y_std + eps)
    #normed_Y = all_labels

    return loss_intervals, (normed_samples.detach().cpu().numpy(), samples_mu, samples_std), (normed_Y.detach().cpu().numpy(), Y_mu, Y_std)


def accum_rule(reg, X_mu, X_std, Y_mu, Y_std, num_steps=100):
    # Batches to draw from normal dist to (noisily) estimate PC1
    B = 1024
    c = torch.cov(r_dist.sample((B,)).t())
    _, eigvecs = torch.linalg.eigh(c)
    # eigvals sorted in ascending order so take last one
    pc1 = eigvecs[:, -1]
    norm_pc1 = pc1 / torch.linalg.vector_norm(pc1)

    eta = 5e-1
    W = torch.normal(torch.zeros(N), torch.ones(N))
    # TODO do we need to change how we z-score
    # We also 
    alphas_accum = alpha_start*torch.exp(torch.arange(num_steps) * (np.log(alpha_end / alpha_start) / (n_train-1)))

    losses = torch.empty((num_steps,))
    for i in range(num_steps):
        r_pre = r_dist.sample()
        r_post = W.t() @ r_pre
        features = prepare_features(r_pre, r_post, W)
        # alphas are same as training so we can z-score after multiplying by alpha
        features = alphas_accum[i] * features
        # LR doesn't decrease so converges faster than gradient descent, but less stable
        features = (features - X_mu) / (X_std + eps)
        loss = 1-torch.abs(W.t() @ norm_pc1 / torch.linalg.vector_norm(W))
        print(f"Iter {i}: {loss.item()}")
        losses[i] = loss
        
        delta_W = torch.from_numpy(reg.predict(features.detach().cpu().numpy())) * Y_std + Y_mu
        W += eta * delta_W

    return losses


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

    fig.suptitle("Oja's Linear Predictor Coefficients")    
    plt.show()


loss_intervals, feature_data, label_data = prepare_train_data(alphas, num_intervals=50, epoch_subset=None)
features, X_mu, X_std = feature_data
labels, Y_mu, Y_std = label_data
#print(f"Loss intervals: {loss_intervals}")
reg = fit_weights(features, labels)
print(f"Coefs: {reg.coef_}")
print(f"R^2: {reg.score(features, labels)}")
compare_coefs(degree=degree, numvars=3, coef=reg.coef_)
plt.savefig("coefs.png")
plt.close()

losses = accum_rule(reg, X_mu, X_std, Y_mu, Y_std, num_steps=n_train*5)
plt.plot(losses)
plt.title("Loss: Learning Rule Accumulation")
plt.savefig("accum_rule.png")