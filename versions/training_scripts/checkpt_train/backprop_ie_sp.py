import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os

gpu = torch.device("cuda:0")
print(torch.cuda.get_device_name(0))

# Use smaller network for testing - ex 2000 neurons
# Even for the project, doing it for 10^6 neurons would take too long
# Problem this creates: test network is denser than actual network b/c we have 10^3 neurons but 10^2 connections per neuron
num_neurons = 2000
num_i = int(0.1 * num_neurons)
num_e = int(0.9 * num_neurons)

# Num excitatory inputs and inhibitory inputs to each neuron (in reality it should be 500 but we reduce it here to make things faster)
k = 100

# Number of olfactory bulb channels (glomeruli) to each neuron
D = 10 ** 3
# For each neuron, how many glomeruli inputs it receives (should be 10^2)
num_channel_inputs = 100

# Channel signal if not active for odor a
i_0 = 2.
# Channel signal if active for odor a
i_1 = 10.
# Probability that a channel is active for an odor a
f = 0.1
# Number of odors
P = 16
# Novel activity is up to P // 2, and familiar activity is after
novel_inds = torch.arange(0, P // 2)
familiar_inds = torch.arange(P // 2, P)

# Creates sparse adjacency matrix with the given probability of edge connection and size mxn
def create_adj_matrix(p, m, n):
    # num_connections = int(p * m * n)
    # m_coords = torch.randint(0, m, (num_connections,))
    # n_coords = torch.randint(0, n, (num_connections,))
    # indices = torch.vstack((m_coords, n_coords))
    # values = torch.ones(num_connections)
    # A_mn = torch.sparse_coo_tensor(indices, values, (m, n))
    probs = torch.ones(m, n) * p
    A_mn = torch.bernoulli(probs)
    return A_mn

P_prime = 4
def correlated_mitral_activity():
    # Each of the P' odors is independent (correlation of 0)
    sigma_p_prime = torch.zeros((P_prime, P_prime)).fill_diagonal_(1)
    dist = torch.distributions.MultivariateNormal(torch.zeros(P_prime), sigma_p_prime)
    p_prime_activity = dist.sample(torch.Size([D]))
    var = 1 / P_prime
    projection = torch.normal(torch.zeros((P_prime, P)), torch.ones(P_prime, P) * np.sqrt(var))
    activity = p_prime_activity @ projection
    return activity.to(gpu)

# Takes in mitral activity I and feedforward weights W_ff and computes feedforward activity h_bar_ff
def compute_feedforward_activity(W_ff, I):
    with torch.device(gpu):
        h_ff = (W_ff @ I) * (1 / np.sqrt(num_channel_inputs))
        h_bar_ff = torch.zeros_like(h_ff)
        # Subtract by mean across (excitatory) neurons for each odor
        h_bar_ff[:num_e] = h_ff[:num_e] - torch.mean(h_ff[:num_e], dim=0, keepdim=True)
    return h_bar_ff

# Computes feedforward (channel) weights mapping mitral activity onto E,I neurons
def compute_feedforward_weights():
    # Probability that a channel weight will be nonzero
    p = num_channel_inputs / D
    with torch.device(gpu):
        a = create_adj_matrix(p, num_e, D)
        # Inhibitory neurons don't receive channel input
        # This is the first simplification, where we neglect the first inhibitory layer I_ff
        b = torch.zeros(size=(num_i, D))
        W_ff = torch.cat(tensors=(a, b), dim=0)

    return W_ff

def compute_initial_recurrent_weights():
    k_ee = k_ei = k_ie = k_ii = k
    #p_ee = k_ee / num_e
    # k inhibitory inputs to that e neuron, out of num_i total inhibitory neurons gives the connection probability per neuron
    p_ei = k_ei / num_i
    p_ie = k_ie / num_e
    #p_ii = k_ii / num_i
    
    # Constants
    #w_ee = 0.1
    w_ei = 0.2
    w_ie = 0.5
    #w_ii = 0.3
    # Ignore ee and ii weights for now:
    p_ee = p_ii = w_ee = w_ii = 0
    with torch.device(gpu):
        W_ee = create_adj_matrix(p_ee, num_e, num_e) * w_ee
        W_ei = create_adj_matrix(p_ei, num_e, num_i) * -w_ei
        W_ie = create_adj_matrix(p_ie, num_i, num_e) * w_ie
        W_ii = create_adj_matrix(p_ii, num_i, num_i) * -w_ii
        
        # Concat
        W_1 = torch.cat(tensors=(W_ee, W_ei), dim=1)
        W_2 = torch.cat(tensors=(W_ie, W_ii), dim=1)
        W_rec = torch.cat(tensors=(W_1, W_2), dim=0)
    
    return W_rec

# Computes activation threshold for neurons, right now set it at 0
def compute_threshold():
    threshold = torch.zeros((num_neurons, P), device=gpu)
    # Since inhibitory neurons are linear
    threshold[num_e:, :] = 0
    return threshold

# ReLU for excitatory, linear for inhibitory
def neuron_activations(X):
    # Mask to keep excitatory
    mask1 = torch.ones((num_neurons, 1), device=gpu)
    mask1[num_e:, :] = 0
    # Mask to keep inhibitory
    mask2 = torch.zeros((num_neurons, 1), device=gpu)
    mask2[num_e:, :] = 1
    return (torch.relu(X) * mask1) + (X * mask2)

# %%
# Computes R for each odor, with the activation threshold theta
def compute_piriform_response(h_bar_ff, W_rec):
    # The coefficient of x_bar
    tau = 1
    # time step
    dt = 0.1
    # Number of time steps
    T = 200
    
    # Initial condition where states are gaussian
    mu_0 = 0.
    sigma_0 = 0.2
    X_0 = torch.normal(mu_0, sigma_0, size=(num_neurons, P))
    X = X_0.to(gpu)
    
    pts = []
    for i in range(T-2):
        with torch.no_grad():
            part1 = -1 * X
            part2 = (W_rec @ neuron_activations(X)) * (1 / np.sqrt(k))
            part3 = h_bar_ff
            dXdt = (1 / tau) * (part1 + part2 + part3)
            X = X + (dXdt * dt)
        # Look at convergence pattern for first odor, assuming that it'll
        # be similar across odors (since they are all independent)
        #pts.append(torch.mean(dXdt, dim=0)[0].item())
   
    # On the last 2 iterations only, track the gradient
    X.requires_grad_(True)
    
    for j in range(2):
        part1 = -1 * X
        part2 = (W_rec @ neuron_activations(X)) * (1 / np.sqrt(k))
        part3 = h_bar_ff
        dXdt = (1 / tau) * (part1 + part2 + part3)
        X = X + (dXdt * dt)
    
    # The total input to the neuron at this last time step (should be equivalent to the resulting value of X after this time step, since dxdt = 0 after the recurrent network converges)
    #total_input = part2 + part3
    threshold = compute_threshold()
    
    # Plot derivatives to see if state converged
    # plt.plot(torch.arange(T-2), pts)
    # plt.show()
    R = neuron_activations(X - threshold)
    
    return R

# Start and stop indices for the section of W_rec we want to update, respectively 
# Takes in R_alpha, a vector of neuron responses to a particular odor
def compute_update(model: torch.nn.Sequential, R_alpha: torch.Tensor, update_inds) -> torch.Tensor:
    # Compute the same pairs of R_i and R_j for every odor
    # 1D vectors of the pre and postsynaptic neurons corresponding to the nonzero weights of W_rec
    postsyn_responses = R_alpha[update_inds[0]]
    presyn_responses = R_alpha[update_inds[1]]
    model_input = torch.vstack((presyn_responses, postsyn_responses)).t()
    slice_updates = model(model_input)
    
    updates = slice_updates.squeeze(dim=1)
    
    return updates

# %%
def odor_corrs(R):
    # We don't care about the actual responses per odor, just about a neuron's fluctuations around its mean response across odors
    R_adjusted = R[:num_e, familiar_inds] - torch.mean(R[:num_e, familiar_inds], dim=1, keepdim=True)
    # Each odor becomes a variable, because we want to calculate correlations between them across neurons
    R_adjusted.t_()
    # Like cov but divides by standard deviations, effectively normalizing the values (the diagonals of the resulting matrix become 1)
    sigma_E = torch.corrcoef(R_adjusted)
    # We only care about the correlations between the familiar odors
    familiar_corrs = sigma_E - torch.eye(P // 2, device=gpu)
    corr_sum = torch.sum(familiar_corrs ** 2)
    avg_corr = torch.mean(torch.abs(familiar_corrs))
    
    return corr_sum, avg_corr

# %%
# Sparsity per odor, across all (E) neurons
def sparsity_per_odor(R):
    # Epsilon for if we have zero responses
    eps = 1e-6
    sp_per_odor = 1 - ((torch.sum(R[:num_e], dim=0) ** 2 + eps) / (num_e * (torch.sum(R[:num_e] ** 2, dim=0)) + eps))
    # Sparsity nan means that the responses were all 0 for an odor, meaning that its max sparsity of 1
    return sp_per_odor

# Sparsity per (E) neuron, across a given odor family
def sparsity_per_neuron(R, odor_inds):
    sp_per_neuron = 1 - (
                (torch.sum(R[:num_e, odor_inds], dim=1) ** 2) / ((P // 2) * torch.sum(R[:num_e, odor_inds] ** 2, dim=1)))
    return sp_per_neuron

# Try to minimize the correlations between values
def loss_fn(R, lambda_corr, lambda_mu, lambda_var, lambda_sp, R_initial=torch.empty((num_neurons, P)), do_print=True):
    corr_sum, avg_corr = odor_corrs(R)
    corr_loss = (1 / P) * corr_sum
    corr_term = lambda_corr * corr_loss
    
    # means = torch.mean(R, dim=0)
    # means_novel = torch.mean(means[novel_inds])
    # means_familiar = torch.mean(means[familiar_inds])
    # mu_term = lambda_mu * (((means_familiar - means_novel) / (means_novel + means_familiar)) ** 2)
    
    # vars = torch.var(R, dim=0)
    # var_novel = torch.mean(vars[novel_inds])
    # var_familiar = torch.mean(vars[familiar_inds])
    # var_term = lambda_var * (((var_familiar - var_novel) / (var_novel + var_familiar)) ** 2)
    
    spars_familiar_initial = sparsity_per_odor(R_initial.detach())[familiar_inds]
    spars_familiar_curr = sparsity_per_odor(R)[familiar_inds]
    spars_term = lambda_sp * torch.sum(((spars_familiar_curr - spars_familiar_initial) / (spars_familiar_curr + spars_familiar_initial)) ** 2)
    
    # spars_novel = torch.mean(sparsities[novel_inds])
    # spars_familiar = torch.mean(sparsities[familiar_inds])
    # spars_term = lambda_sp * (((spars_familiar - spars_novel) / (spars_novel + spars_familiar)) ** 2)

    # sum_i = torch.sum(R[num_e:, :] ** 2) / (num_i * P)
    # i_term = lambda_i * sum_i
    
    if do_print:
        #print("Avg Corr: %.4f, Corr: %.4f, Mu: %.4f, Var: %.4f, Sparsity: %.4f" % (avg_corr, corr_term, mu_term, var_term, spars_term))
        print("Avg Corr: %.4f, Corr: %.4f, Sparsity: %.4f" % (avg_corr, corr_term, spars_term))
    loss = corr_term + spars_term
    return loss

def loss_after_odors(R_initial: torch.Tensor, W_rec: torch.Tensor, h_bar_ff: torch.Tensor, lambda_corr, lambda_mu, lambda_var, lambda_sp, do_print):   
    R_new = compute_piriform_response(h_bar_ff, W_rec)
    loss = loss_fn(R_new, lambda_corr, lambda_mu, lambda_var, lambda_sp, R_initial=R_initial, do_print=do_print)
    
    return loss, R_new

# To check theoretical minimum odor correlations, generate random gaussian matrix of shape (num_neurons, 8) and this is what the minimum correlation should be
# mu = torch.zeros((num_neurons, 16))
# std = torch.ones((num_neurons, 16))
# R_random = torch.normal(mu, std)
# loss_min = loss_fn(R_random)
# Theoretical min of 0.05

# def verify_initial_activities():
#     runs = 50
#     avg_corrs = torch.empty((runs,))
#     total_losses = torch.empty((runs,))
#     for i in range(runs):
#         with torch.no_grad():
#             a = correlated_mitral_activity()
#             h = compute_feedforward_activity(a)
#             w = compute_initial_recurrent_weights()
#             r = compute_piriform_response(h, w, 0)
#         total_loss, avg_corr = odor_corrs(r)
#         print(f"Loss: {total_loss.item()}, Avg Corr: {avg_corr}")
#         total_losses[i] = total_loss
#         avg_corrs[i] = avg_corr
#     plt.hist(avg_corrs, bins=15)
#     plt.show()

def get_update_inds(post, pre, W):
    weights_slice = W[post[0]:post[1], pre[0]:pre[1]]
    inds = torch.nonzero(weights_slice, as_tuple=True)
    update_inds = (inds[0] + post[0], inds[1] + pre[0])
    
    return update_inds

import torch.optim as optim
epochs_inner = 10000

# lambda_corr, lambda_i, lambda_mu, lambda_var, lambda_sp = 10, 1, 0, 0, 0
# 1e1 - better decorrelation but sparsity still goes up
# 0-5: 5e1
# 5-10: 1e2
lambda_corr, lambda_mu, lambda_var, lambda_sp = 1, 0, 0, 5e1

mult = 100
w_ie = 0.5
ie_max_weight = mult * w_ie
ie_min_weight = 0

w_ei = -0.2
ei_max_weight = 0
ei_min_weight = mult * w_ei

ie_post = (num_e, num_neurons)
ie_pre = (0, num_e)

ei_post = (0, num_e)
ei_pre = (num_e, num_neurons)

def train_model(I, W_ff, W_initial, r, snapshot_every=100):
    corrs = torch.zeros((epochs_inner,))
    
    #W_initial = compute_initial_recurrent_weights()
    W_trained = W_initial.clone().to(gpu)
    W_trained.requires_grad_(True)

    ie_update_inds = get_update_inds(ie_post, ie_pre, W_trained)
    ei_update_inds = get_update_inds(ei_post, ei_pre, W_trained)
    
    # Only update the relevant weights
    def w_hook(grad):
        new_grad = torch.zeros_like(grad)
        new_grad[ie_update_inds] = grad[ie_update_inds]
        return new_grad
        
    #optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    W_trained.register_hook(w_hook)

    optimizer = optim.Adam([W_trained], lr=1e-3)
    
    #I = correlated_mitral_activity()
    #W_ff = compute_feedforward_weights()
    hbar_ff = compute_feedforward_activity(W_ff, I)   
    R_initial = compute_piriform_response(hbar_ff, W_trained)
    R_trained = R_initial.clone()
    print(f"Initial loss: \t", end="")
    loss_fn(R_initial, lambda_corr, lambda_mu, lambda_var, lambda_sp, R_initial=R_initial)
    
    clamp_min = torch.zeros_like(W_trained)
    # I->E weights have lower negative bound
    clamp_min[ei_update_inds] = ei_min_weight
    # E->I weights are only positive
    clamp_min[ie_update_inds] = ie_min_weight
    clamp_max = torch.zeros_like(W_trained)
    # E->I weights have higher positive bound
    clamp_max[ie_update_inds] = ie_max_weight
    # I->E weights are only negative
    clamp_max[ei_update_inds] = ei_max_weight
    
    for i in range(epochs_inner):
        do_print=False
        if (i % 100 == 0):
            print(f"Epoch {i}: \t", end="")
            do_print = True
        loss, R_trained = loss_after_odors(R_initial, W_trained, hbar_ff, lambda_corr, lambda_mu, lambda_var, lambda_sp, do_print)
        
        corrs[i] = odor_corrs(R_trained)[1].item()
        
        if (i % snapshot_every) == 0:
            save_snapshot(r, i, W_trained, R_trained)
        
        loss.backward()   
        optimizer.step()
        optimizer.zero_grad()
        
        with torch.no_grad():
            W_trained.clamp_(min=clamp_min, max=clamp_max)
            
    return corrs, I, W_ff, W_initial, W_trained, R_initial, R_trained

def generate_spars_plot(fig, ax, sp_novel, sp_familiar, is_ei, is_trained):
    start = 0.5
    space = 1
    width = space / 2
    coords = [start, start+space]
    spread = (torch.rand(size=(P // 2,)) - 0.5) * width

    if is_ei is not None:
        ax[is_ei, is_trained].scatter(spread + coords[0], sp_novel, label="Novel")
        ax[is_ei, is_trained].scatter(spread + coords[1], sp_familiar, label="Familiar")
        ax[is_ei, is_trained].set_xticks(ticks=coords, labels=[])
        ax[is_ei, is_trained].set_xlim(left=coords[0]-(1.5*width), right=coords[1]+(1.5*width))
        ax[is_ei, is_trained].set_ylim(0, 1)
        ax[is_ei, is_trained].bar(coords, [torch.mean(sp_novel), torch.mean(sp_familiar)], width = width/2, alpha=0.5, color="green")
        ax[is_ei, is_trained].legend(loc="lower left")
    else:
        ax[is_trained].scatter(spread + coords[0], sp_novel, label="Novel")
        ax[is_trained].scatter(spread + coords[1], sp_familiar, label="Familiar")
        ax[is_trained].set_xticks(ticks=coords, labels=[])
        ax[is_trained].set_xlim(left=coords[0]-(1.5*width), right=coords[1]+(1.5*width))
        ax[is_trained].set_ylim(0, 1)
        ax[is_trained].bar(coords, [torch.mean(sp_novel), torch.mean(sp_familiar)], width = width/2, alpha=0.5, color="green")
        ax[is_trained].legend(loc="lower left")

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def save_snapshot(r, epoch, W, R):
    path = f'./ie/realization_{r}'
    subpath = f'{path}/data/snapshots'

    with torch.no_grad():
        os.makedirs(f"{subpath}/epoch_{epoch}", exist_ok=True)
        torch.save(W, f"{subpath}/epoch_{epoch}/W_trained.pt")
        torch.save(R, f"{subpath}/epoch_{epoch}/R_trained.pt")

# Save a particular training realization
def save_realization(I, W_ff, W_initial, r, snapshot_every=100):
    corrs, I, W_ff, W_initial, _, R_initial, R_trained = train_model(I, W_ff, W_initial, r, snapshot_every)

    with torch.no_grad():
        path = f'./ie/realization_{r}'
        os.makedirs(f'{path}/data', exist_ok=True)

        fig = plt.figure()
        plt.plot(torch.arange(epochs_inner), corrs)
        plt.xlabel("Epochs")
        plt.ylabel("Average correlation")
        plt.ylim(bottom=0.)
        plt.title(f"Odor correlation across epochs")
        fig.savefig(f"{path}/ie_corrs.png")
        plt.close()

        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
        ie_sp_odor_initial = sparsity_per_odor(R_initial.detach().cpu())
        ie_sp_odor_trained = sparsity_per_odor(R_trained.detach().cpu())
        ie_sp_novel_0 = ie_sp_odor_initial[novel_inds]
        ie_sp_familiar_0 = ie_sp_odor_initial[familiar_inds]
        ie_sp_novel_f = ie_sp_odor_trained[novel_inds]
        ie_sp_familiar_f = ie_sp_odor_trained[familiar_inds]
        generate_spars_plot(fig, ax, ie_sp_novel_0, ie_sp_familiar_0, None, 0)
        ax[0].set_title("E->I: Initial")
        generate_spars_plot(fig, ax, ie_sp_novel_f, ie_sp_familiar_f, None, 1)
        ax[1].set_title("E->I: Trained")
        fig.savefig(f"{path}/ie_spars.png")
        plt.close()
        
        # Save realization data
        torch.save(corrs, f"{path}/data/corrs.pt")
        torch.save(I, f"{path}/data/I.pt")
        torch.save(W_ff, f"{path}/data/W_ff.pt")
        # Only save initial weights/responses here, we save snapshots of the training elsewhere 
        torch.save(W_initial, f"{path}/data/W_initial.pt")
        torch.save(R_initial, f"{path}/data/R_initial.pt")


for i in range(0, 5):
    realization_type = f"../standard"
    ie_path = f"{realization_type}/ie/realization_{i}/data"
    W_initial = torch.load(f"{ie_path}/W_initial.pt")
    I = torch.load(f"{ie_path}/I.pt")
    W_ff = torch.load(f"{ie_path}/W_ff.pt")
    save_realization(I, W_ff, W_initial, r=i, snapshot_every=100)