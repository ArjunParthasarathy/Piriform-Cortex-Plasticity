import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch.optim as optim

gpu = torch.device("cuda:0")
print(torch.cuda.get_device_name(0))

# Use smaller network for testing - ex 2000 neurons
# Even for the project, doing it for 10^6 neurons would take too long
# Problem this creates: test network is denser than actual network b/c we have 10^3 neurons but 10^2 connections per neuron
num_neurons = 2000
num_i = int(0.1 * num_neurons)
num_e = int(0.9 * num_neurons)

# Epsilon value close to 0 to prevent nan in division by 0
eps = 1e-6

# Num excitatory inputs and inhibitory inputs to each neuron (in reality it should be 500 but we reduce it here to make things faster)
k = 100

# Number of olfactory bulb channels (glomeruli) to each neuron
D = 10 ** 3
# For each neuron, how many glomeruli inputs it receives (should be 10^2)
num_channel_inputs = 100

# Number of odors
P = 16
# Novel activity is up to P // 2, and familiar activity is after
novel_inds = torch.arange(0, P // 2)
familiar_inds = torch.arange(P // 2, P)

# %%
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

# New way of generating correlations between odors: we want different sets of odors to be correlated differently, so that when we subtract each neuron's mean activity over odors, it doesn't cancel out the variation between odors (if all the odors are correlated the same, they will tend to produce similar values for a single neuron and therefore subtracting by the mean will remove these values and only leave small fluctuations)
# So we sample a small set of odors P' and make them linearly independent, and then by multiplying by a P'x P gaussian matrix we project into mitral cell activity space for all P odors, basically making the P odors a linear combination of the set of P' odors (the smaller P' is, the more correlated the resulting set of P odors will be)
# We also scale the variance depending on how small P' is, so we will maintain differently correlated odors, just with higher total correlation if P' is small
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

# Degree of polynomial
degree = 2
# Num vars in polynomial (presyn, postsyn, weight)
numvars = 3
# Num terms in polynomial is numvars ^ (degree + 1)
numterms = numvars ** (degree + 1)

def init_powerseries():
    # Confavreux paper: N(0, 0.1) = N(0, 0.3162^2)
    mu = torch.zeros(numterms)
    std = torch.ones(numterms) * 0.3162
    A = torch.normal(mu, std).to(gpu)
    A.requires_grad_(True)

    return A

def powerseries(A, W_rec, R, update_inds):
    post = R[update_inds[0], :]
    pre = R[update_inds[1], :]
    ws = torch.repeat_interleave(W_rec[update_inds[0], update_inds[1]].unsqueeze(1), repeats=P, dim=1)
    post = post.expand(numterms, -1, -1)
    pre = pre.expand(numterms, -1, -1)
    #print(post.shape)
    #ws = torch.repeat_interleave(ws.unsqueeze(0), repeats=numterms, dim=0)
    ws = ws.expand(numterms, -1, -1)
    combined = torch.stack((post, pre, ws), dim=1)
    exps = torch.cartesian_prod(*([torch.arange(degree + 1)] * numvars)).to(gpu)
    exps = exps.unsqueeze(2).unsqueeze(3)
    exps = exps.broadcast_to(combined.shape)
    result = torch.pow(combined, exps)
    sum_terms = torch.prod(result, dim=1)
    A_coef = A.unsqueeze(1).unsqueeze(2)
    result = A_coef * sum_terms
    updates_per_odor = torch.sum(result, dim=0)
    updates = torch.mean(updates_per_odor, dim=1)

    return updates

# %%
# Start and stop indices for the section of W_rec we want to update, respectively 
# Takes in R matrix (neuron responses for each odor and tuple of update inds representing ie, then ei (each element in that tuple is itself a tuple of (post, pre))
def compute_updates(A, W_rec, R: torch.Tensor, update_inds: tuple) -> torch.Tensor:
    updates = powerseries(A, W_rec, R, update_inds)
    
    return updates

# %%
def odor_corrs(R):
    # We don't care about the actual responses per odor, just about a neuron's fluctuations around its mean response across odors
    R_adjusted = R[:num_e] - torch.mean(R[:num_e], dim=1, keepdim=True)
    # Each odor becomes a variable, because we want to calculate correlations between them across neurons
    R_adjusted.t_()
    # Like cov but divides by standard deviations, effectively normalizing the values (the diagonals of the resulting matrix become 1)
    corrcoefs = torch.corrcoef(R_adjusted)
    # If the responses are 0, variances across neurons will be 0, so denominator of corrcoef is 0, so term becomes nan
    # In this case, the responses are "perfectly correlated" (bc always same value of 0) so its maximum correlation
    # TODO do we need to change this NaN formulation so we propagate grad correctly?
    corrcoefs = torch.nan_to_num(corrcoefs, nan=1.0)
    # We only care about the correlations between the familiar odors
    familiar_corrs = corrcoefs[P//2:P, P//2:P] - torch.eye(P // 2, device=gpu)
    corr_sum = torch.sum(familiar_corrs ** 2)
    avg_corr = torch.mean(torch.abs(familiar_corrs))
    
    return corr_sum, avg_corr

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
def loss_fn(R, W, ie_update_inds, ei_update_inds, lambda_corr, lambda_w, lambda_mu, lambda_var, lambda_sp, do_print=True):
    corr_sum, avg_corr = odor_corrs(R)
    corr_loss = (1 / ((P // 2) ** 2)) * corr_sum
    corr_term = lambda_corr * corr_loss
    
    means = torch.mean(R[:num_e], dim=0)
    means_novel = torch.mean(means[novel_inds])
    means_familiar = torch.mean(means[familiar_inds])
    if torch.abs(means_novel + means_familiar) < eps:
        # All means are the same so there's technically no loss
        mu_term = 0
    else:
        mu_term = lambda_mu * (((means_familiar - means_novel + eps) / (means_novel + means_familiar)) ** 2)
    
    vars = torch.var(R[:num_e], dim=0)
    var_novel = torch.mean(vars[novel_inds])
    var_familiar = torch.mean(vars[familiar_inds])
    if torch.abs(var_novel + var_familiar) < eps:
        # All variances are the same so there's technically no loss
        var_term = 0
    else:
        var_term = lambda_var * (((var_familiar - var_novel) / (var_novel + var_familiar)) ** 2)
    
    sparsities = sparsity_per_odor(R)
    spars_novel = torch.mean(sparsities[novel_inds])
    spars_familiar = torch.mean(sparsities[familiar_inds])
    if torch.abs(spars_novel + spars_familiar) < eps:
        # Sparsities are technically the same so the term shouldn't contribute to loss
        spars_term = 0
    else:
        spars_term = lambda_sp * (((spars_familiar - spars_novel) / (spars_novel + spars_familiar)) ** 2)

    # Multiply by 1/P^2 for the decorrelation term and by 1 / num_e*K for the (EI) weight regularization term and 1/num_i*K for the (IE) weight regularization term
    # Track each term independently to make sure they're on the same scale
    # Do it for backprop too 

    ie_weight_reg = torch.sum((W[ie_update_inds] - w_ie) ** 2) / (num_i * k)
    ei_weight_reg = torch.sum((W[ei_update_inds] - w_ei) ** 2) / (num_e * k)

    # IE and EI have same weight regularization term for now
    ie_weight_term = lambda_w * ie_weight_reg
    ei_weight_term = lambda_w * ei_weight_reg
    
    # Loss is squared norm of excitatory responses to test I->E gradient propagation
    # loss = torch.sum(R[:num_e, familiar_inds] ** 2)
    # if do_print:
    #     print(f"E squared norm: {loss.item()}")

    if do_print:
        #print("Avg Corr: %.4f, Corr: %.4f, Mu: %.4f, Var: %.4f, Sparsity: %.4f" % (avg_corr, corr_term, mu_term, var_term, spars_term))
        print("Avg Corr: %.4f, Corr: %.4f, Sparsity: %.4f, IE: %.4f, EI: %.4f" % (avg_corr, corr_term, spars_term, ie_weight_term, ei_weight_term))
        

    loss = corr_term + mu_term + var_term + spars_term + ie_weight_term + ei_weight_term
    #loss = corr_term
    return loss

# %%
def loss_after_odors(A: torch.Tensor, ie_update_inds, ei_update_inds, W_rec: torch.Tensor, R_current: torch.Tensor, h_bar_ff: torch.Tensor, plasticity_ie, plasticity_ei, weight_decay_rate, weight_range: tuple, lambda_corr, lambda_w, lambda_mu, lambda_var, lambda_sp, detach_grad=True, with_loss=False):
   # First, compute the respective weight updates through the novel and familiar odors from the current neural responses (which are from the current weight matrix)
    
    W_rec.requires_grad_(True)
    
    # updates = compute_updates(A, W_rec, R_current, ie_update_inds)
    # ie_updates = plasticity_ei * (updates)

    updates = compute_updates(A, W_rec, R_current, ei_update_inds)
    ei_updates = plasticity_ei * (updates)

    # with torch.no_grad():
    #     val_tensor = ((1 - plasticity_rate * weight_decay_rate) * W_rec[update_inds]) + updates + (plasticity_rate * odor_update)
    #     condition = torch.logical_and(torch.gt(val_tensor, min_weight), torch.le(val_tensor, max_weight))
    
    # ie_updates = plasticity_ie * (ie_model_updates - weight_decay_rate * W_rec[ie_update_inds])
    # ei_updates = plasticity_ei * (ei_model_updates - weight_decay_rate * W_rec[ei_update_inds])
    
    
    #ie_cond = torch.logical_and(torch.ge(W_rec[ie_update_inds] + ie_updates, weight_range[0][ie_update_inds]), torch.le(W_rec[ie_update_inds] + ie_updates, weight_range[1][ie_update_inds]))
    ei_cond = torch.logical_and(torch.ge(W_rec[ei_update_inds] + ei_updates, weight_range[0][ei_update_inds]), torch.le(W_rec[ei_update_inds] + ei_updates, weight_range[1][ei_update_inds]))
    
    # TODO clamping updates instead of the weights post-update - change in gradient?
    # TODO if we clamp to bounds then we'll see changes in correlation, b/c there is a change in weight instead of none at all
    #ie_bounded_updates = torch.where(ie_cond, ie_updates, 0)
    ei_bounded_updates = torch.where(ei_cond, ei_updates, 0)
   
    # # TODO penalize model for causing weight to go over threshold
    # # Amount below min
    # ie_a = torch.relu(weight_range[0][ie_update_inds] - (W_rec[ie_update_inds] + ie_updates))
    # # Amount above max
    # ie_b = torch.relu((W_rec[ie_update_inds] + ie_updates) - weight_range[1][ie_update_inds])
    # ie_over_weight = torch.mean(ie_a + ie_b)
    # # Amount below min
    # ei_a = torch.relu(weight_range[0][ei_update_inds] - (W_rec[ei_update_inds] + ei_updates))
    # # Amount above max
    # ei_b = torch.relu((W_rec[ei_update_inds] + ei_updates) - weight_range[1][ei_update_inds])
    # ei_over_weight = torch.mean(ei_a + ei_b)

    # print(f"IE: {torch.mean(ie_updates)}: {ie_over_weight}")
    # print(f"EI: {torch.mean(ei_updates)}: {ei_over_weight}")
    
    #W_rec = W_rec.clamp(min=weight_range[0], max=weight_range[1])
    
    #W_rec = torch.index_put(W_rec, ie_update_inds, ie_bounded_updates, accumulate=True)
    W_rec = torch.index_put(W_rec, ei_update_inds, ei_bounded_updates, accumulate=True)
    
    
    R = compute_piriform_response(h_bar_ff, W_rec)
    R_new = R
    
    if detach_grad:
        # We want a new R response tensor which only has the previous weight update but not anything before that
        W_rec = W_rec.detach()
        R_new = compute_piriform_response(h_bar_ff, W_rec)

    if with_loss:
        loss = loss_fn(R, W_rec, ie_update_inds, ei_update_inds, lambda_corr, lambda_w, lambda_mu, lambda_var, lambda_sp, do_print=True)
    else:
        loss = 0
    
    return loss, W_rec, R_new


def get_update_inds(post, pre, W):
    weights_slice = W[post[0]:post[1], pre[0]:pre[1]]
    inds = torch.nonzero(weights_slice, as_tuple=True)
    update_inds = (inds[0] + post[0], inds[1] + pre[0])
    
    return update_inds


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

# def test_regime(ie_val, ei_val):
#     runs = 100
#     loss_ratios = torch.empty((runs,))
#     for i in range(runs):
#         with torch.no_grad():
#             I = correlated_mitral_activity()
#             hbar_ff = compute_feedforward_activity(I)
#             W = compute_initial_recurrent_weights()
#             R = compute_piriform_response(hbar_ff, W, 0)
#             ie_update_inds = get_update_inds(ie_post, ie_pre, W)
#             ei_update_inds = get_update_inds(ei_post, ei_pre, W)
#             initial_loss = loss_fn(R, W, ie_update_inds, ei_update_inds, 1, 0, 0, 0, 0, do_print=False)
#             ie_update_inds = get_update_inds(ie_post, ie_pre, W)
#             ei_update_inds = get_update_inds(ei_post, ei_pre, W)
#             W[ie_update_inds] = ie_val
#             W[ei_update_inds] = ei_val
#             R_0 = compute_piriform_response(hbar_ff, W, 0)
#             final_loss = loss_fn(R_0, W, ie_update_inds, ei_update_inds, 1, 0, 0, 0, 0, do_print=False)
        
#         total_loss = final_loss / initial_loss
#         #print(f"Loss Ratio: {total_loss.item()}")
#         loss_ratios[i] = total_loss
        
#     plt.hist(loss_ratios, bins=15)

# Both zero
# test_regime(0., 0.)
# plt.title("Loss Ratios - both 0")
# plt.show()
# # IE cranked
# test_regime(ie_max_weight, w_ei)
# plt.title("Loss Ratios - E->I cranked")
# plt.show()
# # EI cranked
# test_regime(w_ie, ei_min_weight)
# plt.title("Loss Ratios - I->E cranked")
# plt.show()
# # Both cranked
# test_regime(ie_max_weight, ei_min_weight)
# plt.title("Loss Ratios - both cranked")
# plt.show()

# Number of independent "sniffs" of the 16 odors
n_inner = 200
# Number of different realizations of the odor sniffing process
n_outer = 1

# TODO change back to -3 if too noisy
plasticity_rate = 1e-3
# Same plasticity rate for EI and IE
plasticity_ie = plasticity_ei = plasticity_rate

# TODO experiment w/ diff gradient tracking numbers
# Number of inner epochs between model updates, n_update <= n_inner
n_update = 1
# Number of inner epochs across which the gradient is tracked (right now we detach the gradient after each inner epoch), n_track <= n_inner
n_track = n_update

# n_track = n_update to test formulation where we track the gradient across a subset of the inner epochs and update the model
# If n_update > n_track, the model will have multiple "gradient" paths of loss accumulated: ex. inner epochs 1-5 and a separate branch of 6-10
# If n_update < n_track, it doesn't matter b/c the model will truncate the history past its previous update since the gradient is zeroed

# TODO for now go to simple formulation, no weight decay
weight_decay = 0
# How much to weight each of the regularization terms
# Sparsity = 1000 doesn't improve loss ratios
# Sparsity = 100, 500 epochs, loss ratios 1.0-1.8, blows up sparsity
# Go back to 500 epochs no sparsity reg, w/ new nan clipping formulation
#lambda_corr, lambda_w, lambda_mu, lambda_var, lambda_sp = 10, 1, 0, 0, 0
lambda_corr, lambda_w, lambda_mu, lambda_var, lambda_sp = 1, 0, 0, 0, 0


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

coefs = init_powerseries()
print(coefs)

def train_model():
    #torch.autograd.set_detect_anomaly(True)
    
    # TODO try larger learning rate
    # TODO problem was adaptive threshold - as we tried to boost the weights, the threshold for the response would change too
    # So it would negate the effect of reducing the response
    #ie_optim = optim.SGD([coefs], lr=1e0, momentum=0.9)
    # 1e-2 too high
    ei_optim = optim.SGD([coefs], lr=1e-3, momentum=0.9)
    
    updates_per_outer = n_inner // n_update
    num_losses = int(updates_per_outer * n_outer)
    losses = torch.empty(size=(num_losses,))
    # batch every 20 (ex. tried 50 and was slightly more noisy (but w/ similar mean))
    # batch_every = 20
    # batch_loss = 0
    
    for outer_e in range(n_outer):
        i = correlated_mitral_activity()
        w_ff = compute_feedforward_weights()
        hbar_ff = compute_feedforward_activity(w_ff, i)
        
        W_initial = compute_initial_recurrent_weights()
        W = W_initial.clone().to(gpu)
        W.requires_grad_(True)
        
        with torch.no_grad():
            ie_update_inds = get_update_inds(ie_post, ie_pre, W)
            ei_update_inds = get_update_inds(ei_post, ei_pre, W)
            
            clamp_min = torch.zeros_like(W)
            clamp_min[ei_update_inds] = ei_min_weight
            clamp_min[ie_update_inds] = ie_min_weight
            clamp_max = torch.zeros_like(W)
            clamp_max[ie_update_inds] = ie_max_weight
            clamp_max[ei_update_inds] = ei_max_weight
            weight_range = (clamp_min, clamp_max)
        
        # Initial neuron responses
        R = compute_piriform_response(hbar_ff, W)
        
        for i in range(1, n_inner + 1):
            with_loss = False
            detach_grad = False
            if i % n_update == 0:
                with_loss = True
                print(f"Outer epoch {outer_e}, Inner epoch {i}, Loss: \t", end="")
            if i % n_track == 0:
                detach_grad = True
            
            loss, W, R = loss_after_odors(coefs, ie_update_inds, ei_update_inds, W, R, hbar_ff, plasticity_ie, plasticity_ei, weight_decay, weight_range, lambda_corr, lambda_w, lambda_mu, lambda_var, lambda_sp, detach_grad=detach_grad, with_loss=with_loss)
            if with_loss:
                final_loss = loss

                R_initial = compute_piriform_response(hbar_ff, W_initial)
                initial_loss = loss_fn(R_initial, W_initial, ie_update_inds, ei_update_inds, lambda_corr, lambda_w, lambda_mu, lambda_var, lambda_sp, do_print=False)
                total_loss = final_loss / initial_loss
                #total_loss += overload
                #print(f"Loss ratio: {total_loss}")
                losses[(outer_e * updates_per_outer)  + (i // n_update) - 1] = total_loss.item()
                total_loss.backward()
                #batch_loss += total_loss
                # ie_grad = torch.nn.utils.clip_grad_norm_(ie_model.parameters(), max_norm = 1e5)
                # ei_grad = torch.nn.utils.clip_grad_norm_(ei_model.parameters(), max_norm = 1e5)
                # print(f"ie model grad: {ie_grad}")
                # print(f"ei model grad: {ei_grad}")
                # ie_optim.step()  
                # ei_optim.step()
                # ie_optim.zero_grad()
                # ei_optim.zero_grad()
                ei_optim.step()
                ei_optim.zero_grad()
            
        # if outer_e % batch_every == 0:
        #     batch_loss.backward()   
        #     ie_optim.step()
        #     ei_optim.step()
        #     ie_optim.zero_grad()
        #     ei_optim.zero_grad()
        #     batch_loss = 0
                
    return losses, coefs, W[ei_update_inds]

#losses, coefs, weights = train_model()
# torch.save(coefs, "coefs.pt")
# with torch.no_grad():
#     fig = plt.figure()
#     plt.plot(torch.arange(losses.shape[0]), losses)
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss ratio")
#     plt.title("Loss ratio per realization over time")
#     plt.savefig("loss.png")
#     plt.close()

#     print(coefs)
#     fig = plt.figure()
#     plt.hist(weights.cpu())
#     plt.savefig("weights.png")
#     plt.close()


# def run_realization(coefs, hbar_ff, W_initial, ie_update_inds, ei_update_inds, weight_range):
#     W = W_initial
#     for _ in range(n_inner):
#         _, W, R = loss_after_odors(coefs, ie_update_inds, ei_update_inds, W, R, hbar_ff, plasticity_ie, plasticity_ei, weight_decay, weight_range, 0, 0, 0, 0, 0, detach_grad=True, with_loss=False)

#     return W, R

# 3d colormap - 3 input variables for 1 output plasticity
def make_colormap():
    coefs = torch.load("coefs.pt").detach().cpu()
    R = torch.load("r_final.pt")
    W = torch.load("w_final")
    

    num_ticks = 100
    E_min, E_max = torch.min(R[:num_e, :]), torch.max(R[:num_e, :])
    I_min, I_max = torch.min(R[num_e:, :]), torch.max(R[num_e:, :])
    E_vals = torch.linspace(E_min, E_max, num_ticks)
    I_vals = torch.linspace(I_min, I_max, num_ticks)
    E_coords, I_coords = torch.meshgrid(E_vals, I_vals, indexing="ij")
    R_plot = 0
    if type == "ie":
        R_plot = torch.stack((E_coords.flatten(), I_coords.flatten()), dim=1)
    elif type == "ei":
        R_plot = torch.stack((I_coords.flatten(), E_coords.flatten()), dim=1)
    
    with torch.no_grad():
        plasticity_vals = powerseries(A, R_plot.to(gpu)).squeeze(0).cpu()
        
    plot = plt.scatter(R_plot[:, 0], R_plot[:, 1], c=plasticity_vals, cmap='rainbow')
    clrbar = plt.colorbar(plot)
    clrbar.set_label('Model plasticity')
    plt.xlabel("E responses")
    plt.ylabel("I responses")
        
    return E_min, E_max, I_min, I_max, R_plot, plasticity_vals


with torch.no_grad():
    ie_update_inds = get_update_inds(ie_post, ie_pre, w_initial)
    ei_update_inds = get_update_inds(ei_post, ei_pre, w_initial)
    
    clamp_min = torch.zeros_like(w_initial)
    clamp_min[ei_update_inds] = ei_min_weight
    clamp_min[ie_update_inds] = ie_min_weight
    clamp_max = torch.zeros_like(w_initial)
    clamp_max[ie_update_inds] = ie_max_weight
    clamp_max[ei_update_inds] = ei_max_weight
    weight_range = (clamp_min, clamp_max)

w_ff = compute_feedforward_weights()
r_initial = compute_piriform_response()
make_colormap(coefs, )