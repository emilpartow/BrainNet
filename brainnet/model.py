"""
Model definition for brain-inspired spiking network.
Builds network, simulates spiking, returns all needed data for animation.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

def build_brain_network(N=180, n_modules=4, module_spread=0.6, K_intra=8, K_inter=0, p_inhib=0.2, random_seed=42):
    """
    Create neuron positions, modules, types, adjacency matrix, and weights.
    Returns: positions, adjacency matrix, modules, neuron_types, weights, module_colors
    """
    np.random.seed(random_seed)
    # Color palette for modules
    colors_mod = ['#ffa600', '#0080ff', '#43d94b', '#d443d9']
    module_sizes = [N // n_modules] * n_modules
    module_sizes[-1] += N - sum(module_sizes)
    positions, modules = [], []
    for i, size in enumerate(module_sizes):
        theta = 2 * np.pi * i / n_modules
        center = np.array([2 * np.cos(theta), 2 * np.sin(theta), 0.7 * (i % 2)])
        pos = np.random.normal(center, module_spread, (size, 3))
        positions.append(pos)
        modules.extend([i] * size)
    positions = np.vstack(positions)
    modules = np.array(modules)
    neuron_types = np.ones(N)
    inhib_idx = np.random.choice(N, int(p_inhib * N), replace=False)
    neuron_types[inhib_idx] = -1

    adj = np.zeros((N, N), dtype=bool)
    weights = np.zeros((N, N))
    for m in range(n_modules):
        idx = np.where(modules == m)[0]
        nbrs = NearestNeighbors(n_neighbors=min(K_intra + 1, len(idx)), algorithm='ball_tree').fit(positions[idx])
        _, indices = nbrs.kneighbors(positions[idx])
        for i, row in enumerate(indices):
            src = idx[i]
            for dst_j in row[1:]:
                dst = idx[dst_j]
                adj[src, dst] = True
                sign = neuron_types[src]
                base_weight = np.random.uniform(0.7, 1.4)
                weights[src, dst] = base_weight * sign
        others = np.where(modules != m)[0]
        for src in idx:
            if len(others) > 0 and K_inter > 0:
                to_others = np.random.choice(others, size=K_inter, replace=False)
                for dst in to_others:
                    adj[src, dst] = True
                    sign = neuron_types[src]
                    base_weight = np.random.uniform(0.25, 0.6)
                    weights[src, dst] = base_weight * sign
    return positions, adj, modules, neuron_types, weights, colors_mod

def run_spiking_simulation(positions, adj, neuron_types, weights, modules, n_modules, T=38, THRESH=1.8, REFRACTORY=3):
    """
    Simulate spike propagation and collect spike events and signals for all timesteps.
    Returns: spikes, spike_signals
    """
    N = positions.shape[0]
    spikes = np.zeros((T, N), dtype=bool)
    potentials = np.zeros(N)
    refrac = np.zeros(N, dtype=int)
    spike_signals = []
    max_dist = np.max(np.linalg.norm(positions - positions.mean(0), axis=1))

    def signal_time(i, j):
        """
        Calculate a biologically plausible signal delay (in animation frames) 
        for the connection from neuron i to neuron j.

        - Uses a minimum delay of 2 frames (even for short connections)
        - Delay increases with distance (but less than linearly)
        - Adds slight random jitter to simulate biological variability
        """
        d = np.linalg.norm(positions[i] - positions[j])    # Euclidean distance between neurons
        mean_delay = 2 + 4 * np.sqrt(d / max_dist)         # sqrt scaling: fast for local, slower for distant
        jitter = np.random.normal(loc=0, scale=0.3)        # small random biological "jitter"
        # Clamp the total delay to be between 2 and 10 frames
        return int(np.round(np.clip(mean_delay + jitter, 2, 10)))
    
    # Start: 2 random neurons per module fire at t=0
    for m in range(n_modules):
        idx = np.where(modules == m)[0]
        start_neurons = np.random.choice(idx, 2, replace=False)
        spikes[0, start_neurons] = True

    for t in range(1, T):
        inputs = np.zeros(N)
        for i in range(N):
            for pre in np.where(adj[:, i])[0]:
                if spikes[t - 1, pre]:
                    inputs[i] += weights[pre, i]
                    travel = signal_time(pre, i)
                    spike_signals.append([t - 1, pre, i, travel])
        potentials[refrac == 0] += inputs[refrac == 0]
        fire = (potentials >= THRESH) & (refrac == 0)
        spikes[t, fire] = True
        potentials[fire] = 0
        refrac[fire] = REFRACTORY
        refrac[refrac > 0] -= 1

    return spikes, spike_signals
