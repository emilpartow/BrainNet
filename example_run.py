"""
Example: Create, simulate and animate a modular brain-like spiking network.
"""

from brainnet.model import build_brain_network, run_spiking_simulation
from brainnet.animation import precompute_frames, plot_brain_activity
from tqdm import tqdm

# Parameters for the network
N = 50                # Total number of neurons in the network
n_modules = 3         # Number of distinct modules (clusters) in the network
module_spread = 0.6   # How tightly each module's neurons are grouped (higher = more spread out)
K_intra = 8           # Number of intra-module (within-module) connections per neuron
K_inter = 2           # Number of inter-module (between-module) connections per neuron
T = 30                # Number of simulation timesteps (animation frames)
THRESH = 1.8          # Spiking threshold (membrane potential needed to fire)
REFRACTORY = 3        # Refractory period (timesteps a neuron must wait after spiking)
p_inhib = 0.2         # Proportion of inhibitory neurons (e.g., 0.2 = 20% inhibitory)

# 1. Build network
positions, adj, modules, neuron_types, weights, colors_mod = build_brain_network(
    N=N, n_modules=n_modules, module_spread=module_spread,
    K_intra=K_intra, K_inter=K_inter, p_inhib=p_inhib
)

# 2. Run spike simulation
spikes, spike_signals = run_spiking_simulation(
    positions, adj, neuron_types, weights, modules, n_modules, T=T,
    THRESH=THRESH, REFRACTORY=REFRACTORY
)

# 3. Prepare all animation frames (for speed)
print("Preparing all animation frames ...")
frame_states = precompute_frames(
    positions, modules, neuron_types, spikes, spike_signals, colors_mod, T=T
)

# 4. Show animated plot
plot_brain_activity(
    positions, modules, neuron_types, adj, colors_mod, frame_states, T=T
)
