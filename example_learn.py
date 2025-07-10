"""
Interactive sequence learning demo with a brain-inspired spiking network.

How it works:
-------------
1. User enters a short sequence of numbers (e.g. "2,5,7,5,2").
   Each number is represented by a neuron in the network.

2. The network is trained by repeatedly presenting this sequence.
   For every observed transition (e.g. from 2 to 5, 5 to 7, ...), the connection (synapse)
   between the corresponding neurons is strengthened – a simple Hebbian rule.

3. An animation shows how only the trained connections become stronger.

4. After training, the user can enter any number and the network will predict the most likely
   next number in the learned sequence, along with probabilities for each possible next step.

Try experimenting with different sequences and see how the predictions change!
"""

import numpy as np
from brainnet.model import build_brain_network

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from tqdm import tqdm

# ---- PARAMETERS ----
N = 10               # Number of neurons / numbers that can be represented
K_intra = N - 1      # Each neuron connects to all others (fully connected, no self-loop)
n_modules = 1
EPOCHS = 25          # Number of training repetitions to strengthen learning
LEARNING_RATE = 0.5  # Increment by which a transition is reinforced (learning rate)
FORGETTING_RATE = 0.1  # Decay rate for unused connections per epoch to simulate forgetting

# --- 1. User input for the number sequence ---
user_seq = input(f"Enter a sequence of numbers (0-{N-1}), separated by commas: ")
SEQ = [int(s.strip()) for s in user_seq.split(",") if s.strip().isdigit()]
if any(s >= N or s < 0 for s in SEQ):
    raise ValueError(f"Only numbers from 0 to {N-1} are allowed.")

# --- 2. Build network: fully connected, all excitatory neurons ---
positions, adj, modules, neuron_types, weights, colors_mod = build_brain_network(
    N=N, n_modules=n_modules, p_inhib=0.0, K_intra=K_intra
)

# --- 3. Train network: reinforce transitions and simulate forgetting ---
frames = []

for epoch in tqdm(range(EPOCHS), desc="Precomputing frames"):
    used_pairs = set()  # Track pairs used this epoch
    for t in range(1, len(SEQ)):
        pre = SEQ[t - 1]
        post = SEQ[t]
        if adj[pre, post]:
            # Strengthen observed transitions (Hebbian learning)
            weights[pre, post] += LEARNING_RATE
            weights[pre, post] = np.clip(weights[pre, post], 0.0, 2.0)
            used_pairs.add((pre, post))
    # Decay unused connections to simulate forgetting
    for i in range(N):
        for j in range(N):
            if adj[i, j] and (i, j) not in used_pairs:
                weights[i, j] *= (1 - FORGETTING_RATE)
    frames.append(weights.copy())  # Save weights for animation

# --- 4. Build graph and layout once ---
G = nx.DiGraph()
for i in range(N):
    G.add_node(i)
for i in range(N):
    for j in range(N):
        if adj[i, j]:
            G.add_edge(i, j)
pos = nx.spring_layout(G, seed=2)  # Consistent layout for visualization

xy = np.array([pos[i] for i in range(N)])  # Node coordinates for plotting

fig, ax = plt.subplots(figsize=(6, 5))
ax.set_axis_off()
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

# --- Prepare all edge lines for efficient plotting ---
all_lines = []
for i, j in G.edges():
    all_lines.append([xy[i], xy[j]])
all_lines = np.array(all_lines)

# Create LineCollections: all edges gray initially, strengthened edges red
line_coll = LineCollection(all_lines, colors="#cccccc", linewidths=1, zorder=1, alpha=0.45)
ax.add_collection(line_coll)
strength_coll = LineCollection(all_lines, colors="#e57373", linewidths=1, zorder=2, alpha=0.95)
ax.add_collection(strength_coll)

nodes = None  # Placeholder for node scatter plot, updated in animation
labels = []   # Placeholder for text labels, updated in animation

def animate(frame_idx):
    global nodes, labels
    w = frames[frame_idx]

    # Update edge widths and colors based on current weights
    strengths = []
    colors = []
    THR = 0.03  # Threshold to show strengthened connections
    max_weight = LEARNING_RATE * EPOCHS
    for n, (i, j) in enumerate(G.edges()):
        weight = w[i, j]
        if weight > THR:
            width = 1.5 + 14 * weight / max_weight  # Scale line width by weight
            strengths.append(width)
            colors.append("#e57373")  # Red color for strengthened
        else:
            strengths.append(0.3)  # Thin and nearly invisible for weak connections
            colors.append((0, 0, 0, 0))  # Fully transparent
    strength_coll.set_linewidths(strengths)
    strength_coll.set_colors(colors)

    # Remove previous nodes and labels before drawing new ones
    if nodes:
        nodes.remove()
    for label in labels:
        label.remove()
    labels.clear()

    # Draw nodes on top of edges with high zorder
    nodes = ax.scatter(xy[:, 0], xy[:, 1], c="#ffd700", s=550, zorder=10)

    # Draw node labels (numbers) above nodes
    for i, (x, y) in enumerate(xy):
        txt = ax.text(x, y, str(i), color='k', fontsize=14,
                      ha='center', va='center', zorder=20, weight='bold')
        labels.append(txt)

    ax.set_title(f"Learning transitions (Epoch {frame_idx+1}/{EPOCHS})\nRed = strengthened",
                 fontsize=15)

    return (strength_coll, nodes) + tuple(labels)


ani = FuncAnimation(fig, animate, frames=len(frames), interval=300, blit=True, repeat=False)
plt.show()

# --- 5. After training: allow user to query next number predictions ---
while True:
    n = input(f"\nEnter a number (0-{N-1}) to predict the next in your sequence, or q to quit: ")
    if n.lower().strip() == 'q':
        print("Exiting prediction demo.")
        break
    try:
        n = int(n)
        if not (0 <= n < N):
            print(f"Please enter a number from 0 to {N-1}.")
            continue
    except ValueError:
        print("Please enter a valid number or 'q' to quit.")
        continue

    out_weights = weights[n]
    possible_next = np.where(adj[n])[0]
    pos_weights = np.clip(out_weights[possible_next], a_min=0, a_max=None)
    if pos_weights.sum() == 0:
        print("No learned positive transitions from this number.")
    else:
        probs = pos_weights / pos_weights.sum()
        print(f"Next-number probabilities from {n}:")
        for idx, p in zip(possible_next, probs):
            print(f"  {idx}: {p:.2f}")
        best_next = possible_next[np.argmax(probs)]
        print(f"Prediction: Most likely next number is {best_next}")