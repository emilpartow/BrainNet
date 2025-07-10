"""
Visualization module for brain-inspired spiking network.
Animates network using precomputed frame states.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

def precompute_frames(
        positions, modules, neuron_types, spikes, spike_signals, colors_mod, T=38
    ):
    """
    Precompute all frame states for the animation.
    Returns: frame_states (list of dicts)
    """
    N = positions.shape[0]
    neuron_colors = np.array([
        "#e74c3c" if neuron_types[i] == -1 else colors_mod[modules[i]]
        for i in range(N)
    ])
    neuron_colors_faded = np.array([
        "#e74c3c33" if neuron_types[i] == -1 else colors_mod[modules[i]] + "33"
        for i in range(N)
    ])
    frame_states = []
    glow_decay = np.zeros(N)
    for frame in range(T):
        acts = spikes[frame]
        glow_decay = glow_decay * 0.85
        glow_decay[acts] = 1.0
        colors = np.copy(neuron_colors_faded)
        sizes = np.ones(N) * 30
        for i in range(N):
            if glow_decay[i] > 0.02:
                if neuron_types[i] == -1:
                    base = np.array([1.0, 0.5, 0.4])  # sanftes Orange-Rot
                    modc = np.array(mcolors.to_rgb("#e74c3c"))
                    c = 0.85 * base + 0.15 * modc
                else:
                    base = np.array([1.0, 1.0, 0.0])
                    modc = np.array(mcolors.to_rgb(colors_mod[modules[i]]))
                    c = 0.8 * base + 0.2 * modc
                c = c * glow_decay[i] + (1 - glow_decay[i]) * np.array(mcolors.to_rgb('#cccccc'))
                colors[i] = mcolors.to_hex(c)
            if acts[i]:
                sizes[i] = 120
        now = frame
        frame_signals = []
        for s in spike_signals:
            start_frame, src, dst, travel = s
            phase = now - start_frame
            if 0 <= phase <= travel:
                if neuron_types[src] == -1:
                    mod_col = np.array(mcolors.to_rgb("#e74c3c"))
                    blend = 0.55 * mod_col + 0.45 * np.array([1.0, 1.0, 1.0])
                else:
                    mod_col = np.array(mcolors.to_rgb(colors_mod[modules[dst]]))
                    blend = 0.45 * mod_col + 0.55 * np.array([1.0, 1.0, 1.0])
                soft_color = mcolors.to_hex(blend)
                p = positions[src] * (1 - phase / travel) + positions[dst] * (phase / travel)
                frame_signals.append({
                    'pos': p,
                    'src': src,
                    'color': soft_color,
                    'line_to': positions[src],
                    'alpha': 0.32
                })
        frame_states.append({
            'colors': colors,
            'sizes': sizes,
            'acts': acts,
            'signals': frame_signals
        })
    return frame_states

def plot_brain_activity(
        positions, modules, neuron_types, adj, colors_mod, frame_states, T=38
    ):
    """
    Create and show the animated 3D plot of the brain-like spiking network.
    """
    N = positions.shape[0]
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.suptitle("Brain-like Spiking Neural Network Simulation\nExcitatory neurons by module color, blue = inhibitory", fontsize=16, y=0.96)

    # Draw connections as Line3DCollection for performance
    lines = []
    colors_lines = []
    for i in range(N):
        for j in range(N):
            if adj[i, j]:
                lines.append([positions[i], positions[j]])
                if neuron_types[i] == -1:
                    colors_lines.append(mcolors.to_rgba("#e74c3c", 0.12))
                else:
                    if modules[i] == modules[j]:
                        colors_lines.append(mcolors.to_rgba(colors_mod[modules[i]], 0.09))
                    else:
                        colors_lines.append(mcolors.to_rgba("#cccccc", 0.08))
    lc = Line3DCollection(lines, colors=colors_lines, linewidths=0.5)
    ax.add_collection3d(lc)

    # Module legend patches
    module_ids = sorted(set(modules))
    legend_patches = [mpatches.Patch(color=colors_mod[i], label=f"Module {i+1} (excitatory)") for i in module_ids]
    legend_patches.append(mpatches.Patch(color="#e74c3c", label="Inhibitory neuron"))

    neuron_colors_faded = np.array([
        "#e74c3c33" if neuron_types[i] == -1 else colors_mod[modules[i]] + "33"
        for i in range(N)
    ])
    sc = ax.scatter(
        positions[:,0], positions[:,1], positions[:,2],
        c=neuron_colors_faded, s=32, zorder=10
    )
    active_signals = []

    def add_legend_once():
        if not getattr(add_legend_once, "added", False):
            ax.legend(handles=legend_patches, loc=(1.04,0.47), fontsize=12, frameon=False, title="Neuron Type")
            add_legend_once.added = True

    def update(frame):
        nonlocal active_signals
        while active_signals:
            l = active_signals.pop()
            l.remove()
        state = frame_states[frame]
        sc._offsets3d = (positions[:,0], positions[:,1], positions[:,2])
        sc.set_color(state['colors'])
        sc.set_sizes(state['sizes'])
        for sig in state['signals']:
            pt = ax.scatter(*sig['pos'], color=sig['color'], s=56, zorder=30, alpha=0.43)
            l, = ax.plot(
                [sig['line_to'][0], sig['pos'][0]],
                [sig['line_to'][1], sig['pos'][1]],
                [sig['line_to'][2], sig['pos'][2]],
                color=sig['color'], lw=1.3, alpha=sig['alpha'], zorder=9)
            active_signals.extend([pt, l])
        num_ex = np.sum(state['acts'] & (neuron_types == 1))
        num_in = np.sum(state['acts'] & (neuron_types == -1))
        ax.set_title(f"Step {frame+1} | Spiking: {num_ex} excitatory, {num_in} inhibitory", fontsize=13)
        ax.set_xlabel("X", fontsize=12, labelpad=8)
        ax.set_ylabel("Y", fontsize=12, labelpad=8)
        ax.set_zlabel("Z", fontsize=12, labelpad=8)
        add_legend_once()
        return [sc] + active_signals

    ax.set_axis_off()
    ani = FuncAnimation(fig, update, frames=T, interval=350, blit=False, repeat=True)
    plt.show()
