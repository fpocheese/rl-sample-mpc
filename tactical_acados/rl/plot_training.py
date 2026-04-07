"""
Plot training curves comparing 3 RL variants.

Generates IEEE-quality figures:
  1. Episode reward over training (3 curves)
  2. Episode length over training
  3. Loss curves

Usage:
    python plot_training.py
"""

import os
import sys
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

dir_path = os.path.dirname(os.path.abspath(__file__))
tactical_dir = os.path.join(dir_path, '..')

# IEEE styling
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.5,
})

VARIANT_STYLES = {
    'A-oursrl': {'color': '#2196F3', 'label': 'A-Ours+RL (Advanced)', 'ls': '-'},
    'oursrl':   {'color': '#4CAF50', 'label': 'Ours+RL (Basic)',      'ls': '--'},
    'pure-rl':  {'color': '#FF9800', 'label': 'Pure RL',              'ls': '-.'},
}


def smooth(data, window=20):
    """Exponential moving average smoothing."""
    if len(data) < window:
        return data
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    smoothed = np.convolve(data, weights, mode='valid')
    # Pad start
    pad = np.full(len(data) - len(smoothed), smoothed[0])
    return np.concatenate([pad, smoothed])


def load_stats(variant):
    """Load training stats for a variant."""
    save_dir = os.path.join(tactical_dir, 'checkpoints', variant.replace('-', '_'))
    stats_path = os.path.join(save_dir, 'training_stats.json')
    if not os.path.exists(stats_path):
        print(f"  WARNING: No stats for {variant} at {stats_path}")
        return None
    with open(stats_path, 'r') as f:
        return json.load(f)


def plot_training_curves(save_path=None):
    """Plot 3-panel training curves."""
    if save_path is None:
        save_path = os.path.join(tactical_dir, 'checkpoints', 'training_curves.pdf')

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for variant, style in VARIANT_STYLES.items():
        stats = load_stats(variant)
        if stats is None:
            continue

        episodes = np.arange(len(stats['episode_reward']))
        rewards = np.array(stats['episode_reward'])
        lengths = np.array(stats['episode_length'])
        losses = np.array(stats.get('ppo_loss', [0] * len(rewards)))

        # Panel 1: Episode Reward
        ax = axes[0]
        ax.plot(episodes, smooth(rewards, 20), color=style['color'],
                ls=style['ls'], label=style['label'], alpha=0.9)
        ax.fill_between(episodes, smooth(rewards - np.std(rewards) * 0.3, 20),
                        smooth(rewards + np.std(rewards) * 0.3, 20),
                        color=style['color'], alpha=0.1)

        # Panel 2: Episode Length
        ax = axes[1]
        ax.plot(episodes, smooth(lengths, 20), color=style['color'],
                ls=style['ls'], label=style['label'], alpha=0.9)

        # Panel 3: PPO Loss
        ax = axes[2]
        if len(losses) > 0 and np.any(losses != 0):
            ax.plot(episodes, smooth(losses, 20), color=style['color'],
                    ls=style['ls'], label=style['label'], alpha=0.9)

    # Labels
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Episode Reward')
    axes[0].set_title('(a) Training Reward')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Episode Length')
    axes[1].set_title('(b) Episode Duration')
    axes[1].legend(loc='lower right')
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('PPO Loss')
    axes[2].set_title('(c) Training Loss')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    print(f"Training curves saved to {save_path}")
    plt.close()


def plot_curriculum_phases(save_path=None):
    """Plot curriculum phase progression for A-oursrl."""
    if save_path is None:
        save_path = os.path.join(tactical_dir, 'checkpoints', 'curriculum_phases.pdf')

    stats = load_stats('A-oursrl')
    if stats is None or 'curriculum_phase' not in stats:
        print("No curriculum data for A-oursrl")
        return

    fig, ax = plt.subplots(figsize=(8, 3))

    episodes = np.arange(len(stats['episode_reward']))
    rewards = np.array(stats['episode_reward'])
    phases = np.array(stats['curriculum_phase'])

    # Color by phase
    phase_colors = ['#E3F2FD', '#BBDEFB', '#90CAF9', '#64B5F6']
    phase_names = ['Phase 1: Solo', 'Phase 2: Single Opp',
                   'Phase 3: Full', 'Phase 4: Multi-Scenario']

    for p in range(4):
        mask = phases == p
        if mask.any():
            ax.fill_between(episodes, -200, 200, where=mask,
                            alpha=0.3, color=phase_colors[p],
                            label=phase_names[p])

    ax.plot(episodes, smooth(rewards, 15), color='#1565C0', lw=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('A-Ours+RL: Curriculum Learning Progression')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=min(rewards) - 10, top=max(rewards) + 10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.savefig(save_path.replace('.pdf', '.png'))
    print(f"Curriculum plot saved to {save_path}")
    plt.close()


if __name__ == '__main__':
    print("Generating training curve plots...")
    plot_training_curves()
    plot_curriculum_phases()
    print("Done.")
