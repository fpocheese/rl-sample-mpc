import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# Path to the latest checkpoint that has stats
checkpoint_path = 'tactical_acados/checkpoints/checkpoint_5000.pt'
save_path = 'tactical_acados/checkpoints/reward_plot.png'

if not os.path.exists(checkpoint_path):
    # Try to find the latest checkpoint if 5000 doesn't exist
    checkpoints = [f for f in os.listdir('tactical_acados/checkpoints') if f.startswith('checkpoint_') and f.endswith('.pt')]
    if not checkpoints:
        print("No checkpoints found.")
        exit(1)
    # Sort by episode number
    checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    checkpoint_path = os.path.join('tactical_acados/checkpoints', checkpoints[-1])

print(f"Loading stats from {checkpoint_path}...")
data = torch.load(checkpoint_path, map_location='cpu')
stats = data.get('stats', {})

if not stats:
    print("No stats found in checkpoint.")
    exit(1)

rewards = stats.get('episode_reward', [])
lengths = stats.get('episode_length', [])

# Smoothing
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

plt.figure(figsize=(12, 10))

# Plot Rewards
plt.subplot(2, 1, 1)
plt.plot(rewards, color='blue', alpha=0.3, label='Raw Reward')
if len(rewards) > 50:
    plt.plot(smooth(rewards, 50), color='blue', linewidth=2, label='Smoothed (50)')
plt.title('RL Training: Episode Reward')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True, alpha=0.3)
plt.legend()

# Plot Episode Lengths
plt.subplot(2, 1, 2)
plt.plot(lengths, color='green', alpha=0.3, label='Raw Length')
if len(lengths) > 50:
    plt.plot(smooth(lengths, 50), color='green', linewidth=2, label='Smoothed (50)')
plt.title('RL Training: Episode Length')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(save_path)
print(f"Plot saved to {save_path}")
