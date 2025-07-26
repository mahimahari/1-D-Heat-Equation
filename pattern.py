import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import os
import glob
import re
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# Parameters (must match training parameters)
Lx = 64 * np.pi       # Domain length
T = 5.0               # Total simulation time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model architecture (must match training)
class SH_PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, xt):
        x = xt[:, 0:1].clone().requires_grad_(True)
        t = xt[:, 1:2].clone().requires_grad_(True)
        fourier = torch.cat([torch.sin(x), torch.cos(x)], dim=1)
        return self.fc(torch.cat([x, t, fourier], dim=1))

def load_checkpoint(checkpoint_path, device):
    """Load a saved checkpoint"""
    model = SH_PINN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    losses = checkpoint['losses']
    return model, epoch, losses

def visualize_pattern_evolution(checkpoint_dir='checkpoints', output_dir='pattern_evolution'):
    """Visualize pattern formation at different epochs"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all checkpoint files and sort by epoch number
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'model_epoch_*.pth'))
    checkpoint_files.sort(key=lambda x: int(re.search(r'model_epoch_(\d+).pth', x).group(1)))
    
    # Create a grid for visualization
    x = torch.linspace(0, Lx, 256)
    t = torch.linspace(0, T, 100)
    X, T_grid = torch.meshgrid(x, t, indexing='ij')
    xt = torch.stack([X.flatten(), T_grid.flatten()], dim=1).to(device)
    
    # Find global min/max for consistent color scaling
    print("Finding global min/max values...")
    min_val, max_val = float('inf'), -float('inf')
    for checkpoint_file in checkpoint_files:
        model, epoch, _ = load_checkpoint(checkpoint_file, device)
        with torch.no_grad():
            U = model(xt).reshape(X.shape).cpu().numpy()
        min_val = min(min_val, U.min())
        max_val = max(max_val, U.max())
    
    print(f"Global value range: {min_val:.2f} to {max_val:.2f}")
    
    # Create plots for each checkpoint
    for checkpoint_file in checkpoint_files:
        model, epoch, losses = load_checkpoint(checkpoint_file, device)
        
        with torch.no_grad():
            U = model(xt).reshape(X.shape).cpu().numpy()
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot temperature pattern
        im = ax1.pcolormesh(x.numpy(), t.numpy(), U.T, 
                           cmap='hot', 
                           shading='auto',
                           norm=Normalize(vmin=min_val, vmax=max_val))
        fig.colorbar(im, ax=ax1, label='Temperature (u)')
        ax1.set_xlabel('Position (x)')
        ax1.set_ylabel('Time (t)')
        ax1.set_title(f'Temperature Pattern Formation (Epoch {epoch})')
        
        # Plot loss history
        ax2.semilogy(losses['total'], label='Total Loss')
        ax2.semilogy(losses['pde'], label='PDE Loss')
        ax2.semilogy(losses['ic'], label='IC Loss')
        ax2.semilogy(losses['bc'], label='BC Loss')
        ax2.axvline(200, color='gray', linestyle='--', label='Optimizer Switch')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss Value')
        ax2.set_title('Training Loss History')
        ax2.legend()
        ax2.grid(True, which="both", ls="-")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'pattern_epoch_{epoch:04d}.png'))
        plt.close()
        print(f"Saved visualization for epoch {epoch}")
    
    print("All visualizations saved to", output_dir)

def create_animation(output_dir='pattern_evolution', fps=10):
    """Create an animation from the saved images"""
    import imageio.v2 as imageio
    
    # Get all image files and sort by epoch number
    image_files = glob.glob(os.path.join(output_dir, 'pattern_epoch_*.png'))
    image_files.sort(key=lambda x: int(re.search(r'pattern_epoch_(\d+).png', x).group(1)))
    
    # Read images and create animation
    images = []
    for image_file in image_files:
        images.append(imageio.imread(image_file))
    
    # Save as GIF
    output_gif = os.path.join(output_dir, 'pattern_evolution.gif')
    imageio.mimsave(output_gif, images, fps=fps)
    print(f"Animation saved to {output_gif}")

if __name__ == "__main__":
    # Visualize pattern evolution at different epochs
    visualize_pattern_evolution()
    
    # Create an animation (optional)
    create_animation(fps=5)