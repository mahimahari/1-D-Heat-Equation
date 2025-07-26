import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import os
import glob
import re
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Heat_PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, xt):
        return self.fc(xt)

def load_checkpoint(checkpoint_path, device):
    model = Heat_PINN().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    losses = checkpoint['losses']
    return model, epoch, losses

def visualize_heat_evolution(checkpoint_dir='checkpoints', output_dir='heat_evolution'):
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'model_epoch_*.pth'))
    checkpoint_files.sort(key=lambda x: int(re.search(r'model_epoch_(\d+).pth', x).group(1)))
    
    Lx = 1.0
    T = 1.0
    
    x = torch.linspace(0, Lx, 256)
    t = torch.linspace(0, T, 100)
    X, T_grid = torch.meshgrid(x, t, indexing='ij')
    xt = torch.stack([X.flatten(), T_grid.flatten()], dim=1).to(device)
    
    # Find global min/max
    min_val, max_val = float('inf'), -float('inf')
    for checkpoint_file in checkpoint_files:
        model, epoch, _ = load_checkpoint(checkpoint_file, device)
        with torch.no_grad():
            U = model(xt).reshape(X.shape).cpu().numpy()
        min_val = min(min_val, U.min())
        max_val = max(max_val, U.max())
    
    # Create plots
    for checkpoint_file in checkpoint_files:
        model, epoch, losses = load_checkpoint(checkpoint_file, device)
        
        with torch.no_grad():
            U = model(xt).reshape(X.shape).cpu().numpy()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        im = ax1.pcolormesh(x.numpy(), t.numpy(), U.T, 
                           cmap='hot', 
                           shading='auto',
                           norm=Normalize(vmin=min_val, vmax=max_val))
        fig.colorbar(im, ax=ax1, label='Temperature (u)')
        ax1.set_xlabel('Position (x)')
        ax1.set_ylabel('Time (t)')
        ax1.set_title(f'Heat Equation Solution (Epoch {epoch})')
        
        ax2.semilogy(losses['total'], label='Total Loss')
        ax2.semilogy(losses['pde'], label='PDE Loss')
        ax2.semilogy(losses['ic'], label='IC Loss')
        ax2.semilogy(losses['bc'], label='BC Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss Value')
        ax2.set_title('Training Loss History')
        ax2.legend()
        ax2.grid(True, which="both", ls="-")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'heat_epoch_{epoch:04d}.png'))
        plt.close()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    visualize_heat_evolution()