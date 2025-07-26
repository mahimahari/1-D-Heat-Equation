import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Parameters
Lx = 1.0               # Domain length (0 to 1)
T = 1.0                # Total simulation time
alpha = 0.01           # Thermal diffusivity
N_train = 10000        # Number of training points
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Heat_PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 64),  # Input: [x, t]
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def forward(self, xt):
        return self.fc(xt)
        
    def pde_loss(self, xt, alpha):
        xt = xt.clone().requires_grad_(True)
        u = self(xt)
        
        # Compute gradients
        def gradient(output, input):
            return torch.autograd.grad(
                output, input, 
                grad_outputs=torch.ones_like(output),
                create_graph=True,
                retain_graph=True
            )[0]
        
        # First derivatives
        du = gradient(u, xt)
        u_t = du[:, 1:2]
        u_x = du[:, 0:1]
        
        # Second derivative
        u_xx = gradient(u_x, xt)[:, 0:1]
        
        # Heat equation residual
        residual = u_t - alpha * u_xx
        return torch.mean(residual**2)

def generate_training_data():
    """Generate training data for heat equation"""
    # Interior points
    x = torch.rand(N_train, 1) * Lx
    t = torch.rand(N_train, 1) * T
    X_train = torch.cat([x, t], dim=1)
    
    # Initial condition points: u(x,0) = sin(πx)
    x_ic = torch.linspace(0, Lx, N_train).reshape(-1, 1)
    t_ic = torch.zeros_like(x_ic)
    u_ic = torch.sin(np.pi * x_ic / Lx)  # Initial condition
    X_ic = torch.cat([x_ic, t_ic], dim=1)
    
    # Boundary condition points: u(0,t) = u(Lx,t) = 0 (Dirichlet BC)
    t_bc = torch.rand(N_train // 4, 1) * T
    x0 = torch.zeros_like(t_bc)
    xL = x0 + Lx
    X_bc_0 = torch.cat([x0, t_bc], dim=1)
    X_bc_L = torch.cat([xL, t_bc], dim=1)
    u_bc = torch.zeros_like(t_bc)  # Boundary condition value
    
    return (X_train.to(device), X_ic.to(device), 
            u_ic.to(device), X_bc_0.to(device), X_bc_L.to(device), 
            u_bc.to(device))

def save_checkpoint(model, epoch, losses, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'losses': losses,
    }, checkpoint_path)

def train_model(model, X_train, X_ic, u_ic, X_bc_0, X_bc_L, u_bc, epochs=500):
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)
    losses = {'total': [], 'pde': [], 'ic': [], 'bc': []}
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # PDE loss
        pde_loss = model.pde_loss(X_train, alpha)
        
        # IC loss
        u_pred_ic = model(X_ic)
        ic_loss = torch.mean((u_pred_ic - u_ic)**2)
        
        # BC loss (Dirichlet boundary conditions)
        u_pred_bc0 = model(X_bc_0)
        u_pred_bcL = model(X_bc_L)
        bc_loss = torch.mean((u_pred_bc0 - u_bc)**2) + torch.mean((u_pred_bcL - u_bc)**2)
        
        # Weighted loss
        loss = 1*pde_loss + 100*ic_loss + 100*bc_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        # Store losses
        losses['total'].append(loss.item())
        losses['pde'].append(pde_loss.item())
        losses['ic'].append(ic_loss.item())
        losses['bc'].append(bc_loss.item())
        
        # Save checkpoint every epoch
        save_checkpoint(model, epoch, losses)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}:")
            print(f"  Total: {loss.item():.2e}  PDE: {pde_loss.item():.2e}")
            print(f"  IC: {ic_loss.item():.2e}  BC: {bc_loss.item():.2e}")
            
            if epoch % 100 == 0:
                with torch.no_grad():
                    plt.figure(figsize=(10, 4))
                    plt.plot(X_ic.cpu()[:, 0], u_ic.cpu(), 'k-', label='Exact IC')
                    plt.plot(X_ic.cpu()[:, 0], model(X_ic).cpu(), 'r--', label='Predicted')
                    plt.title(f'IC Fit at Epoch {epoch}')
                    plt.legend()
                    plt.show()
    
    return losses

def visualize_solution(model, Lx, T):
    x = torch.linspace(0, Lx, 100)
    t = torch.linspace(0, T, 50)
    X, T_grid = torch.meshgrid(x, t, indexing='ij')
    xt = torch.stack([X.flatten(), T_grid.flatten()], dim=1).to(device)
    
    with torch.no_grad():
        U = model(xt).reshape(X.shape).cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(x.numpy(), t.numpy(), U.T, cmap='hot', shading='auto')
    plt.colorbar(label='Temperature (u)')
    plt.xlabel("Position (x)")
    plt.ylabel("Time (t)")
    plt.title("1D Heat Equation Solution")
    plt.tight_layout()
    plt.show()

def run_experiment():
    X_train, X_ic, u_ic, X_bc_0, X_bc_L, u_bc = generate_training_data()
    model = Heat_PINN().to(device)
    losses = train_model(model, X_train, X_ic, u_ic, X_bc_0, X_bc_L, u_bc)
    
    # Save loss history
    df = pd.DataFrame(losses)
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/loss_history.csv', index_label='Epoch')
    
    visualize_solution(model, Lx, T)

if __name__ == "__main__":
    run_experiment()