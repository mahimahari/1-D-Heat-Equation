import torch
import torch.nn as nn

class SH_PINN(nn.Module):
    def __init__(self, architecture):
        super(SH_PINN, self).__init__()
        layers = []
        for i in range(len(architecture) - 1):
            layers.append(nn.Linear(architecture[i], architecture[i+1]))
            if i < len(architecture) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, xt):
        """Input: (x, t), Output: u(x,t)"""
        return self.net(xt)

    def pde_loss(self, xt, r=0.2, b_2=1.2):
        """
        Corrected Swift-Hohenberg PDE residual:
        u_t + (1 + r)*u + 2*u_xx + u_xxxx + b_2*u² - u³ = 0
        
        Note: The original formulation had incorrect operator expansion
        """
        # Ensure gradients are tracked
        xt = xt.clone().requires_grad_(True)
        
        # Forward pass
        u = self.forward(xt)
        
        # Get spatial and temporal coordinates
        x = xt[:, 0:1]
        t = xt[:, 1:2]
        
        # Compute gradients
        def gradient(output, input):
            return torch.autograd.grad(
                output, input, 
                grad_outputs=torch.ones_like(output),
                create_graph=True,
                retain_graph=True
            )[0]
        
        # First derivatives
        u_t = gradient(u, t)
        u_x = gradient(u, x)
        
        # Second derivative
        u_xx = gradient(u_x, x)
        
        # Fourth derivative (via repeated differentiation)
        u_xxx = gradient(u_xx, x)
        u_xxxx = gradient(u_xxx, x)
        
        # Correct Swift-Hohenberg residual
        residual = (
            u_t +                 # Temporal derivative
            (1 + r)*u +           # Linear term
            2*u_xx +              # Second derivative term
            u_xxxx +              # Fourth derivative term
            b_2*u**2 -            # Quadratic nonlinearity
            u**3                  # Cubic nonlinearity
        )
        
        return torch.mean(residual**2)

    def compute_derivatives(self, xt, order=4):
        """
        Helper method to compute derivatives up to specified order
        Returns dictionary of derivatives {u, u_x, u_xx, u_xxx, u_xxxx}
        """
        xt = xt.clone().requires_grad_(True)
        u = self.forward(xt)
        x = xt[:, 0:1]
        
        derivatives = {'u': u}
        
        def gradient(output, input):
            return torch.autograd.grad(
                output, input, 
                grad_outputs=torch.ones_like(output),
                create_graph=True,
                retain_graph=True
            )[0]
        
        current = u
        for i in range(1, order+1):
            current = gradient(current, x)
            derivatives[f'u_{"x"*i}'] = current
            
        return derivatives