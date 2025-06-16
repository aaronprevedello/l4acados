import torch
import numpy as np
import casadi
from casadi import Callback

class GPDiscreteCallback(Callback):
    def __init__(self, gp_model, nx, nu, name="gp_dynamics"):
        self.gp_model = gp_model
        self.nx = nx
        self.nu = nu
        self.input_dim = nx + nu
        self.output_dim = nx  # assuming full next state prediction

        # Always call super AFTER setting attributes
        Callback.__init__(self)
        self.construct(name)
    
    def get_n_in(self): 
        return 2  # Inputs: [x, u]

    def get_n_out(self): 
        return 1  # Outputs: [x_next]

    def get_name_in(self, idx):
        return ['x', 'u'][idx]

    def get_name_out(self, idx):
        return ['x_next'][idx]

    def get_sparsity_in(self, idx):
        if idx == 0:
            return casadi.Sparsity.dense(self.nx)
        elif idx == 1:
            return casadi.Sparsity.dense(self.nu)

    def get_sparsity_out(self, idx):
        return casadi.Sparsity.dense(self.nx)

    def has_forward(self, nfwd):
        return True  # <-- We provide derivatives!

    def eval(self, arg):
        """Evaluate the GP model: x_next = f(x, u)"""
        x_np = np.array(arg[0]).flatten()
        u_np = np.array(arg[1]).flatten()
        xu = np.concatenate([x_np, u_np])[None, :]  # shape (1, nx+nu)

        xu_tensor = torch.tensor(xu, dtype=torch.float32)

        with torch.no_grad():
            y = self.gp_model(xu_tensor).mean  # GP predictive mean
        return [y.numpy().flatten()]

    def forward(self, arg, fwd):
        """Compute Jacobian-vector product"""
        # arg = [x, u]
        # fwd = [dx, du]

        x_np = np.array(arg[0]).flatten()
        u_np = np.array(arg[1]).flatten()
        dx_np = np.array(fwd[0]).flatten()
        du_np = np.array(fwd[1]).flatten()

        xu = np.concatenate([x_np, u_np])[None, :]     # (1, nx+nu)
        dxu = np.concatenate([dx_np, du_np])[None, :]   # (1, nx+nu)

        # Make input a differentiable tensor
        xu_tensor = torch.tensor(xu, dtype=torch.float32, requires_grad=True)
        
        output = self.gp_model(xu_tensor).mean  # (1, nx)

        grads = []
        for i in range(self.output_dim):
            grad_outputs = torch.zeros_like(output)
            grad_outputs[0, i] = 1.0
            grad = torch.autograd.grad(
                outputs=output,
                inputs=xu_tensor,
                grad_outputs=grad_outputs,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )[0]  # (1, nx+nu)
            grads.append(grad)

        # Stack all gradients -> (output_dim, input_dim)
        J = torch.cat(grads, dim=0).detach().numpy()  # (nx, nx+nu)

        # Now compute Jacobian-vector product
        vjp = np.dot(J, dxu.T)  # (nx,)

        return [vjp.flatten()]
