"""
Poisson equation residual computation module v2

Adapted for normalized data (rho stores charge q, not q/h²)

Governing equation: (-Δ)U = ρ_physical = ρ_normalized / h²
Boundary condition: U|∂Ω = 0 (homogeneous Dirichlet)

Residual: r = ΔU + ρ_normalized / h²
"""

import torch
from src.grad_utils import GradientsHelper
from src.unet_model import generalized_image_to_b_xy_c, generalized_b_xy_c_to_image


class ResidualsPoissonV2:
    def __init__(
        self, 
        model, 
        pixels_per_dim=64, 
        pixels_at_boundary=True,
        device='cpu', 
        domain_length=1.0,
        fd_acc=2,
        use_ddim_x0=False, 
        ddim_steps=0
    ):
        """
        Initialize Poisson residual computer (v2 - normalized version)
        
        Key differences:
        - Input rho is normalized (charge q)
        - Need to convert rho to physical quantity (rho / h²) when computing residual
        """
        self.gov_eqs = 'poisson'
        self.model = model
        self.pixels_per_dim = pixels_per_dim
        self.pixels_at_boundary = pixels_at_boundary
        self.device = device
        
        # Compute grid spacing
        if pixels_at_boundary:
            h = domain_length / (pixels_per_dim - 1)  # h = 1/63
        else:
            h = domain_length / pixels_per_dim
        
        self.h = h
        self.h_squared = h * h
        
        # Finite difference gradient computer
        self.grads = GradientsHelper(
            d0=h, d1=h, 
            fd_acc=fd_acc, 
            periodic=False, 
            device=device
        )
        
        self.use_ddim_x0 = use_ddim_x0
        self.ddim_steps = ddim_steps
        
    def compute_residual(
        self, 
        input, 
        reduce='none', 
        return_model_out=False, 
        return_optimizer=False, 
        return_inequality=False,
        sample=False,
        ddim_func=None,
        pass_through=False,
        rho_condition=None,  # For pass_through mode
        return_projections=False,  # Whether to return projection results
        skip_model_call=False,  # Skip model call (for projection head residual computation)
        given_model_output=None  # Directly provide model output (for projection head residual computation)
    ):
        """
        Compute Poisson equation residual
        
        Key improvements:
        - rho_normalized needs to be divided by h² to convert to physical quantity
        - Residual: r = ΔU + rho_normalized / h²
        
        New parameters:
        - return_projections: If model supports projection heads, return projection results
        - skip_model_call: Skip model call, directly use given_model_output
        - given_model_output: Directly provided model output (for projection head residual computation)
        """
        projections = {}
        
        if pass_through:
            # Directly pass predicted U
            assert isinstance(input, torch.Tensor), 'Input should be tensor in pass_through mode'
            U_pred = input
            model_out = U_pred
            rho = rho_condition  # Use passed condition
        elif skip_model_call and given_model_output is not None:
            # Skip model call, directly use given output (for projection head residual computation)
            U_pred = given_model_output
            model_out = given_model_output
            # Parse rho
            if len(input) >= 2:
                rho = input[1]
            else:
                rho = rho_condition
        else:
            # Parse input
            assert len(input) >= 2, 'Input must contain (model_input, rho)'
            model_input_tuple = input[0]
            rho = input[1]  # Condition: normalized charge density field [B, 1, H, W]
            
            noisy_in, time = model_input_tuple
            
            # Model forward pass
            if self.use_ddim_x0:
                U_pred, model_out = ddim_func(
                    noisy_in, time, self.model, noisy_in.shape, 
                    self.ddim_steps, 0., gov_eqs=self.gov_eqs
                )
            else:
                # Check if need to return projections
                if return_projections and hasattr(self.model, 'use_projection_heads') and self.model.use_projection_heads:
                    result = self.model(noisy_in, time, return_projections=True)
                    U_pred = result[0]
                    projections = result[1]
                    model_out = U_pred
                else:
                    U_pred = self.model(noisy_in, time)
                    model_out = U_pred
        
        # Ensure correct shape
        if len(U_pred.shape) == 3:
            U_pred = generalized_b_xy_c_to_image(U_pred)
        
        assert len(U_pred.shape) == 4, f'U_pred must be image tensor, got shape {U_pred.shape}'
        batch_size = U_pred.shape[0]
        
        # Extract U
        if U_pred.shape[1] > 1:
            U = U_pred[:, 0:1]
        else:
            U = U_pred
        
        # Compute Laplacian: ΔU = ∂²U/∂x² + ∂²U/∂y²
        U_squeezed = U.squeeze(1)  # [B, H, W]
        U_d00 = self.grads.stencil_gradients(U_squeezed, mode='d_d00')
        U_d11 = self.grads.stencil_gradients(U_squeezed, mode='d_d11')
        lap_U = U_d00 + U_d11  # [B, H, W]
        
        # PDE residual: r = ΔU + ρ_physical (interior nodes)
        # Key: rho_physical = rho_normalized / h²
        if rho is not None:
            rho_squeezed = rho.squeeze(1)  # [B, H, W]
            # Convert normalized rho to physical quantity
            rho_physical = rho_squeezed / self.h_squared
            residual_pde = lap_U[:, 1:-1, 1:-1] + rho_physical[:, 1:-1, 1:-1]
        else:
            residual_pde = lap_U[:, 1:-1, 1:-1]
        
        # Boundary residual: U|∂Ω = 0 (Dirichlet BC)
        residual_bc_top = U[:, :, 0, :]
        residual_bc_bottom = U[:, :, -1, :]
        residual_bc_left = U[:, :, 1:-1, 0]
        residual_bc_right = U[:, :, 1:-1, -1]
        
        residual_bc = torch.cat([
            residual_bc_top.flatten(1),
            residual_bc_bottom.flatten(1),
            residual_bc_left.flatten(1),
            residual_bc_right.flatten(1)
        ], dim=1)
        
        # Combine all residuals
        residual_pde_flat = residual_pde.reshape(batch_size, -1)
        residual_all = torch.cat([residual_pde_flat, residual_bc], dim=1)
        
        # Build output dictionary
        output = {
            'residual': residual_all,
            'residual_pde': residual_pde,
            'residual_bc': residual_bc,
        }
        
        if return_model_out:
            output['model_out'] = model_out
        
        if return_projections and projections:
            output['projections'] = projections
            
        if return_optimizer:
            output['optimizer'] = torch.zeros(batch_size, device=U_pred.device)
            
        if return_inequality:
            output['inequality'] = torch.zeros(batch_size, device=U_pred.device)
        
        # Reduction
        if reduce == 'full':
            result = {}
            for k, v in output.items():
                if k == 'projections':
                    result[k] = v  # Don't reduce projection dictionary
                elif isinstance(v, torch.Tensor):
                    result[k] = v.mean()
                else:
                    result[k] = v
            return result
        elif reduce == 'per-batch':
            result = {}
            for k, v in output.items():
                if k == 'projections':
                    result[k] = v  # Don't reduce projection dictionary
                elif isinstance(v, torch.Tensor) and v.ndim > 1 and k not in ['model_out', 'residual']:
                    result[k] = v.mean(dim=tuple(range(1, v.ndim)))
                else:
                    result[k] = v
            return result
        else:
            return output
    
    def normalize_residual(self, residual):
        """
        Normalize residual for logging
        
        Since PDE residual is computed in physical quantities, directly return original residual
        """
        return residual
    
    def residual_correction(self, x0_pred_in, rho_condition=None):
        """Residual correction"""
        if len(x0_pred_in.shape) == 3:
            x0_pred = generalized_b_xy_c_to_image(x0_pred_in.detach().clone())
        else:
            x0_pred = x0_pred_in.detach().clone()
        
        x0_pred.requires_grad_(True)
        
        residual_out = self.compute_residual(
            x0_pred, pass_through=True, rho_condition=rho_condition
        )
        residual = residual_out['residual']
        
        loss = (residual ** 2).sum()
        grad = torch.autograd.grad(loss, x0_pred)[0]
        
        grad_norm = grad.abs().max().clamp(min=1e-8)
        step_size = 1e-4 / grad_norm
        
        x0_corrected = x0_pred_in.detach() - step_size * generalized_image_to_b_xy_c(grad)
        
        residual_corrected = self.compute_residual(
            generalized_b_xy_c_to_image(x0_corrected), 
            pass_through=True,
            rho_condition=rho_condition
        )['residual']
        
        return x0_corrected, residual_corrected
