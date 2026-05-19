import torch
from src_pr.grad_utils import *
from src_pr.unet_new import generalized_image_to_b_xy_c, generalized_b_xy_c_to_image
import einops as ein
        
class ResidualsCharge:
    def __init__(self, model, fd_acc, pixels_per_dim, pixels_at_boundary, device = 'cpu', bcs = 'dirichlet0', domain_length = 1., residual_grad_guidance = False, use_ddim_x0 = False, ddim_steps = 0):
        """
        Initialize the residual evaluation for Poisson equation: (-Δ)U = ρ

        :param model: The neural network model to compute the residuals for.
        :param fd_acc: Finite difference accuracy.
        :param pixels_per_dim: Number of pixels per dimension.
        :param pixels_at_boundary: Whether pixels are at boundary.
        :param device: Device to run on.
        :param bcs: Boundary conditions ('dirichlet0').
        :param domain_length: Domain length.
        :param residual_grad_guidance: Whether to use residual gradient guidance.
        :param use_ddim_x0: Whether to use DDIM for x0 estimation.
        :param ddim_steps: Number of DDIM steps.
        """
        self.gov_eqs = 'charge'
        self.model = model
        self.pixels_at_boundary = pixels_at_boundary
        self.periodic = False
        self.input_dim = 2

        if self.pixels_at_boundary:
            d0 = domain_length / (pixels_per_dim - 1)
            d1 = domain_length / (pixels_per_dim - 1)
        else:
            d0 = domain_length / pixels_per_dim
            d1 = domain_length / pixels_per_dim
        
        self.grads = GradientsHelper(d0=d0, d1=d1, fd_acc = fd_acc, periodic=self.periodic, device=device)
        self.relu = torch.nn.ReLU()

        self.pixels_per_dim = pixels_per_dim
        self.device = device
        self.bcs = bcs
        self.domain_length = domain_length

        self.residual_grad_guidance = residual_grad_guidance
        self.use_ddim_x0 = use_ddim_x0
        self.ddim_steps = ddim_steps

    def compute_residual(self, input, reduce = 'none', return_model_out = False, return_optimizer = False, return_inequality = False, sample = False, ddim_func = None, pass_through = False, return_projections = False, skip_model_call = False, given_model_output = None):

        if pass_through:
            assert isinstance(input, torch.Tensor), 'Input is assumed to directly be given output.'
            x0_pred = input
            model_out = x0_pred
        elif skip_model_call and given_model_output is not None:
            # 直接使用给定的模型输出，跳过模型调用
            x0_pred = given_model_output
            model_out = x0_pred
        else:
            assert len(input[0]) == 2 and isinstance(input[0], tuple), 'Input[0] must be a tuple consisting of noisy signal and time.'
            noisy_in, time = iter(input[0])

            if self.residual_grad_guidance:
                assert not self.use_ddim_x0, 'Residual gradient guidance is not implemented with sample estimation for residual.'
                noisy_in.requires_grad = True
                residual_noisy_in = self.compute_residual(generalized_b_xy_c_to_image(noisy_in), pass_through = True)['residual']
                dr_dx = torch.autograd.grad(residual_noisy_in.abs().mean(), noisy_in)[0]
                if sample:
                    x0_pred = self.model.forward_with_guidance_scale(noisy_in, time, cond = dr_dx, guidance_scale = 3.)
                    model_out = x0_pred
                else:
                    # residual guidance 情况下不返回投影，避免干扰guidance计算
                    x0_pred = self.model(noisy_in, time, cond = dr_dx, null_cond_prob = 0.1)
                    model_out = x0_pred
            else:
                if self.use_ddim_x0:
                    x0_pred, model_out = ddim_func(noisy_in, time, self.model, noisy_in.shape, self.ddim_steps, 0.)
                else:
                    # 按需请求投影头输出（仅当模型支持且开启时）
                    call_kwargs = {}
                    if return_projections and hasattr(self.model, 'use_projection_heads') and getattr(self.model, 'use_projection_heads'):
                        call_kwargs['return_projections'] = True
                    x0_pred = self.model(noisy_in, time, **call_kwargs)
                    model_out = x0_pred

        # 统一处理模型输出：可能是 (tensor, projections)
        projections = None
        if isinstance(x0_pred, tuple):
            x0_pred, projections = x0_pred
        if isinstance(model_out, tuple):
            model_out = model_out[0]

        # 支持直接传入 b_xy_c（例如来自投影头）的情况
        if x0_pred.ndim == 3:
            x0_pred = generalized_b_xy_c_to_image(x0_pred)

        assert len(x0_pred.shape) == 4, 'Model output must be a tensor shaped as an image (with explicit axes for the spatial dimensions).'
        batch_size, output_dim, pixels_per_dim, pixels_per_dim = x0_pred.shape
        
        # x0_pred contains [U, rho] where U is potential and rho is charge density
        U = x0_pred[:, 0]  # potential
        rho = x0_pred[:, 1]  # charge density
        
        # Compute Laplacian of U: ΔU = d²U/dx² + d²U/dy²
        U_d00 = self.grads.stencil_gradients(U, mode='d_d00')
        U_d11 = self.grads.stencil_gradients(U, mode='d_d11')
        laplacian_U = U_d00 + U_d11
        
        # Poisson equation: (-Δ)U = ρ, so residual is: -ΔU - ρ = 0
        # Therefore: residual = -laplacian_U - rho
        residual_interior = -laplacian_U - rho
        
        # Apply Dirichlet boundary conditions (U = 0 at boundary)
        residual_bc = None
        if self.bcs == 'dirichlet0':
            if self.pixels_at_boundary:
                # Boundary pixels: residual is U itself (since U should be 0 at boundary)
                residual_bc = torch.zeros_like(U)
                residual_bc[:, 0, :] = U[:, 0, :]  # top boundary
                residual_bc[:, -1, :] = U[:, -1, :]  # bottom boundary
                residual_bc[:, :, 0] = U[:, :, 0]  # left boundary
                residual_bc[:, :, -1] = U[:, :, -1]  # right boundary
                
                # Combine interior and boundary residuals
                residual = torch.stack([residual_interior, residual_bc], dim=1)  # [batch, 2, H, W]
            else:
                # If no boundary pixels, only interior residual
                residual = residual_interior.unsqueeze(1)  # [batch, 1, H, W]
        else:
            # Only interior residual for other boundary conditions
            residual = residual_interior.unsqueeze(1)  # [batch, 1, H, W]
        
        residual = generalized_image_to_b_xy_c(residual)

        output = {}
        output['residual'] = residual

        if return_inequality:
            pass # not considered here
        if return_optimizer:
            pass # not considered here

        if return_model_out:
            output['model_out'] = model_out
        if return_projections and (projections is not None):
            output['projections'] = projections

        if reduce == 'full':
            # mean over all items in dict
            return {k: v.mean() for k, v in output.items()}
        elif reduce == 'per-batch':
            # mean over all but first dimension (batch dimension) for tensor values only
            reduced = {}
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    if v.ndim > 1 and (k != 'model_out' and k != 'residual'):
                        reduced[k] = v.mean(dim=tuple(range(1, v.ndim)))
                    else:
                        reduced[k] = v
                else:
                    # keep non-tensor values (e.g., dict of projections) as-is
                    reduced[k] = v
            return reduced
        elif reduce == 'none':
            # return as-is
            return output
        else:
            raise ValueError('Unknown reduction method.')
