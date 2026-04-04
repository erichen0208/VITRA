from vitra.models.action_model.dit import DiT
from vitra.models.action_model import create_diffusion
from . import gaussian_diffusion as gd
from vitra.datasets.dataset_utils import ActionFeature, get_robot_7d_loss_components
import torch
from torch import nn


def _build_navigation_time_weights(window: int, time_weights=None, tail_boost: float = 0.0):
    if window <= 0:
        return None
    if time_weights is not None:
        tw = torch.tensor(time_weights, dtype=torch.float32)
        if tw.numel() != window:
            raise ValueError(f"navigation_time_weights must have length {window}, got {tw.numel()}")
    else:
        # Uniform baseline with optional linear emphasis on farther waypoints.
        tw = torch.ones(window, dtype=torch.float32)
        if float(tail_boost) > 0.0 and window > 1:
            tw = tw + float(tail_boost) * torch.linspace(0.0, 1.0, window)

    tw = tw / max(float(tw.mean()), 1e-6)
    return tw

def DiT_T(**kwargs):
    return DiT(depth=3, hidden_size=256, num_heads=4, **kwargs)
def DiT_S(**kwargs):
    return DiT(depth=6, hidden_size=384, num_heads=4, **kwargs)
def DiT_M(**kwargs):
    return DiT(depth=12, hidden_size=384, num_heads=6, **kwargs)
def DiT_B(**kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)
def DiT_L(**kwargs):
    return DiT(depth=24, hidden_size=1024, num_heads=16, **kwargs)

DiT_models = {'DiT-S': DiT_S, 'DiT-M': DiT_M, 'DiT-B': DiT_B, 'DiT-T': DiT_T, 'DiT-L': DiT_L}

class DiffusionPolicy(nn.Module):
    def __init__(
        self, 
        token_size, 
        model_type='DiT-B', 
        in_channels=192, 
        future_action_window_size=16, 
        past_action_window_size=0, 
        use_state=None, 
        action_type='angle',
        diffusion_steps=100,
        state_dim=None,
        loss_type='human',
        use_wrist_cross_attn=False,
        navigation_time_weights=None,
        navigation_tail_boost=0.0,
        navigation_straight_alpha=0.0,
        navigation_straight_decay_deg=15.0,
    ):
        super().__init__()
        # SimpleMLP takes in x_t, timestep, and condition, and outputs predicted noise.
        self.in_channels = in_channels
        # GaussianDiffusion offers forward and backward functions q_sample and p_sample.
        self.diffusion_steps = diffusion_steps
        self.diffusion = create_diffusion(timestep_respacing="", 
                                        noise_schedule = 'squaredcos_cap_v2', 
                                        diffusion_steps=self.diffusion_steps, 
                                        sigma_small=True, 
                                        learn_sigma = False
                                        ) 
        #self.diffusion = create_diffusion(timestep_respacing="", noise_schedule = 'linear', diffusion_steps=100, sigma_small=True, learn_sigma = False)
        self.ddim_diffusion = None
        if self.diffusion.model_var_type in [gd.ModelVarType.LEARNED, gd.ModelVarType.LEARNED_RANGE]:
            learn_sigma = True
        else:
            learn_sigma = False
        self.past_action_window_size = past_action_window_size
        self.future_action_window_size = future_action_window_size
        self.use_state = use_state
        self.action_type = action_type
        self.navigation_time_weights = _build_navigation_time_weights(
            self.future_action_window_size,
            time_weights=navigation_time_weights,
            tail_boost=navigation_tail_boost,
        )
        self.navigation_straight_alpha = float(navigation_straight_alpha)
        self.navigation_straight_decay_deg = float(max(navigation_straight_decay_deg, 1e-6))
        
        # Get loss components and hand group mapping from ActionFeature
        if loss_type == 'human':
            self.loss_components = ActionFeature.get_loss_components(action_type)
        elif loss_type == 'robot':
            self.loss_components = ActionFeature.get_robot_loss_components()
        elif loss_type == 'xhand':
            self.loss_components = ActionFeature.get_xhand_loss_components()
        elif loss_type == 'robot_7d':
            self.loss_components = get_robot_7d_loss_components()
        elif loss_type == 'navigation':
            # Navigation control.
            # Preferred target layout: [x, y, sin(theta), cos(theta)] so that
            # heading is continuous across +/-pi boundary.
            if in_channels >= 4:
                self.loss_components = {
                    "delta_xy": (0, 2, 2.0),
                    "heading_sincos": (2, 4, 1.5),
                }
                if in_channels > 4:
                    self.loss_components["aux"] = (4, in_channels, 1.0)
            elif in_channels >= 3:
                # Backward compatibility for [x, y, theta]-style targets.
                self.loss_components = {
                    "delta_xy": (0, 2, 2.0),
                    "yaw": (2, 3, 1.5),
                }
            else:
                self.loss_components = {
                    "action": (0, in_channels, 1.0),
                }
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        self.net = DiT_models[model_type](
            token_size = token_size, 
            action_dim = in_channels, 
            class_dropout_prob = 0.1, 
            learn_sigma = learn_sigma, 
            future_action_window_size = future_action_window_size, 
            past_action_window_size = past_action_window_size,
            use_state = use_state,
            state_dim=state_dim,
            use_wrist_cross_attn=use_wrist_cross_attn,
        )

    # Given condition z and ground truth token x, x_mask, compute loss
    def loss(self, x, z, x_mask, state=None, state_mask=None, wrist_features=None):
        # sample random noise and timestep
        noise = torch.randn_like(x) # [B, T, C]
        timestep = torch.randint(0, self.diffusion.num_timesteps, (x.size(0),), device= x.device)
        
        # sample x_t from x
        x_t = self.diffusion.q_sample(x, timestep, noise)
        x_t = x_t * x_mask
        x_t = torch.cat([x_t, x_mask], dim=2) # [B, T, D]

        # predict noise from x_t
        noise_pred = self.net(x_t, timestep, z, state, state_mask, wrist_features=wrist_features)

        assert noise_pred.shape == noise.shape == x.shape

        # L2 loss with mask
        square_delta = (noise_pred - noise) ** 2 * x_mask

        time_weights = None
        if self.loss_components is not None and self.navigation_time_weights is not None and "delta_xy" in self.loss_components:
            tw = self.navigation_time_weights.to(x.device)
            if tw.numel() == x.shape[1] - 1:
                # VITRA keeps future_action_window_size=chunk_size-1 while action
                # supervision is chunk_size; extend the last weight for the extra step.
                tw = torch.cat([tw, tw[-1:].clone()], dim=0)
            elif tw.numel() != x.shape[1]:
                raise ValueError(
                    f"navigation_time_weights length {tw.numel()} incompatible with action horizon {x.shape[1]}"
                )
            time_weights = tw.view(1, -1, 1)

        sample_weights = None
        if (
            self.loss_components is not None
            and "delta_xy" in self.loss_components
            and self.navigation_straight_alpha > 0.0
            and x.shape[1] >= 2
        ):
            # Build per-sample weights from GT heading changes so straighter clips
            # (smaller cumulative rotation) contribute more to the training loss.
            if x.shape[2] >= 4:
                yaw = torch.atan2(x[:, :, 2], x[:, :, 3])
            elif x.shape[2] >= 3:
                yaw = x[:, :, 2]
            else:
                yaw = None

            if yaw is not None:
                dyaw = yaw[:, 1:] - yaw[:, :-1]
                dyaw = torch.atan2(torch.sin(dyaw), torch.cos(dyaw))
                valid_t = x_mask.any(dim=-1).float()
                valid_pair = valid_t[:, 1:] * valid_t[:, :-1]

                rot_abs_deg = dyaw.abs() * (180.0 / torch.pi)
                denom = valid_pair.sum(dim=1).clamp_min(1.0)
                mean_rot_deg = (rot_abs_deg * valid_pair).sum(dim=1) / denom

                straight_factor = torch.exp(-mean_rot_deg / self.navigation_straight_decay_deg)
                sample_weights = 1.0 + self.navigation_straight_alpha * straight_factor
                sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-6)
                sample_weights = sample_weights.view(-1, 1, 1)
        
        # Generic mask loss computation function
        def mask_loss(from_dim, to_dim):
            if time_weights is None:
                w = 1.0
            else:
                w = time_weights
            if sample_weights is not None:
                w = w * sample_weights
            s = (square_delta[:, :, from_dim:to_dim] * w).sum()
            n = (x_mask[:, :, from_dim:to_dim] * w).sum()
            return s / n if n > 0 else 0
        
        # Compute loss for each component using ActionFeature definitions
        component_losses = {}
        component_counts = {}
        
        for name, (start, end, weight) in self.loss_components.items():
            component_losses[name] = mask_loss(start, end) * weight
            # Count samples where ANY dim in the component range is active.
            # Using x_mask[:, :, start].sum() only checks the first dim, which
            # breaks when sparse retarget masks leave the first dim inactive
            # (e.g. right_hand_joints starts at dim 57 but gripper retarget
            # activates dims 59, 68, 77, 86).
            component_counts[name] = x_mask[:, :, start:end].any(dim=-1).float().sum()
        
        total_count = sum(component_counts.values())

        if total_count == 0:
            loss = square_delta[0, 0, 0]
        else:
            loss = sum(
                component_losses[k] * component_counts[k]
                for k in component_counts.keys()
            ) / total_count

        # Return loss with detailed component losses for logging
        return {
            "loss": loss,
            **component_losses,  # Unpack all component losses
        }
    
    # Given condition and noise, sample x using reverse diffusion process
    def sample(self, 
            action_features,
            cfg_scale,
            current_state,
            current_state_mask,
            use_ddim,
            num_ddim_steps,
            action_masks,
            wrist_features=None,
        ):
        B = action_features.shape[0]
        noise = torch.randn(action_features.shape[0], self.future_action_window_size+1, 
                self.in_channels,  device=action_features.device)   #[B, T, D]

        x_mask = action_masks.to(action_features.device)

        using_cfg = cfg_scale > 1.0
        if using_cfg:
            noise = torch.cat([noise, noise], 0)
            uncondition = self.net.z_embedder.uncondition
            uncondition = uncondition.unsqueeze(0)  #[1, D]
            uncondition = uncondition.expand(B, 1, -1) #[B, 1, D]
            z = torch.cat([action_features, uncondition], 0)
            cfg_scale = cfg_scale

            if self.use_state == 'DiT':
                model_kwargs = dict(
                    z=z, x_mask=x_mask, 
                    cfg_scale=cfg_scale, state=current_state, 
                    state_mask=current_state_mask,
                    wrist_features=wrist_features,
                )
            else:
                model_kwargs = dict(z=z, x_mask=x_mask, cfg_scale=cfg_scale, wrist_features=wrist_features)
            sample_fn = self.net.forward_with_cfg
        else:
            z = action_features

            # Without CFG we still need to apply x_mask and concatenate it
            # before calling DiT.forward(), because forward() expects
            # x ∈ [B, T, 2D] (action + mask channels), same as in loss().
            def _fwd_with_mask(x, t, **kwargs):
                xm = kwargs.pop('x_mask')
                x_in = torch.cat([x * xm, xm], dim=2)
                return self.net.forward(x_in, t, **kwargs)

            if self.use_state == 'DiT':
                model_kwargs = dict(z=z, x_mask=x_mask, state=current_state, state_mask=current_state_mask, wrist_features=wrist_features)
            else:
                model_kwargs = dict(z=z, x_mask=x_mask, wrist_features=wrist_features)
            sample_fn = _fwd_with_mask

        if use_ddim and num_ddim_steps is not None:
            if self.ddim_diffusion is None:
                self.create_ddim(ddim_step=num_ddim_steps)
            samples = self.ddim_diffusion.ddim_sample_loop(
                sample_fn, 
                noise.shape, 
                noise, 
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=action_features.device,
                eta=0.0
            )
        else:
            samples = self.ddim_diffusion.diffusion.p_sample_loop(
                sample_fn, 
                noise.shape, 
                noise, 
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=False,
                device=action_features.device
            )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        return samples

    # Create DDIM sampler
    def create_ddim(self, ddim_step=10):
        self.ddim_diffusion = create_diffusion(
            timestep_respacing="ddim"+str(ddim_step), 
            noise_schedule = 'squaredcos_cap_v2', 
            diffusion_steps=self.diffusion_steps, 
            sigma_small=True, 
            learn_sigma = False
        )
        return self.ddim_diffusion