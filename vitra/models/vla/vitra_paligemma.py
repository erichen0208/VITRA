import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, List, Callable
import copy
import numpy as np
import json

from PIL import Image
from functools import partial

from vitra.utils.tensor_utils import move_masked_to_left, get_mask_of_last_masked_index, move_masked_to_left_ids
from vitra.models.vlm_builder import build_vlm
from vitra.utils.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

class VITRA_Paligemma(nn.Module):
    def __init__(
        self,
        configs,
        train_setup_configs=None,
        act_model_configs=None,
        fwd_pred_next_n=1,
        repeated_diffusion_steps:int = 8,
        use_state='DiT',
        use_fov=True,
        use_bf16=False,
        **kwargs,
    ):
        super().__init__()

        self.configs = configs
        self.train_setup_configs = train_setup_configs
        self.act_model_configs = act_model_configs
        self.use_state = use_state
        self.use_fov = use_fov
        self.repeated_diffusion_steps = repeated_diffusion_steps
        self.past_action_window_size = 0
        # chunk_size for action prediction
        self.chunk_size = self.configs.get("fwd_pred_next_n", 16)
        self.future_action_window_size = self.chunk_size-1
        self.state_mask_prob = self.configs.get("state_mask_prob", 0.1)
        self.action_type = self.configs['train_dataset'].get("action_type", "angle")
        self.use_state = use_state
        self.use_fov = use_fov
        self.use_bf16 = use_bf16
        if self.action_type == 'angle':
            self.hand_dim = 51
        elif self.action_type == 'keypoints':
            self.hand_dim = 69

        # ── New config: wrist camera ─────────────────────────────────────
        self.action_head_type = "diffusion"  # always DiT
        wrist_cfg = configs.get("wrist_cam", {})
        self.wrist_encoder_type = wrist_cfg.get("encoder", None)  # e.g. "dinov2_vitb14"
        self.use_wrist_cam_train = self.wrist_encoder_type is not None
        self.wrist_mask_prob = wrist_cfg.get("mask_prob", 0.15)

        # Initialize the tokenizer and VLM backbone
        self.tokenizer, self.backbone = self._init_backbone()
        if self.train_setup_configs is not None and self.train_setup_configs.get("reinit", False):
            initialize_param(self.backbone)

        # ── DiT diffusion action head (always used) ───────────────────────
        self.act_model = self._init_act_model()

        if self.use_state == 'VLM':
            self.state_and_mask_dim = 2 * self.configs["state_encoder"]["state_dim"]
            self.vlm_state_encoder = self._init_state_encoder()

        if self.use_fov:
            self.fov_encoder = self._init_fov_encoder()

        # The `cognition_token_id` is set to an unused token ID.
        self.cognition_token_id = self.configs.get("cognition_token_id", 10)
        untied_cognition_token = self.configs.get("untied_cognition_token", True)

        # Use a separately learned `cognition_token_embedding` that does not share parameters with the word embedding matrix
        # if `untied_cognition_token` is True. We initialize this separate `cognition_token_embedding` 
        # with the word embedding parameter corresponding to the specified `cognition_token_id`. 

        if untied_cognition_token:
            if overwatch.rank() == 0:
                overwatch.info(f"Using separate cognition token")
                overwatch.info(f"Cognition token id: {self.cognition_token_id}")
            init_id = self.configs.get("cognition_token_init_id", None)
            if init_id is None:
                init_id = self.cognition_token_id
            if overwatch.rank() == 0:
                overwatch.info(f"Init cognition token with id={init_id}")
            ebd = self.model.get_input_embeddings().weight.data[init_id]
            self.cognition_token = nn.Parameter(ebd.clone())
        else:
            self.cognition_token = None

        # ── Wrist camera → action head conditioning ───────────────────────
        # Wrist image is processed through frozen SigLIP and projected to
        # DiT hidden size, then added to the DiT adaLN conditioning c.
        # This keeps the VLM backbone clean (head cam only).
        if self.use_wrist_cam_train:
            self._init_wrist_action_modules()

    def _init_backbone(self):
        processor, model = build_vlm(self.configs["vlm"])
        self.processor = processor
        self.tokenizer = self.processor.tokenizer
        return self.tokenizer, model

    def _init_fov_encoder(self):
        from vitra.utils.nn_utils import MLPProjector
        fov_dim = 2 # fov_x, fov_y
        mlp = MLPProjector(fov_dim, self.hidden_size)
        nn.init.normal_(mlp.projector[0].weight, mean=0.0, std=0.02)
        nn.init.normal_(mlp.projector[2].weight, mean=0.0, std=0.02)
        nn.init.normal_(mlp.projector[0].bias, mean=0.0, std=0.02)
        nn.init.normal_(mlp.projector[2].bias, mean=0.0, std=0.02)
        return mlp

    def _init_state_encoder(self):
        from vitra.utils.nn_utils import MLPProjector
        mlp = MLPProjector(self.state_and_mask_dim, self.hidden_size)
        nn.init.normal_(mlp.projector[0].weight, mean=0.0, std=0.02)
        nn.init.normal_(mlp.projector[2].weight, mean=0.0, std=0.02)
        nn.init.normal_(mlp.projector[0].bias, mean=0.0, std=0.02)
        nn.init.normal_(mlp.projector[2].bias, mean=0.0, std=0.02)
        return mlp

    def _init_act_model(self):
        from vitra.models.action_model.diffusion_policy import DiffusionPolicy
        # Enable wrist cross-attention when wrist camera is active
        use_wrist_cross_attn = self.use_wrist_cam_train and self.act_model_configs.get(
            "use_wrist_cross_attn", self.act_model_configs.get("use_wrist_token", False))
        action_head = DiffusionPolicy(
            model_type = self.act_model_configs.get("model_type", 'DiT-B'),
            token_size = self.act_model_configs.get("token_size", -1),
            in_channels = self.act_model_configs.get("action_dim", 192),
            future_action_window_size = self.future_action_window_size,
            past_action_window_size = self.past_action_window_size,
            use_state = self.use_state,
            action_type = self.configs['train_dataset'].get("action_type", "angle"),
            state_dim = self.configs["state_encoder"]["state_dim"] if self.use_state=='DiT' else None,
            loss_type = self.configs.get("loss_type", "human"),
            use_wrist_cross_attn = use_wrist_cross_attn,
        )

        for param in action_head.parameters():
            assert param.dtype == torch.float32, f"Loaded diffusion action model parameter not in full precision: {param}"

        return action_head

    def _init_wrist_action_modules(self):
        """Wrist camera RGBD → DiT cross-attention (Local Spatial Pathway).

        Supports two encoder backends (selected by config wrist_cam.encoder):
        - "dinov2_vitb14": DINOv2 ViT-B/14 (768-d, 16×16 patches from 224px).
        - "resnet34":     ResNet-34      (512-d,  7×7  spatial from 224px).

        Both are modified for 4-channel RGBD input.  The RGB channels reuse
        pretrained ImageNet weights; the depth channel is zero-initialised so
        it begins as a no-op and gradually learns during fine-tuning.

        Output: [B, N, D_hidden] spatial tokens fed as K,V into each
        DiTBlock's WristCrossAttention layer.
        """
        import timm

        wrist_cfg = self.configs.get("wrist_cam", {})
        image_size = wrist_cfg.get("image_size", 224)
        freeze_backbone = wrist_cfg.get("freeze_backbone", True)

        # DiT hidden size (read from actually-constructed DiT)
        dit_hidden = self.act_model.net.blocks[0].attn.qkv.in_features  # 768 for DiT-B

        if self.wrist_encoder_type == "resnet34":
            # ── ResNet-34, modified for 4-channel RGBD input ─────────────
            self.wrist_rgbd_encoder = timm.create_model(
                "resnet34",
                pretrained=True,
                num_classes=0,  # remove classification head
            )

            # Modify first conv: 3ch → 4ch  (zero-init depth channel)
            old_conv = self.wrist_rgbd_encoder.conv1
            new_conv = nn.Conv2d(
                4, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=(old_conv.bias is not None),
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                new_conv.weight[:, 3:] = 0.0          # depth starts as no-op
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
            self.wrist_rgbd_encoder.conv1 = new_conv

            # Freeze backbone — only conv1 is trainable (depth channel)
            if freeze_backbone:
                for param in self.wrist_rgbd_encoder.parameters():
                    param.requires_grad = False
                for param in self.wrist_rgbd_encoder.conv1.parameters():
                    param.requires_grad = True

            encoder_dim = self.wrist_rgbd_encoder.num_features  # 512
            # ResNet-34 downsamples 32× → 224/32 = 7
            self.wrist_grid_size = image_size // 32              # 7
            self.wrist_num_patches_raw = self.wrist_grid_size ** 2  # 49

        else:
            # ── DINOv2 ViT-B/14, modified for 4-channel RGBD input ──────
            self.wrist_rgbd_encoder = timm.create_model(
                "vit_base_patch14_dinov2.lvd142m",
                pretrained=True,
                img_size=image_size,
                num_classes=0,  # remove classification head
            )

            # Modify patch embedding: 3ch → 4ch  (zero-init depth channel)
            old_proj = self.wrist_rgbd_encoder.patch_embed.proj
            new_proj = nn.Conv2d(
                4, old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
                bias=(old_proj.bias is not None),
            )
            with torch.no_grad():
                new_proj.weight[:, :3] = old_proj.weight
                new_proj.weight[:, 3:] = 0.0          # depth starts as no-op
                if old_proj.bias is not None:
                    new_proj.bias.copy_(old_proj.bias)
            self.wrist_rgbd_encoder.patch_embed.proj = new_proj

            # Freeze backbone — only patch-embed is trainable (depth channel)
            if freeze_backbone:
                for param in self.wrist_rgbd_encoder.parameters():
                    param.requires_grad = False
                for param in self.wrist_rgbd_encoder.patch_embed.parameters():
                    param.requires_grad = True

            encoder_dim = self.wrist_rgbd_encoder.embed_dim  # 768 for ViT-B
            # ViT-B/14 with 224 → 16×16 = 256 patches
            patch_size = self.wrist_rgbd_encoder.patch_embed.proj.kernel_size[0]
            self.wrist_grid_size = image_size // patch_size          # 16
            self.wrist_num_patches_raw = self.wrist_grid_size ** 2   # 256

        # ── Shared modules ────────────────────────────────────────────────

        # Project encoder dim → DiT hidden dim if they differ
        if encoder_dim != dit_hidden:
            self.wrist_projector = nn.Sequential(
                nn.Linear(encoder_dim, dit_hidden),
                nn.GELU(),
                nn.Linear(dit_hidden, dit_hidden),
            )
        else:
            self.wrist_projector = nn.Identity()

        # Spatial token pooling (only when pool_size < grid_size)
        self.wrist_pool_size = wrist_cfg.get("pool_size", 8)
        if self.wrist_pool_size < self.wrist_grid_size:
            self.wrist_num_patches = self.wrist_pool_size ** 2
        else:
            self.wrist_num_patches = self.wrist_num_patches_raw

        # Learned null tokens for training-time masking (wrist absent)
        self.missing_wrist_tokens = nn.Parameter(torch.zeros(1, 1, dit_hidden))
        nn.init.normal_(self.missing_wrist_tokens, std=0.02)

        # ImageNet normalisation constants for RGB channels
        self.register_buffer(
            "wrist_rgb_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "wrist_rgb_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def trainable_params_setup(self):
        model = self.model
        model.config.use_cache = False

        if self.train_setup_configs.get("freeze_option", "full_finetune") == "full_finetune":
            model.requires_grad_(True)
            self.vision_tower.requires_grad_(True)
            self.word_embedding.requires_grad_(True)

        if self.train_setup_configs.get("freeze_option", "only_head_and_token") == "only_head_and_token":
            model.requires_grad_(False)
            self.vision_tower.requires_grad_(False)
            self.word_embedding.requires_grad_(False)

        if self.train_setup_configs.get("freeze_option", "freeze_vision_encoder") == "freeze_vision_encoder":
            model.requires_grad_(True)
            self.vision_tower.requires_grad_(False)
            self.word_embedding.requires_grad_(True)

        if self.act_model is not None:
            self.act_model.requires_grad_(True)

        if self.use_state == 'VLM':
            self.vlm_state_encoder.requires_grad_(True)
        
        if self.use_fov:
            self.fov_encoder.requires_grad_(True)

        if self.cognition_token is not None:
            self.cognition_token.requires_grad_(True)

    def apply_lora_adapters(self, lora_cfg: dict):
        """Apply LoRA to LLM attention layers. Call after model build."""
        from vitra.utils.lora import apply_lora
        target = lora_cfg.get("target_modules", ["q_proj", "v_proj"])
        rank = lora_cfg.get("rank", 16)
        alpha = lora_cfg.get("alpha", 32)
        apply_lora(self.model.language_model, target, rank, alpha)
        self._has_lora = True

    def set_training_phase(self, phase: str):
        """Two-phase training for robot fine-tuning.

        phase1 (action expert warm-up):
            Train: action_head, P_w, view_embed, missing_cam_token,
                   cognition_token, fov_encoder
            Freeze: VLM backbone, vision_tower, P_g, LoRA adapters

        phase2 (joint fine-tune):
            Also unfreeze: P_g + LoRA adapters on LLM
            Vision encoder stays frozen.
        """
        self.model.config.use_cache = False

        # Freeze everything first
        for p in self.parameters():
            p.requires_grad_(False)

        # Always trainable across both phases
        self.act_model.requires_grad_(True)
        if self.cognition_token is not None:
            self.cognition_token.requires_grad_(True)
        if self.use_fov:
            self.fov_encoder.requires_grad_(True)

        # Wrist → action head cross-attention modules
        if self.use_wrist_cam_train:
            self.missing_wrist_tokens.requires_grad_(True)
            # Re-enable trainable input projection for depth channel.
            # set_training_phase freezes all params first, so we must
            # explicitly re-enable the encoder's input conv / patch embed.
            if self.wrist_encoder_type == "resnet34":
                self.wrist_rgbd_encoder.conv1.requires_grad_(True)
            else:
                self.wrist_rgbd_encoder.patch_embed.requires_grad_(True)
            if hasattr(self, 'wrist_projector') and not isinstance(self.wrist_projector, nn.Identity):
                self.wrist_projector.requires_grad_(True)

        # Phase 2: also unfreeze P_g + LoRA
        if phase == "phase2":
            self.model.multi_modal_projector.requires_grad_(True)
            if getattr(self, '_has_lora', False):
                from vitra.utils.lora import lora_params
                for p in lora_params(self):
                    p.requires_grad_(True)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[{phase}] Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M")

    @property
    def image_processor(self):
        return self.model.processor

    @property
    def hidden_size(self):
        return self.model.config.text_config.hidden_size

    @property
    def word_embedding(self):
        return self.model.language_model.model.embed_tokens

    @property
    def text_tower(self):
        return self.model.language_model.model

    @property
    def vision_tower(self):
        return self.model.vision_tower

    @property
    def model(self):
        return self.backbone

    def _encode_wrist_for_action_head(self, wrist_rgbd):
        """Encode 4-ch RGBD wrist image into spatial tokens for DiT cross-attention.

        Args:
            wrist_rgbd: [B, 4, H, W] float32.
                        Channels: [R, G, B, depth_metres].
                        RGB in [0, 1], depth in metres (positive).

        Returns:
            wrist_tokens: [B, N, D_hidden] spatial tokens for cross-attention.
        """
        B = wrist_rgbd.shape[0]

        # Normalise RGB with ImageNet stats; normalise depth to ~[0,1]
        rgb = wrist_rgbd[:, :3]   # [B, 3, H, W]
        depth = wrist_rgbd[:, 3:4]  # [B, 1, H, W]
        rgb_norm = (rgb - self.wrist_rgb_mean) / self.wrist_rgb_std
        depth_norm = torch.clamp(depth, 0.0, 1.5) / 1.5
        x = torch.cat([rgb_norm, depth_norm], dim=1)  # [B, 4, H, W]

        # Forward through encoder backbone
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            if self.wrist_encoder_type == "resnet34":
                # ResNet: forward_features → [B, C, H, W]
                feat_2d = self.wrist_rgbd_encoder.forward_features(x)
                _B, C, H, W = feat_2d.shape
                features = feat_2d.permute(0, 2, 3, 1).reshape(_B, H * W, C)
            else:
                # ViT: forward_features → [B, N+prefix, D]
                features = self.wrist_rgbd_encoder.forward_features(x)
                num_prefix = getattr(self.wrist_rgbd_encoder, "num_prefix_tokens", 1)
                features = features[:, num_prefix:]  # [B, N, D]

        # Project to DiT hidden dim
        wrist_tokens = self.wrist_projector(features.float())  # [B, N_raw, D_hidden]

        # Pool spatial tokens: 16×16 = 256 → 8×8 = 64 (4× reduction)
        if self.wrist_pool_size < self.wrist_grid_size:
            D = wrist_tokens.shape[-1]
            # Reshape to spatial grid: [B, G, G, D] → [B, D, G, G]
            wrist_tokens = wrist_tokens.reshape(
                B, self.wrist_grid_size, self.wrist_grid_size, D
            ).permute(0, 3, 1, 2)
            wrist_tokens = F.adaptive_avg_pool2d(
                wrist_tokens, (self.wrist_pool_size, self.wrist_pool_size)
            )
            # [B, D, P, P] → [B, P*P, D]
            wrist_tokens = wrist_tokens.permute(0, 2, 3, 1).reshape(
                B, self.wrist_num_patches, D
            )

        # Training-time masking: replace entire sample with learned fallback
        if self.training and self.wrist_mask_prob > 0:
            mask = torch.rand(B, device=wrist_tokens.device) < self.wrist_mask_prob
            N = wrist_tokens.shape[1]
            fallback = self.missing_wrist_tokens.expand(B, N, -1)
            wrist_tokens = torch.where(mask[:, None, None], fallback, wrist_tokens)

        return wrist_tokens

    def _forward_act_model(
        self,
        vlm_features: torch.Tensor,
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
        attention_mask: torch.Tensor = None,
        action_masks: Optional[torch.BoolTensor] = None,
        current_state: Optional[torch.FloatTensor] = None,
        current_state_mask: Optional[torch.BoolTensor] = None,
        wrist_features: Optional[torch.FloatTensor] = None,
        mode: str = "train",
        repeated_diffusion_steps: int = 1,
        cfg_scale: float = 5.0,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        **kwargs,
    ):
        
        actions = None
        action_loss = None

        B = vlm_features.shape[0]
        action_features = self.extract_cognition_token(vlm_features, attention_mask) #[B, D]

        # ── DiT diffusion head ────────────────────────────────────────────
        model_dtype = next(self.act_model.net.parameters()).dtype
        action_features = action_features.to(model_dtype)
        
        action_features_repeated = action_features.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1, 1)
        action_masks_repeated = action_masks.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1, 1)

        action_features_repeated = action_features_repeated.view(B*repeated_diffusion_steps, 1, action_features.shape[-1])
        action_masks_repeated = action_masks_repeated.view(B*repeated_diffusion_steps, action_masks.shape[1], action_masks.shape[2])

        # Repeat wrist spatial tokens if present  [B, N, D] → [B*R, N, D]
        wrist_features_repeated = None
        if wrist_features is not None:
            wrist_features = wrist_features.to(model_dtype)
            wrist_features_repeated = wrist_features.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1, 1)
            wrist_features_repeated = wrist_features_repeated.view(
                B * repeated_diffusion_steps, wrist_features.shape[1], wrist_features.shape[2]
            )

        if self.use_state == 'DiT':
            current_state_repeated = current_state.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1)
            current_state_repeated = current_state_repeated.view(B*repeated_diffusion_steps, 1, current_state.shape[1])
            current_state_mask_repeated = current_state_mask.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1)
            current_state_mask_repeated = current_state_mask_repeated.view(B*repeated_diffusion_steps, 1, current_state_mask.shape[1])
        else:
            current_state_repeated = None
            current_state_mask_repeated = None

        if mode == "train":
            actions_repeated = action_labels.unsqueeze(0).repeat(repeated_diffusion_steps, 1, 1, 1)
            actions_repeated = actions_repeated.view(B*repeated_diffusion_steps, action_labels.shape[1], action_labels.shape[2])
            if self.use_state == 'DiT':
                action_loss = self.act_model.loss(actions_repeated, action_features_repeated, action_masks_repeated, current_state_repeated, current_state_mask_repeated, wrist_features=wrist_features_repeated)
            else:
                action_loss = self.act_model.loss(actions_repeated, action_features_repeated, action_masks_repeated, wrist_features=wrist_features_repeated)
            return actions, action_loss
        else:
            # evaluate mode
            actions = self.act_model.sample(
                action_features_repeated,
                cfg_scale,
                current_state_repeated,
                current_state_mask_repeated,
                use_ddim,
                num_ddim_steps,
                action_masks_repeated,
                wrist_features=wrist_features_repeated,
            )

            return actions, action_loss

    def extract_cognition_token(self, output_hs, attention_mask):
        cumulative_sum = attention_mask.cumsum(dim=1)
        last_true_indices = (cumulative_sum == cumulative_sum.max(dim=1, keepdim=True)[0]).float().argmax(dim=1)
        expanded_indices = last_true_indices.unsqueeze(-1).expand(-1, output_hs.size(-1))
        action_features = output_hs.gather(1, expanded_indices.unsqueeze(1))  # [B, 1, D]
        return action_features

    def prepare_vlm_input_embeddings(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        current_state_mask: Optional[torch.BoolTensor] = None,
        current_state: Optional[torch.FloatTensor] = None,
        fov: Optional[torch.FloatTensor] = None,
        **kwargs
    ):
        B = input_ids.shape[0]
        word_embeds = self.model.get_input_embeddings()(input_ids)
        input_ids_mask = attention_mask 

        cog_ids = torch.ones_like(input_ids[:, 0:1]) * self.cognition_token_id # [B, 1]
        cog_embeds = self.model.get_input_embeddings()(cog_ids)
        cog_ids_mask = torch.ones_like(cog_ids, dtype=torch.bool)

        if hasattr(self, 'cognition_token') and self.cognition_token is not None:
            assert self.cognition_token.shape[0] == cog_embeds.shape[-1], f"cognition token shape {self.cognition_token.shape} does not match cog embeds shape {cog_embeds.shape}"
            cog_embeds = self.cognition_token.unsqueeze(0).unsqueeze(0).expand(B, -1, -1) 

        # Build the list of embeddings and masks to concatenate
        embeds_list = [word_embeds]
        masks_list = [input_ids_mask]
        num_additional_tokens = 0

        if self.use_state == 'VLM':
            current_state = current_state * current_state_mask.to(current_state.dtype)
            state_embeds = self.state_encoder(torch.cat([current_state, current_state_mask.to(current_state.dtype)], dim=1))
            state_ids_mask = torch.ones((B, 1), dtype=torch.bool).to(input_ids_mask.device)
        
            embeds_list.append(state_embeds.unsqueeze(1))
            masks_list.append(state_ids_mask)
            num_additional_tokens += 1

        if self.use_fov:
            fov_embeds = self.fov_encoder(fov)
            fov_ids_mask = torch.ones((B, 1), dtype=torch.bool).to(input_ids_mask.device)

            embeds_list.append(fov_embeds.unsqueeze(1))
            masks_list.append(fov_ids_mask)
            num_additional_tokens += 1

        # Always append cognition token at the end
        embeds_list.append(cog_embeds)
        masks_list.append(cog_ids_mask)
        num_additional_tokens += 1

        # Concatenate all
        inputs_embeds = torch.cat(embeds_list, dim=1)
        inputs_masks = torch.cat(masks_list, dim=1)

        # Note: Here we only use `self.cognition_token_id` as a placeholder for the token corresponding to the FOV or states (if any) input. 
        # In practice, the embedding passed to the LLM will be replaced with the actual FOV or states (if any) embedding.
        additional_tokens = torch.full((B, num_additional_tokens), self.cognition_token_id, dtype=input_ids.dtype, device=input_ids.device)

        inputs_embeds, attention_mask = move_masked_to_left(inputs_embeds, inputs_masks)
        input_ids = torch.cat([input_ids, additional_tokens], dim=1)
        input_ids, inputs_masks = move_masked_to_left_ids(input_ids, inputs_masks)

        past_seen_tokens = 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0) + 1  # Paligemma positions are 1-indexed
        # Merge text and images (head camera only — wrist goes to action head)
        if pixel_values is not None:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                image_features = self.model.get_image_features(pixel_values)

            special_image_mask = (input_ids == self.model.config.image_token_index).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            if inputs_embeds[special_image_mask].numel() != image_features.numel():
                image_tokens_in_text = torch.sum(input_ids == self.config.image_token_index)
                raise ValueError(
                    f"Number of images does not match number of special image tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                    "tokens from image embeddings."
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        causal_mask = self.model._update_causal_mask(
            attention_mask, None, None, cache_position, input_ids, inputs_embeds, False
        )
        return {
            "attention_mask": causal_mask,
            "position_ids": position_ids,
            "inputs_embeds": inputs_embeds,
            "cache_position": cache_position,
        }, inputs_masks

    def prepare_vlm_features(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        current_state_mask: Optional[torch.BoolTensor] = None,
        current_state: Optional[torch.FloatTensor] = None,
        fov: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
        **kwargs,
    ):

        vlm_inputs, inputs_masks = self.prepare_vlm_input_embeddings(
                pixel_values,
                input_ids,
                attention_mask,
                current_state_mask,
                current_state,
                fov,
                **kwargs,
            )

        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
            outputs = self.model.language_model(
                past_key_values=None,
                use_cache=use_cache,
                output_hidden_states=True,
                num_logits_to_keep=0, # can be modified
                **vlm_inputs
            )

        output_hs = outputs.hidden_states[-1]
        return output_hs, inputs_masks

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        position_ids: torch.LongTensor = None,
        action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
        action_masks: Optional[torch.BoolTensor] = None,
        current_state_mask: Optional[torch.BoolTensor] = None,
        current_state: Optional[torch.FloatTensor] = None,
        fov: Optional[torch.FloatTensor] = None,
        wrist_rgbd: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        mode="train",
        **kwargs,
    ):

        assert mode == "train", f"mode {mode} not supported in the forward function."

        loss = {}

        output_hs, inputs_masks = self.prepare_vlm_features(
            pixel_values,
            input_ids,
            attention_mask,
            current_state_mask,
            current_state,
            fov,
            use_cache,
            **kwargs,
        )

        # Encode wrist camera RGBD for action head conditioning
        wrist_features = None
        if self.use_wrist_cam_train and wrist_rgbd is not None:
            wrist_features = self._encode_wrist_for_action_head(wrist_rgbd)

        _, action_loss = self._forward_act_model(
            vlm_features = output_hs, 
            action_labels = action_labels, 
            attention_mask = inputs_masks, 
            action_masks = action_masks, 
            current_state = current_state, 
            current_state_mask = current_state_mask, 
            wrist_features = wrist_features,
            mode = mode,
            repeated_diffusion_steps = self.repeated_diffusion_steps,
        )

        self._update_loss(loss, action_loss)

        return loss

    def image_preprocess(self, image: Image, size: Tuple[int, int] = (224, 224)) -> Image:
        width, height = image.size

        return image
        

    def predict_action(
        self, 
        image, 
        instruction: str, 
        current_state, 
        current_state_mask=None, 
        use_ddim=True, 
        num_ddim_steps=10, 
        cfg_scale=5.0, 
        action_mask_torch=None, 
        fov=None, 
        sample_times=1, 
        use_cache=False,
        wrist_rgbd=None,
    ) -> np.ndarray:
        """Predict actions using DiT diffusion head.

        Args:
            image: head camera RGB (np.ndarray H×W×3 or PIL Image).
            instruction: text instruction.
            current_state: normalised 7-dim state [B, 7].
            current_state_mask: [B, 7] (default: all-ones).
            action_mask_torch: [B, T, 7] (default: all-ones for 7-dim).
            fov: [B, 2] field-of-view.
            wrist_rgbd: [B, 4, H, W] float32 RGBD (RGB in [0,1], depth metres).
            sample_times: number of diffusion samples.

        Returns:
            Predicted normalised 7-dim actions [sample_times, T, 7].
        """
        B = current_state.shape[0]
        assert B == 1, f"Batch size {B} not supported in predict_action for now."

        # Prepare head camera for VLM
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                image = Image.fromarray(image)
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        prefix = '<image>'
        model_inputs = self.processor(text=prefix + instruction, images=image, return_tensors="pt").to('cuda')
        pixel_value = model_inputs['pixel_values']
        input_ids = model_inputs['input_ids']

        if isinstance(pixel_value, torch.Tensor):
            pixel_value = pixel_value.to('cuda')
        elif isinstance(pixel_value, dict):
            pixel_value = {
                k: torch.stack([pixel_value[idx][k] for idx in range(len(input_ids))]).to('cuda') for k in pixel_value[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_value)}")
        if pixel_value.dim() == 5:
            pixel_value = pixel_value.view(-1, *pixel_value.shape[2:])

        attention_mask = torch.ones_like(input_ids, dtype=torch.bool).to(input_ids.device)
        current_state = current_state.to(input_ids.device)
        fov = fov.to(input_ids.device) if fov is not None else None

        # Encode wrist RGBD for action head conditioning
        wrist_features = None
        if self.use_wrist_cam_train and wrist_rgbd is not None:
            wrist_rgbd_t = wrist_rgbd.to('cuda') if isinstance(wrist_rgbd, torch.Tensor) else wrist_rgbd
            wrist_features = self._encode_wrist_for_action_head(wrist_rgbd_t)

        # Action mask: all 7 dims active (single-arm robot, all dims used)
        if action_mask_torch is None:
            x_mask = torch.ones(B, self.chunk_size, self.act_model.in_channels, device=input_ids.device)
        else:
            x_mask = action_mask_torch.to(input_ids.device)

        current_state_mask = current_state_mask.to(input_ids.device)

        output_hs, inputs_masks = self.prepare_vlm_features(
            pixel_value,
            input_ids,
            attention_mask,
            current_state_mask,
            current_state,
            fov,
            use_cache=use_cache,
        )
        # handle multiple samples for one input
        samples, _ = self._forward_act_model(
            vlm_features = output_hs,
            attention_mask = inputs_masks,
            action_masks = x_mask,
            current_state = current_state,
            current_state_mask = current_state_mask,
            wrist_features = wrist_features,
            mode = "eval",
            repeated_diffusion_steps = sample_times,
            cfg_scale = cfg_scale,
            use_ddim = use_ddim,
            num_ddim_steps = num_ddim_steps,
        )
        action_np = samples.cpu().numpy()  # [sample_times, T, 7]
        return action_np

    def _format_loss(self, loss):
        # for visualization and loss backward in pytorch
        _loss = 0
        _keys = list(loss.keys())

        for k in _keys:
            if "loss" in k:
                _loss += loss[k]

        loss["loss"] = _loss
        return loss

    @staticmethod
    def _update_loss(loss, new_loss, suffix=None):
        """
        use new_loss to update loss.
            * if suffix is not None, the key from new_loss will be reformatted as: key|suffix
            * otherwise, if the key from new_loss is not in loss, it will be directly used: key
            * otherwise, the key from the new_loss will be reformatted as: key|index, where index is
                searched from 0->+inf so that key|index is not in loss.

        """

        def get_key(k, d):
            if suffix is not None:
                new_k = f"{k}_{suffix}"
                assert new_k not in d
                return new_k

            ind = 0
            while True:
                if ind == 0:
                    new_k = k
                else:
                    new_k = f"{k}_{ind}"
                if new_k not in d:
                    return new_k
                ind += 1

        for k in new_loss:
            new_k = get_key(k, loss)
            loss[new_k] = new_loss[k]

        return loss