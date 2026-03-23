from typing import Any, Dict, Optional

import numpy as np

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType
from torchvision import models
import torch.nn.functional as F
import torchvision.transforms as T
from ray.rllib.models.torch.misc import normc_initializer
from loguru import logger
from torchvision.models import mobilenet_v3_small
from torchvision import transforms

torch, nn = try_import_torch()

class VizDoomEncoderLite(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        features1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),     # (32, 31, 31)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),    # (64, 14, 14)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),   # (128, 6, 6)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (256, 4, 4)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                   # (256, 1, 1)
            )
        features2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),     # (32, 31, 31)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),    # (64, 14, 14)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),   # (128, 6, 6)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (256, 4, 4)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                   # (256, 1, 1)
            )

        fc1 = nn.Sequential(
            nn.Flatten(),                  # (256,)
            nn.Linear(256, 368),           # å‚æ•°ï¼š131K
            )
        fc2 = nn.Sequential(
            nn.Flatten(),                  # (256,)
            nn.Linear(256, 128),           # å‚æ•°ï¼š131K
            )
        
        self.encoder_rgb = nn.Sequential(features1, fc1)
        self.encoder_layout = nn.Sequential(features2, fc2)
        self.encoder_grasp = nn.Embedding(num_embeddings = 2, embedding_dim = 16)

    def forward(self, x):
        batch_shape = x.shape[:-1]
        rgb_layout = x[:,:,:-1].reshape(*batch_shape,128,128,6)
        grasping = x[:,:,-1]

        rgb = rgb_layout[:,:,:,:,:3]
        rgb = rgb.float() / 255.0  # Normalize input to [0, 1]
        B, T, H, W, C = rgb.shape
        rgb = rgb.permute(0, 1, 4, 2, 3).contiguous()  # [B, T, 3, H, W]
        rgb = rgb.view(B * T, C, H, W)                 # [B*T, 3, 128, 128]
        rgb = self.encoder_rgb(rgb)                    # [B*T, 368, 1, 1]
        rgb = rgb.view(B, T, -1)                       # [B, T, 368]
        
        layout = rgb_layout[:,:,:,:,3:]
        layout = layout.float() / 255.0  # Normalize input to [0, 1]
        B, T, H, W, C = layout.shape
        layout = layout.permute(0, 1, 4, 2, 3).contiguous()  # [B, T, 3, H, W]
        layout = layout.view(B * T, C, H, W)                 # [B*T, 3, 128, 128]
        layout = self.encoder_layout(layout)                 # [B*T, 128, 1, 1]
        layout = layout.view(B, T, -1)                       # [B, T, 128]

        grasp = grasping.long()
        B, T = grasp.shape
        grasp = grasp.view(B * T) 
        grasp = self.encoder_grasp(grasp)
        grasp = grasp.view(B , T, -1) 
        
        feature = torch.cat([rgb, grasp, layout], dim=2)

        return feature
    
    
class NoLSTMVizDoomEncoderLite(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        features1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),     # (32, 31, 31)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),    # (64, 14, 14)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),   # (128, 6, 6)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (256, 4, 4)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                   # (256, 1, 1)
            )
        features2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),     # (32, 31, 31)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),    # (64, 14, 14)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),   # (128, 6, 6)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (256, 4, 4)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                   # (256, 1, 1)
            )

        fc1 = nn.Sequential(
            nn.Flatten(),                  # (256,)
            nn.Linear(256, 368),           # å‚æ•°ï¼š131K
            )
        fc2 = nn.Sequential(
            nn.Flatten(),                  # (256,)
            nn.Linear(256, 128),           # å‚æ•°ï¼š131K
            )
        
        self.encoder_rgb = nn.Sequential(features1, fc1)
        self.encoder_layout = nn.Sequential(features2, fc2)
        self.encoder_grasp = nn.Embedding(num_embeddings = 2, embedding_dim = 16)

    def forward(self, x):
        batch_size = x.shape[0]
        rgb_layout = x[:,:-1].reshape(batch_size,128,128,6)
        grasping = x[:,-1]

        rgb = rgb_layout[:,:,:,:3]
        rgb = rgb.float() / 255.0  # Normalize input to [0, 1]
        B, H, W, C = rgb.shape
        rgb = rgb.permute(0, 3, 1, 2).contiguous()  # [B, 3, H, W]
        rgb = self.encoder_rgb(rgb)                    # [B*T, 368, 1, 1]
        rgb = rgb.view(B, -1)
        
        layout = rgb_layout[:,:,:,3:]
        layout = layout.float() / 255.0
        B, H, W, C = layout.shape
        layout = layout.permute(0, 3, 1, 2).contiguous()
        layout = self.encoder_layout(layout)
        layout = layout.view(B, -1)

        grasp = grasping.long() 
        grasp = self.encoder_grasp(grasp)
        grasp = grasp.view(B, -1) 
        
        feature = torch.cat([rgb, grasp, layout], dim=1)

        return feature
    
def preprocess_uint8_tensor(t: torch.Tensor) -> torch.Tensor:
    t = t.float() / 255.0  # Now float32 in [0, 1]
    # Step 2: Resize to (224, 224)
    t = F.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)
    # Step 3: Normalize using ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406], device=t.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=t.device).view(1, 3, 1, 1)
    t = (t - mean) / std

    return t


class VizDoomEncoderLite_pretrained(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.image_features = mobilenet_v3_small(pretrained=True)
        self.image_features.classifier[3] = nn.Identity()
        for param in self.image_features.parameters():
            param.requires_grad = False

        self.fc_rgb = nn.Linear(1024, 368)
        self.fc_layout = nn.Linear(1024, 128)
        self.encoder_grasp = nn.Embedding(num_embeddings = 2, embedding_dim = 16)

    def forward(self, x):
        batch_shape = x.shape[:-1]
        rgb_layout = x[:,:,:-1].reshape(*batch_shape,128,128,6)
        grasping = x[:,:,-1]

        rgb = rgb_layout[:,:,:,:,:3]
        # rgb = rgb.float() / 255.0  # Normalize input to [0, 1]
        B, T, H, W, C = rgb.shape
        rgb = rgb.permute(0, 1, 4, 2, 3).contiguous()  # [B, T, 3, H, W]
        rgb = rgb.view(B * T, C, H, W)                 # [B*T, 3, 128, 128]
        rgb = self.fc_rgb(self.image_features(preprocess_uint8_tensor(rgb)))                   # [B*T, 368, 1, 1]
        rgb = rgb.view(B, T, -1)                       # [B, T, 368]
        
        layout = rgb_layout[:,:,:,:,3:]
        # layout = layout.float() / 255.0  # Normalize input to [0, 1]
        B, T, H, W, C = layout.shape
        layout = layout.permute(0, 1, 4, 2, 3).contiguous()  # [B, T, 3, H, W]
        layout = layout.view(B * T, C, H, W)                 # [B*T, 3, 128, 128]
        layout = self.fc_layout(self.image_features(preprocess_uint8_tensor(layout)))                # [B*T, 128, 1, 1]
        layout = layout.view(B, T, -1)                       # [B, T, 128]

        grasp = grasping.long()
        B, T = grasp.shape
        grasp = grasp.view(B * T) 
        grasp = self.encoder_grasp(grasp)
        grasp = grasp.view(B , T, -1) 
        
        feature = torch.cat([rgb, grasp, layout], dim=2)

        return feature
    
class LSTMContainingRLModule(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def setup(self):
        in_size = 512

        self.nature_cnn = VizDoomEncoderLite()

        # Get the LSTM cell size from the `model_config` attribute:
        self._lstm_cell_size = self.model_config.get("lstm_cell_size", 512)
        self._lstm = nn.LSTM(in_size, self._lstm_cell_size, batch_first=True)
        in_size = self._lstm_cell_size

        # Build a sequential stack.
        layers = []
        # Get the dense layer pre-stack configuration from the same config dict.
        dense_layers = self.model_config.get("dense_layers", [128, 128])
        for out_size in dense_layers:
            # Dense layer.
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.Tanh())
            in_size = out_size

        self._fc_net = nn.Sequential(*layers)

        # Logits layer (no bias, no activation).
        self._pi_head = nn.Linear(in_size, self.action_space.n)
        # Single-node value layer.
        self._values = nn.Linear(in_size, 1)
        normc_initializer(0.01)(self._values.weight)

    @override(TorchRLModule)
    def get_initial_state(self) -> Any:
        return {
            "h": np.zeros(shape=(self._lstm_cell_size,), dtype=np.float32),
            "c": np.zeros(shape=(self._lstm_cell_size,), dtype=np.float32)
        }

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        # Compute the basic 1D embedding tensor (inputs to policy- and value-heads).
        embeddings, state_outs = self._compute_embeddings_and_state_outs(batch)
        logits = self._pi_head(embeddings)

        # Return logits as ACTION_DIST_INPUTS (categorical distribution).
        # Note that the default `GetActions` connector piece (in the EnvRunner) will
        # take care of argmax-"sampling" from the logits to yield the inference (greedy)
        # action.
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.STATE_OUT: state_outs,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        # Same logic as _forward, but also return embeddings to be used by value
        # function branch during training.
        embeddings, state_outs = self._compute_embeddings_and_state_outs(batch)
        logits = self._pi_head(embeddings)
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.STATE_OUT: state_outs,
            Columns.EMBEDDINGS: embeddings,
        }

    # We implement this RLModule as a ValueFunctionAPI RLModule, so it can be used
    # by value-based methods like PPO or IMPALA.
    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: Dict[str, Any], embeddings: Optional[Any] = None
    ) -> TensorType:
        if embeddings is None:
            embeddings, _ = self._compute_embeddings_and_state_outs(batch)
        values = self._values(embeddings).squeeze(-1)
        return values

    def _compute_embeddings_and_state_outs(self, batch):
        obs = batch[Columns.OBS]
        state_in = batch[Columns.STATE_IN]
        h, c = state_in["h"], state_in["c"]
        # Unsqueeze the layer dim (we only have 1 LSTM layer).
        _obs = self.nature_cnn(obs)
        embeddings, (h, c) = self._lstm(_obs, (h.unsqueeze(0), c.unsqueeze(0)))
        # Push through our FC net.
        embeddings = self._fc_net(embeddings)
        # Squeeze the layer dim (we only have 1 LSTM layer).
        return embeddings, {"h": h.squeeze(0), "c": c.squeeze(0)}
    
class NoLSTMRLModule(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def setup(self):
        in_size = 512

        self.nature_cnn = NoLSTMVizDoomEncoderLite()
        # Build a sequential stack.
        layers = []
        # Get the dense layer pre-stack configuration from the same config dict.
        dense_layers = self.model_config.get("dense_layers", [128, 128])
        for out_size in dense_layers:
            # Dense layer.
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.Tanh())
            in_size = out_size

        self._fc_net = nn.Sequential(*layers)

        # Logits layer (no bias, no activation).
        self._pi_head = nn.Linear(in_size, self.action_space.n)
        # Single-node value layer.
        self._values = nn.Linear(in_size, 1)
        normc_initializer(0.01)(self._values.weight)

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        # Compute the basic 1D embedding tensor (inputs to policy- and value-heads).
        embeddings = self._compute_embeddings_and_state_outs(batch)
        logits = self._pi_head(embeddings)

        # Return logits as ACTION_DIST_INPUTS (categorical distribution).
        # Note that the default `GetActions` connector piece (in the EnvRunner) will
        # take care of argmax-"sampling" from the logits to yield the inference (greedy)
        # action.
        return {
            Columns.ACTION_DIST_INPUTS: logits,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        # Same logic as _forward, but also return embeddings to be used by value
        # function branch during training.
        embeddings = self._compute_embeddings_and_state_outs(batch)
        logits = self._pi_head(embeddings)
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.EMBEDDINGS: embeddings,
        }

    # We implement this RLModule as a ValueFunctionAPI RLModule, so it can be used
    # by value-based methods like PPO or IMPALA.
    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: Dict[str, Any], embeddings: Optional[Any] = None
    ) -> TensorType:
        if embeddings is None:
            embeddings=  self._compute_embeddings_and_state_outs(batch)
        values = self._values(embeddings).squeeze(-1)
        return values

    def _compute_embeddings_and_state_outs(self, batch):
        obs = batch[Columns.OBS]
        # Unsqueeze the layer dim (we only have 1 LSTM layer).
        _obs = self.nature_cnn(obs)
        # Push through our FC net.
        embeddings = self._fc_net(_obs)
        return embeddings
    
class LSTMContainingRLModule_pretrained(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def setup(self):
        in_size = 512

        self.nature_cnn = VizDoomEncoderLite_pretrained()

        # Get the LSTM cell size from the `model_config` attribute:
        self._lstm_cell_size = self.model_config.get("lstm_cell_size", 512)
        self._lstm = nn.LSTM(in_size, self._lstm_cell_size, batch_first=True)
        in_size = self._lstm_cell_size

        # Build a sequential stack.
        layers = []
        # Get the dense layer pre-stack configuration from the same config dict.
        dense_layers = self.model_config.get("dense_layers", [128, 128])
        for out_size in dense_layers:
            # Dense layer.
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.Tanh())
            in_size = out_size

        self._fc_net = nn.Sequential(*layers)

        # Logits layer (no bias, no activation).
        self._pi_head = nn.Linear(in_size, self.action_space.n)
        # Single-node value layer.
        self._values = nn.Linear(in_size, 1)
        normc_initializer(0.01)(self._values.weight)

    @override(TorchRLModule)
    def get_initial_state(self) -> Any:
        return {
            "h": np.zeros(shape=(self._lstm_cell_size,), dtype=np.float32),
            "c": np.zeros(shape=(self._lstm_cell_size,), dtype=np.float32)
        }

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        # Compute the basic 1D embedding tensor (inputs to policy- and value-heads).
        embeddings, state_outs = self._compute_embeddings_and_state_outs(batch)
        logits = self._pi_head(embeddings)

        # Return logits as ACTION_DIST_INPUTS (categorical distribution).
        # Note that the default `GetActions` connector piece (in the EnvRunner) will
        # take care of argmax-"sampling" from the logits to yield the inference (greedy)
        # action.
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.STATE_OUT: state_outs,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        # Same logic as _forward, but also return embeddings to be used by value
        # function branch during training.
        embeddings, state_outs = self._compute_embeddings_and_state_outs(batch)
        logits = self._pi_head(embeddings)
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.STATE_OUT: state_outs,
            Columns.EMBEDDINGS: embeddings,
        }

    # We implement this RLModule as a ValueFunctionAPI RLModule, so it can be used
    # by value-based methods like PPO or IMPALA.
    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: Dict[str, Any], embeddings: Optional[Any] = None
    ) -> TensorType:
        if embeddings is None:
            embeddings, _ = self._compute_embeddings_and_state_outs(batch)
        values = self._values(embeddings).squeeze(-1)
        return values

    def _compute_embeddings_and_state_outs(self, batch):
        obs = batch[Columns.OBS]
        state_in = batch[Columns.STATE_IN]
        h, c = state_in["h"], state_in["c"]
        # Unsqueeze the layer dim (we only have 1 LSTM layer).
        _obs = self.nature_cnn(obs)
        embeddings, (h, c) = self._lstm(_obs, (h.unsqueeze(0), c.unsqueeze(0)))
        # Push through our FC net.
        embeddings = self._fc_net(embeddings)
        # Squeeze the layer dim (we only have 1 LSTM layer).
        return embeddings, {"h": h.squeeze(0), "c": c.squeeze(0)}