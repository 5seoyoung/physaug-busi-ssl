# -*- coding: utf-8 -*-
import torch, torch.nn as nn
import timm

def mlp(in_dim, hid, out):
    return nn.Sequential(
        nn.Linear(in_dim, hid),
        nn.BatchNorm1d(hid),
        nn.ReLU(inplace=True),
        nn.Linear(hid, out)
    )

class OnlineNet(nn.Module):
    """
    BYOL online network with lazy projector/predictor:
    - encoder: timm backbone (num_classes=0, global_pool='avg')
    - projector/predictor are created on-the-fly based on encoder output dim
      and moved to the SAME device as the encoder output.
    """
    def __init__(self, encoder_name="mobilenetv3_small_100", proj_dim=256, proj_hidden=4096, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        self._proj_dim = proj_dim
        self._proj_hidden = proj_hidden
        self.projector = None  # lazy
        self.predictor = None  # lazy

    def _ensure_heads(self, feat_dim: int, device):
        if self.projector is None:
            self.projector = mlp(feat_dim, self._proj_hidden, self._proj_dim).to(device)
            self.add_module("projector", self.projector)
        if self.predictor is None:
            hid = max(256, self._proj_hidden // 4)
            self.predictor = mlp(self._proj_dim, hid, self._proj_dim).to(device)
            self.add_module("predictor", self.predictor)

    def forward(self, x):
        z = self.encoder(x)           # [N, feat_dim]
        if isinstance(z, (list, tuple)):
            z = z[-1]
        self._ensure_heads(z.shape[1], z.device)
        p = self.projector(z)         # [N, proj_dim]
        q = self.predictor(p)         # [N, proj_dim]
        return p, q

class TargetNet(nn.Module):
    """
    BYOL target network with lazy projector (no predictor).
    """
    def __init__(self, encoder_name="mobilenetv3_small_100", proj_dim=256, proj_hidden=4096, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(encoder_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        self._proj_dim = proj_dim
        self._proj_hidden = proj_hidden
        self.projector = None  # lazy

    def _ensure_head(self, feat_dim: int, device):
        if self.projector is None:
            self.projector = mlp(feat_dim, self._proj_hidden, self._proj_dim).to(device)
            self.add_module("projector", self.projector)

    @torch.no_grad()
    def forward(self, x):
        z = self.encoder(x)
        if isinstance(z, (list, tuple)):
            z = z[-1]
        self._ensure_head(z.shape[1], z.device)
        p = self.projector(z)
        return p

@torch.no_grad()
def update_momentum(model_q: OnlineNet, model_k: TargetNet, m: float):
    for p_q, p_k in zip(model_q.parameters(), model_k.parameters()):
        p_k.data.mul_(m).add_(p_q.data, alpha=1. - m)

def byol_loss(p, z):
    p = nn.functional.normalize(p, dim=1)
    z = nn.functional.normalize(z, dim=1)
    return 2 - 2 * (p * z).sum(dim=1).mean()
