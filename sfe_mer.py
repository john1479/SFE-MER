"""SFE-MER: Selective Feature Exchange for Micro-Expression Recognition.

The model combines an MAE-pretrained ViT encoder, facial motion feature
refinement (FMFR), selective feature exchange (SFE), and exchange-guided
prediction regularization (EGPR).

The original model interfaces and loss formulations are retained for
compatibility with existing training scripts and checkpoints.
"""

import torch
from torch import nn
from torch.nn import functional as F

import models.models_mae as models_mae


def exchange_loss(
    exp_out,
    exchange_info,
    labels,
    w_inv=1.0,
    w_cons=1.0,
    w_comp=0.5,
    eps=1e-6,
    margin=0.10,
    comp_conf_thresh=0.0,
):
    """Compute exchange-guided prediction regularization (EGPR).

    Identity-guided exchange combines expression label preservation (L_inv)
    and prediction consistency (L_cons). Expression-guided exchange uses a
    relative confidence reduction objective (L_comp).

    Args:
        exp_out: Original expression logits of shape [B, C].
        exchange_info: Exchange metadata returned by SFEMER.forward().
        labels: Ground-truth expression labels of shape [B].
        w_inv: Weight of the label preservation loss.
        w_cons: Weight of the prediction consistency loss.
        w_comp: Weight of the expression sensitivity loss.
        eps: Retained for compatibility with existing callers; unused.
        margin: Desired minimum decrease in original-class confidence.
        comp_conf_thresh: Optional confidence threshold for eligible anchors.
            A value of 0 disables confidence filtering.

    Returns:
        total: Weighted exchange loss, as a scalar tensor.
        logs: Dictionary of loss values and exchange statistics.
    """
    labels = labels.view(-1).long()

    # Preserve the computation graph even when no eligible pairs exist.
    total = exp_out.sum() * 0.0
    logs = {
        "loss_inv": 0.0,
        "loss_cons": 0.0,
        "loss_comp": 0.0,
        "p_before": 0.0,
        "p_after": 0.0,
        "comp_active_ratio": 0.0,
    }

    # Identity-guided exchange: label preservation and prediction consistency.
    idw = exchange_info["identity_swap"]
    if idw["valid_count"] > 0:
        a = idw["anchor_idx"]
        swap_logits = idw["swap_logits"]
        target_labels = labels[a]

        # Preserve the original expression label after exchange.
        loss_inv = F.cross_entropy(swap_logits.float(), target_labels)

        # Use the detached original distribution as the consistency target.
        with torch.no_grad():
            target_prob = F.softmax(exp_out[a].float(), dim=-1)

        swap_log_prob = F.log_softmax(swap_logits.float(), dim=-1)
        loss_cons = F.kl_div(
            swap_log_prob, target_prob, reduction="batchmean"
        )

        total = total + w_inv * loss_inv + w_cons * loss_cons
        logs["loss_inv"] = loss_inv.detach().item()
        logs["loss_cons"] = loss_cons.detach().item()

    logs["id_swap_ratio"] = idw["valid_ratio"]

    # Expression-guided exchange: relative confidence reduction.
    exw = exchange_info["expression_swap"]
    if exw["valid_count"] > 0:
        a = exw["anchor_idx"]
        swap_logits = exw["swap_logits"]
        target_labels = labels[a]

        # Detach the original-class probability before exchange.
        with torch.no_grad():
            original_prob = F.softmax(exp_out[a].float(), dim=-1)
            p_before = original_prob.gather(
                1, target_labels.unsqueeze(1)
            ).squeeze(1)

        # Retain gradients for the original-class probability after exchange.
        swapped_prob = F.softmax(swap_logits.float(), dim=-1)
        p_after = swapped_prob.gather(1, target_labels.unsqueeze(1)).squeeze(1)

        # Penalize confidence reductions smaller than the specified margin.
        loss_per_sample = F.relu(margin + p_after - p_before)

        # Optionally exclude anchors with low original-class confidence.
        if comp_conf_thresh > 0.0:
            valid_mask = (p_before >= comp_conf_thresh).float()
        else:
            valid_mask = torch.ones_like(p_before)

        loss_comp = (
            (loss_per_sample * valid_mask).sum()
            / valid_mask.sum().clamp_min(1.0)
        )
        total = total + w_comp * loss_comp

        logs["loss_comp"] = loss_comp.detach().item()
        logs["p_before"] = p_before.mean().item()
        logs["p_after"] = p_after.detach().mean().item()
        logs["comp_active_ratio"] = valid_mask.mean().item()

    logs["exp_swap_ratio"] = exw["valid_ratio"]
    return total, logs


@torch.no_grad()
def branch_similarity(x):
    """Return the mean off-diagonal cosine similarity within each sample.

    Args:
        x: Tensor of shape [B, K, D], where K is the number of masks or
            regional features and D is the number of patches or channels.

    Returns:
        Mean pairwise cosine similarity as a float, or NaN if K <= 1.
    """
    x = F.normalize(x.detach().float(), p=2, dim=-1)
    K = x.size(1)
    if K <= 1:
        return float("nan")

    similarity = torch.matmul(x, x.transpose(1, 2))
    off_diagonal = ~torch.eye(K, dtype=torch.bool, device=x.device)
    return similarity[:, off_diagonal].mean().item()


def prepare_model(chkpt_dir, arch="mae_vit_large_patch16"):
    """Build an MAE encoder and load pretrained checkpoint weights.

    Args:
        chkpt_dir: Path to a trusted MAE checkpoint containing a model state.
        arch: Architecture factory name in models.models_mae.

    Returns:
        Initialized MAE model.
    """
    model = getattr(models_mae, arch)()
    checkpoint = torch.load(chkpt_dir, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    print(msg)
    return model


class SpatialMaskGenerator(nn.Module):
    """Generate learnable spatial masks from patch tokens.

    A lightweight MLP predicts one logit per region and patch. A
    temperature-controlled sigmoid produces deterministic soft masks.
    The legacy argument num_au denotes the number of spatial masks, not
    the number of supervised facial action units.
    """

    def __init__(self, token_dim, num_au, tau=0.5):
        super().__init__()
        self.tau = tau
        self.mask_net = nn.Sequential(
            nn.Linear(token_dim, token_dim // 4),
            nn.LayerNorm(token_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(token_dim // 4, num_au),
        )

    def forward(self, x_tokens, is_eval=False):
        """Map patch tokens [B, N, C] to spatial masks [B, num_au, N]."""
        logits = self.mask_net(x_tokens)
        logits = logits.permute(0, 2, 1)
        mask_weights = self.gumbel_sigmoid(logits, hard=is_eval)
        return mask_weights

    def gumbel_sigmoid(self, logits, hard=False, eps=1e-10):
        """Apply a temperature-controlled sigmoid without Gumbel noise.

        The legacy method name and the unused hard and eps arguments are
        retained for compatibility. Training and inference both use soft
        masks.
        """
        return torch.sigmoid(logits / self.tau)


class TokenSpatialModule(nn.Module):
    """Facial Motion Feature Refinement (FMFR).

    Extract regional motion features using learnable spatial masks, refine
    the features through mask-guided cross-attention, and adaptively fuse
    the regional representations. The single-region and uncentered modes
    are retained for compatibility with existing configurations.
    """

    def __init__(
        self,
        token_dim,
        num_tokens,
        num_au,
        center_tokens=True,
        mask_tau=0.5,
        mask_prior_strength=0.5,
    ):
        super().__init__()
        self.num_au = num_au
        self.center_tokens = center_tokens
        self.mask_prior_strength = mask_prior_strength

        # num_tokens is retained for compatibility with existing callers.
        self.mask_generator = SpatialMaskGenerator(
            token_dim=token_dim, num_au=num_au, tau=mask_tau
        )

        # Region scores estimate fusion importance, not AU class logits.
        if num_au > 1:
            self.region_score = nn.Sequential(
                nn.Linear(token_dim, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1),
            )
        else:
            self.region_score = None

        self.cross_attn_proj_q = nn.Linear(token_dim, token_dim)
        self.cross_attn_proj_k = nn.Linear(token_dim, token_dim)
        self.cross_attn_proj_v = nn.Linear(token_dim, token_dim)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, x_tokens):
        """Refine and fuse regional features from patch tokens [B, N, C].

        Returns:
            fused_au_feat: Refined regional features [B, num_au, C].
            mask_weights: Spatial mask weights [B, num_au, N].
            region_logits: Regional fusion scores [B, num_au].
            region_weights: Normalized fusion weights [B, num_au].
            local_feature: Fused expression representation [B, C].
        """
        B, N, C = x_tokens.shape

        # 1. Generate spatial masks from the original patch tokens.
        mask_weights = self.mask_generator(
            x_tokens, is_eval=not self.training
        )

        # 2. Apply spatial centering and mask-guided pooling.
        if self.center_tokens:
            token_mean = x_tokens.mean(dim=1, keepdim=True)
            pool_tokens = x_tokens - token_mean
        else:
            pool_tokens = x_tokens

        au_feats = torch.bmm(mask_weights, pool_tokens)
        mask_sum = mask_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        au_feats = au_feats / mask_sum

        # 3. Refine regional features with mask-guided cross-attention.
        # Queries: pooled regional features; keys: original patch tokens;
        # values: centered patch tokens (or original tokens if disabled).
        q = self.cross_attn_proj_q(au_feats)
        k = self.cross_attn_proj_k(x_tokens)
        v = self.cross_attn_proj_v(pool_tokens)
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (C ** 0.5)

        # Add the spatial mask prior to the attention logits.
        mask_log_prior = torch.log(mask_weights.clamp_min(1e-6))
        attn_scores = attn_scores + self.mask_prior_strength * mask_log_prior
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        fused_au_feat = self.norm(au_feats + attn_output)

        # 4. Adaptively fuse the refined regional features.
        if self.num_au == 1:
            local_feature = fused_au_feat[:, 0, :]
            region_logits = fused_au_feat.new_zeros((B, 1))
            region_weights = fused_au_feat.new_ones((B, 1))
        else:
            region_logits = self.region_score(fused_au_feat).squeeze(-1)
            region_weights = F.softmax(region_logits, dim=-1)
            local_feature = torch.sum(
                fused_au_feat * region_weights.unsqueeze(-1), dim=1
            )

        return (
            fused_au_feat,
            mask_weights,
            region_logits,
            region_weights,
            local_feature,
        )


class SelectiveDualFeatureExchange(nn.Module):
    """Selective Feature Exchange (SFE) in a shared expression space.

    Identity-guided pairs share an expression label but differ in subject
    identity. Expression-guided pairs share a subject identity but differ
    in expression label. Selected anchor dimensions are replaced by donor
    values when alpha is 1, or interpolated otherwise.

    Top-k is the default selection strategy. Bottom-k and random selection
    are supported for ablation experiments.
    """

    def __init__(
        self, exchange_ratio=0.25, min_exchange_dims=1, selection_strategy="topk"
    ):
        super().__init__()
        if not 0.0 < exchange_ratio <= 1.0:
            raise ValueError("exchange_ratio must be in (0, 1]")
        if selection_strategy not in ("topk", "bottomk", "random"):
            raise ValueError(
                "selection_strategy must be 'topk', 'bottomk', or 'random'"
            )
        self.exchange_ratio = exchange_ratio
        self.min_exchange_dims = min_exchange_dims
        self.selection_strategy = selection_strategy

    @staticmethod
    def _sample_pairs(candidate_mask):
        """Sample one eligible donor for every anchor with valid candidates."""
        B = candidate_mask.size(0)
        device = candidate_mask.device
        valid_mask = candidate_mask.any(dim=1)
        if not valid_mask.any():
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty

        random_score = torch.rand(B, B, device=device).masked_fill(
            ~candidate_mask, -1.0
        )
        donor_idx_all = random_score.argmax(dim=1)
        anchor_idx_all = torch.arange(B, device=device)
        return anchor_idx_all[valid_mask], donor_idx_all[valid_mask]

    def _build_exchange_mask(self, anchor_feat, donor_feat):
        """Select an equal number of exchange dimensions for every pair."""
        D = anchor_feat.size(1)
        difference = (anchor_feat - donor_feat).abs()
        k = min(D, max(self.min_exchange_dims, int(round(D * self.exchange_ratio))))

        if self.selection_strategy == "topk":
            # Select dimensions with the largest absolute differences.
            selected_idx = difference.detach().topk(
                k=k, dim=-1, largest=True
            ).indices
        elif self.selection_strategy == "bottomk":
            # Ablation: select dimensions with the smallest differences.
            selected_idx = difference.detach().topk(
                k=k, dim=-1, largest=False
            ).indices
        else:
            # Ablation: sample distinct dimensions independently per pair.
            random_scores = torch.rand(difference.shape, device=difference.device)
            selected_idx = random_scores.topk(k=k, dim=-1).indices

        # Index selection is discrete; exchanged feature values retain gradients.
        mask = torch.zeros_like(difference).scatter_(1, selected_idx, 1.0)
        return mask

    def _exchange(self, feature, candidate_mask, alpha):
        """Exchange selected feature dimensions for eligible anchor-donor pairs."""
        B = feature.size(0)
        anchor_idx, donor_idx = self._sample_pairs(candidate_mask)
        if anchor_idx.numel() == 0:
            return {
                "anchor_idx": anchor_idx,
                "donor_idx": donor_idx,
                "swapped_feat": None,
                "mask": None,
                "valid_count": 0,
                "valid_ratio": 0.0,
            }

        anchor_feat = feature[anchor_idx]
        donor_feat = feature[donor_idx]
        mask = self._build_exchange_mask(anchor_feat, donor_feat)

        # Keep the legacy interpolation coefficient for API compatibility.
        swapped_feat = anchor_feat + alpha * mask * (donor_feat - anchor_feat)
        return {
            "anchor_idx": anchor_idx,
            "donor_idx": donor_idx,
            "swapped_feat": swapped_feat,
            "mask": mask,
            "valid_count": int(anchor_idx.numel()),
            "valid_ratio": float(anchor_idx.numel() / B),
        }

    def forward(self, exp_feat, labels, subjects, alpha=1.0):
        """Construct both types of sample pairs and return exchange metadata."""
        labels = labels.view(-1)
        subjects = subjects.view(-1)
        B = labels.size(0)

        same_exp = labels[:, None] == labels[None, :]
        same_sub = subjects[:, None] == subjects[None, :]
        not_self = ~torch.eye(B, dtype=torch.bool, device=labels.device)

        identity_swap = self._exchange(
            exp_feat, same_exp & ~same_sub & not_self, alpha
        )
        expression_swap = self._exchange(
            exp_feat, same_sub & ~same_exp & not_self, alpha
        )
        return {
            "identity_swap": identity_swap,
            "expression_swap": expression_swap,
        }


class SFEMER(nn.Module):
    """SFE-MER model with a shared expression representation.

    The legacy class name and parameter names are preserved for compatibility
    with the existing training code. The model has no separate identity
    encoder or identity classifier. Feature exchange is bypassed when
    is_eval=True.
    """

    def __init__(
        self,
        num_classes=5,
        au_num=8,
        exchange_ratio=0.25,
        mask_tau=0.5,
        mask_prior_strength=0.5,
        mae_ckpt="pretrained_models/mae_visualize_vit_large.pth",
        selection_strategy="topk",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.au_num = au_num

        self.MAE = prepare_model(mae_ckpt, arch="mae_vit_large_patch16")
        if hasattr(self.MAE, "cls_token"):
            embed_dim = self.MAE.cls_token.shape[-1]
        elif hasattr(self.MAE, "pos_embed"):
            embed_dim = self.MAE.pos_embed.shape[-1]
        else:
            embed_dim = 1024
        self.embed_dim = embed_dim

        self.exp_masker = TokenSpatialModule(
            token_dim=embed_dim,
            num_tokens=196,
            num_au=au_num,
            center_tokens=True,
            mask_tau=mask_tau,
            mask_prior_strength=mask_prior_strength,
        )
        self.feature_exchange = SelectiveDualFeatureExchange(
            exchange_ratio=exchange_ratio,
            selection_strategy=selection_strategy,
        )
        self.exp_classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(
        self, img, subjects=None, labels=None, is_eval=False, alpha=1.0, **kwargs
    ):
        """Extract expression representations and optionally exchange features.

        Args:
            img: Input tensor [B, 3, C, H, W]; the third stream contains
                the optical flow representation.
            subjects: Subject IDs [B], required for feature exchange.
            labels: Expression labels [B], required for feature exchange.
            is_eval: If True, bypass exchange and return logits and features.
            alpha: Legacy exchange interpolation coefficient (1 = replacement).
            **kwargs: Additional arguments retained for API compatibility.

        Returns:
            If is_eval=True: (exp_out, f_exp).
            Otherwise: (exp_out, f_exp, region_logits, mask_weights,
            exchange_info).
        """
        img_flow = img[:, 2]

        # Encode the motion input without masking and restore patch order.
        features, _, ids_restore = self.MAE.forward_encoder(img_flow, mask_ratio=0)
        f_patches = features[:, 1:, :]
        f_patches = torch.gather(
            f_patches,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, f_patches.shape[-1]),
        )

        # Refine patch tokens and classify the fused expression feature.
        _, mask_weights, region_logits, _, f_exp = self.exp_masker(f_patches)
        exp_out = self.exp_classifier(f_exp)

        if is_eval:
            return exp_out, f_exp

        # Both exchange strategies operate in the same expression feature space.
        exchange_info = self.feature_exchange(
            exp_feat=f_exp, labels=labels, subjects=subjects, alpha=alpha
        )

        # Reuse the expression classifier for both exchanged representations.
        for key in ("identity_swap", "expression_swap"):
            info = exchange_info[key]
            info["swap_logits"] = (
                self.exp_classifier(info["swapped_feat"])
                if info["valid_count"] > 0
                else None
            )

        return exp_out, f_exp, region_logits, mask_weights, exchange_info
