"""
Centered Kernel Alignment (CKA) Loss for Cross-Modality Geometry Regularization.

CKA measures the similarity between representations by comparing their geometry.
This implementation uses a linear kernel for efficiency.
"""

# import torch


# def cka_loss(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
#     """
#     Compute CKA loss (1 - CKA similarity) for cross-modality alignment.
    
#     CKA measures the similarity of representational geometry between two sets of features.
#     Minimizing this loss encourages similar within-modality geometry across modalities.
    
#     Args:
#         X: Features from modality 1, shape (n_samples, feature_dim)
#         Y: Features from modality 2, shape (n_samples, feature_dim)
        
#     Returns:
#         CKA loss value (1 - CKA), scalar tensor. Lower means more similar geometry.
#     """
#     # Ensure inputs are 2D
#     if X.dim() > 2:
#         X = X.reshape(-1, X.shape[-1])
#     if Y.dim() > 2:
#         Y = Y.reshape(-1, Y.shape[-1])
    
#     # Use minimum number of samples from both modalities
#     n = min(X.shape[0], Y.shape[0])
#     X = X[:n]
#     Y = Y[:n]
    
#     # Center features (important for CKA)
#     X = X - X.mean(dim=0, keepdim=True)
#     Y = Y - Y.mean(dim=0, keepdim=True)
    
#     # Compute linear kernel Gram matrices
#     # K = X @ X.T, L = Y @ Y.T
#     K = torch.mm(X, X.t())
#     L = torch.mm(Y, Y.t())
    
#     # HSIC computation using Frobenius inner product
#     # HSIC(X, Y) = trace(K @ H @ L @ H) / (n-1)^2
#     # With centering already done, we can use simplified form
#     hsic_xy = (K * L).sum()
#     hsic_xx = (K * K).sum()
#     hsic_yy = (L * L).sum()
    
#     # CKA = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))
#     denominator = torch.sqrt(hsic_xx * hsic_yy)
    
#     # Add small epsilon for numerical stability
#     cka = hsic_xy / (denominator + 1e-8)
    
#     # Clamp to valid range to handle numerical issues
#     cka = torch.clamp(cka, -1.0, 1.0)
    
#     # Return loss: 1 - CKA (minimize to maximize alignment)
#     return 1.0 - cka


import torch


def cka_loss(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Compute CKA loss (1 - CKA similarity) for cross-modality alignment.

    Changes vs your original (without changing the signature):
      - If inputs are >=3D (e.g., (B, T, D)), pool to (B, D) by mean over tokens
        (avoids mixing tokens across different examples).
      - Compute CKA in float32 for numerical stability.
      - Stabilize hsic_xx / hsic_yy as well (not just denominator).
      - Remove clamp (can zero gradients if saturated); use nan_to_num instead.
    """

    def _to_2d(Z: torch.Tensor) -> torch.Tensor:
        if Z.dim() == 2:
            return Z
        if Z.dim() >= 3:
            # Treat first dim as batch; pool over all remaining non-feature dims
            # (B, ..., D) -> (B, N, D) -> (B, D)
            B = Z.shape[0]
            Z = Z.reshape(B, -1, Z.shape[-1]).mean(dim=1)
            return Z
        raise ValueError(f"Expected tensor with dim >= 2, got dim={Z.dim()}")

    # Convert to (n_samples, feature_dim) with sensible pooling
    X2 = _to_2d(X)
    Y2 = _to_2d(Y)

    # Use minimum number of samples from both modalities
    n = min(X2.shape[0], Y2.shape[0])
    if n < 2:
        # Not enough samples to define geometry; return a neutral/high loss
        return X2.new_tensor(1.0)

    X2 = X2[:n]
    Y2 = Y2[:n]

    # Compute in float32 for stability (still fully differentiable)
    Xc = X2.to(dtype=torch.float32)
    Yc = Y2.to(dtype=torch.float32)

    # Center across samples
    Xc = Xc - Xc.mean(dim=0, keepdim=True)
    Yc = Yc - Yc.mean(dim=0, keepdim=True)

    # Linear Gram matrices
    K = Xc @ Xc.t()  # (n, n)
    L = Yc @ Yc.t()  # (n, n)

    # HSIC via Frobenius inner products
    eps = 1e-8
    hsic_xy = (K * L).sum()
    hsic_xx = (K * K).sum().add(eps)
    hsic_yy = (L * L).sum().add(eps)

    denom = torch.sqrt(hsic_xx * hsic_yy).add(eps)
    cka = hsic_xy / denom

    # Avoid NaNs/Infs without clamping (clamp can kill gradients when saturated)
    cka = torch.nan_to_num(cka, nan=0.0, posinf=0.0, neginf=0.0)

    return 1.0 - cka

