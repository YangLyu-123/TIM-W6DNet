import torch
def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad) 
def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def _cal_kl(mu, logvar, lambda_kl):
    if lambda_kl > 0.0:
        logvar = torch.clamp(logvar, max=10.0)  # to prevent nan
        loss_kl = torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * (-0.5 * lambda_kl)
    else:
        loss_kl = 0.0
    return loss_kl