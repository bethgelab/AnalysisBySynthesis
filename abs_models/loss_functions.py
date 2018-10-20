import torch
from abs_models import utils as u


def squared_L2_loss(a, b, axes, keepdim=True):
    return u.tsum((a - b)**2, axes=axes, keepdim=keepdim)


def KLD(mu_latent_q, sig_q=1., dim=-3):
    """

    :param mu_latent_q: z must be shape (..., n_latent ...) at i-th pos
    :param sig_q:  scalar
    :param dim: determines pos i
    :return:
    """
    return -0.5 * torch.sum(1 - mu_latent_q ** 2 + u.tlog(sig_q) - sig_q**2,
                            dim=dim, keepdim=True)


def ELBOs(x_rec: torch.Tensor, samples_latent: torch.Tensor, x_orig: torch.Tensor,
          beta=1, dist_fct=squared_L2_loss,
          auto_batch_size=1600):
    """
    :param x_rec: shape (..., n_channels, nx, ny)
    :param samples_latent:  (..., n_latent, 1, 1)
    :param x_orig:  (..., n_channels, nx, ny)
    :param beta:
    :param dist_fct:
    :param auto_batch_size:
    :return:
    """
    n_ch, nx, ny = x_rec.shape[-3:]
    kld = KLD(samples_latent, sig_q=1.)
    if auto_batch_size is not None:
        rec_loss = u.auto_batch(auto_batch_size, dist_fct, x_orig, x_rec,
                                axes=[-1, -2, -3])   # sum over nx, ny, n_ch
    else:
        rec_loss = dist_fct(x_orig, x_rec, axes=[-1, -2, -3])
    elbo = rec_loss + beta * kld
    # del x_rec, x_orig, kld
    # del x_rec, samples_latent, x_orig
    return elbo / (n_ch * nx * ny)


