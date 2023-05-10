import numpy as np
import jax
import jax.numpy as jnp
import json
from register_to_reference import register_to_reference


@jax.jit
def proj_A(f0, exp_data):
    f1 = jnp.fft.fftn(f0)
    f1 = exp_data * jnp.exp(1j * jnp.angle(f1))
    f1 = jnp.fft.ifftn(f1).real

    return f1


@jax.jit
def proj_supp(f0, support_mask):
    f2 = f0 * support_mask
    return f2


@jax.jit
def proj_nonneg(f0):
    f2 = f0 * (f0 >= 0)
    return f2


@jax.jit
def step_diffmap_supp(curr_x, exp_data, support_mask):
    projB = proj_supp(curr_x, support_mask)
    projA = proj_A(2 * projB - curr_x, exp_data)

    curr_x = curr_x + projA - projB
    recon = proj_supp(curr_x, support_mask)

    res = jnp.linalg.norm(projA - projB) / jnp.linalg.norm(projB)

    return curr_x, recon, res


@jax.jit
def step_diffmap_nonneg(curr_x, exp_data):
    projB = proj_nonneg(curr_x)
    projA = proj_A(2 * projB - curr_x, exp_data)

    curr_x = curr_x + projA - projB
    recon = proj_nonneg(curr_x)

    res = jnp.linalg.norm(projA - projB) / jnp.linalg.norm(projB)

    return curr_x, recon, res


def run_diffmap_algo(init_x, n_iter, exp_data, aux="supp", support_mask=None):
    residuals = np.zeros(n_iter)
    curr_x = np.copy(init_x)

    for i in range(n_iter):
        if aux == "supp":
            curr_x, recon, res = step_diffmap_supp(curr_x, exp_data, support_mask)
        elif aux == "nonneg":
            curr_x, recon, res = step_diffmap_nonneg(curr_x, exp_data)

        residuals[i] = res

    return curr_x, recon, residuals


def step_altproj_step_(curr_x, exp_data, support_mask):
    projBA = proj_supp(proj_A(curr_x, exp_data), support_mask)
    res = np.linalg.norm(projBA - curr_x) / np.linalg.norm(curr_x)

    return projBA, res


def run_altproj_algo(init_x, n_iter, exp_data, support_mask):
    residuals = np.zeros(n_iter)
    curr_x = np.copy(init_x)

    for i in range(n_iter):
        curr_x, res = step_altproj_step_(curr_x, exp_data, support_mask)
        residuals[i] = res

    return curr_x, residuals
