import numpy as np
import json
from register_to_reference import register_to_reference
from compute_residual import compute_residual


class PhaseRetSolver:
    def __init__(self):
        self.proj_B_ = False
        self.init_proj_A_ = False
        self.config_init_ = False
        return

    def check_inits_(self):
        assert (
            self.init_proj_B_
        ), "Please initialize Projector to space B with set_proj_B()"
        assert (
            self.init_proj_A_
        ), "Please initialize Projector to space A with set_proj_A()"
        assert (
            self.config_init_
        ), "Please initialize your config with either set_config_from_file() or set_config_from_dict()"
        return

    def set_config_from_dict(self, config):
        self.config_ = config
        self.config_init_ = True
        return

    def set_config_from_file(self, config_fname):
        self.config_ = json.load(open(config_fname))
        self.config_init_ = True
        return

    def set_proj_B(self, proj_B, args):
        self.proj_B_ = proj_B
        self.proj_B_args_ = args
        self.init_proj_B_ = True
        return

    def set_proj_A(self, proj_A, args):
        self.proj_A_ = proj_A
        self.proj_A_args_ = args
        self.init_proj_A_ = True
        return

    def step_diffmap_algo_(self, curr_x):
        projB = self.proj_B_(curr_x, *self.proj_B_args_)
        projA = self.proj_A_(2 * projB - curr_x, *self.proj_A_args_)

        curr_x = curr_x + projA - projB
        recon = self.proj_B_(curr_x, *self.proj_B_args_)

        res = np.linalg.norm(projA - projB) / np.linalg.norm(projB)

        return curr_x, recon, res

    def run_diffmap_algo(self, init_x, ref_x, save_all=False):
        residuals = np.zeros(self.config_["n_iter"])
        errors = np.zeros(self.config_["n_iter"])
        curr_x = np.copy(init_x)

        if save_all:
            recon_x = np.zeros((self.config_["n_iter"] + 1, *init_x.shape))
            recon_x[0] = init_x

        for i in range(self.config_["n_iter"]):
            curr_x, recon, res = self.step_diffmap_algo_(curr_x)
            residuals[i] = res
            errors[i] = np.linalg.norm(recon - ref_x)

            if save_all:
                recon_x[i + 1] = curr_x

            max_it = i
            if errors[i] < self.config_["tol"]:
                break

        if save_all:
            return (
                recon_x[: max_it + 1, :],
                residuals[: max_it + 1],
                errors[: max_it + 1],
            )
        else:
            return recon, residuals[: max_it + 1], errors[: max_it + 1]

    def run_altproj_algo(self, init_x, ref_x, save_all=False):
        residuals = np.zeros(self.config_["n_iter"])
        errors = np.zeros(self.config_["n_iter"])
        curr_x = np.copy(init_x)

        if save_all:
            recon_x = np.zeros((self.config_["n_iter"] + 1, *init_x.shape))
            recon_x[0] = init_x

        for i in range(self.config_["n_iter"]):
            projBA = self.proj_B_(
                self.proj_A_(curr_x, *self.proj_A_args_), *self.proj_B_args_
            )
            residuals[i] = np.linalg.norm(projBA - curr_x) / np.linalg.norm(curr_x)
            curr_x = projBA
            errors[i] = np.linalg.norm(curr_x - ref_x)

            if save_all:
                recon_x[i + 1] = curr_x

            max_it = i
            if errors[i] < self.config_["tol"]:
                break

        if save_all:
            return (
                recon_x[: max_it + 1, :],
                residuals[: max_it + 1],
                errors[: max_it + 1],
            )

        else:
            return residuals[: max_it + 1], errors[: max_it + 1]
