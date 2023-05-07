import numpy as np


def calc_p_neigh(image, p):
    supp = np.where(np.abs(image) > 1e-14)
    supp_img = np.zeros_like(image)

    for i in range(p + 1):
        supp_p = (supp[0] + i, supp[1] + i)
        supp_img[supp_p] = 1.0

    return supp_img


def gen_image(n_pixels, n_pad, k, n_disks, supp_neigh=0, seed=None):
    if seed is not None:
        np.random.seed(seed)

    image = np.zeros((n_pixels, n_pixels))

    grid_x, grid_y = np.meshgrid(
        np.linspace(-1, 1, n_pixels),
        np.linspace(-1, 1, n_pixels),
        indexing="ij",
    )

    for i in range(n_disks):
        center = np.random.uniform(low=-1, high=1, size=2) * 0.7
        radius = np.abs(np.random.uniform(low=-1, high=1) * 0.2)
        disk = np.sqrt((grid_x - center[0]) ** 2 + (grid_y - center[1]) ** 2) / radius

        image += (disk <= 1) * (1 - disk**2) ** k

    pad_width = int(np.ceil((n_pad - 1) / 2 * n_pixels))
    image = np.pad(image, pad_width=(pad_width, pad_width))
    image /= np.max(image)

    support_mask = calc_p_neigh(image, supp_neigh)

    return (image, support_mask)
