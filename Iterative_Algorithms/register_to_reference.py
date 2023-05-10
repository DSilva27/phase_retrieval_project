import numpy as np


def fracshift(img_align, dx, dy):
    img_shift = img_align.copy()
    img_hat = np.fft.fftn(img_shift)
    N1, N2 = img_shift.shape

    M1a = np.floor((N1 - 1) / 2)
    M1b = np.floor(N1 / 2)
    kxrange = np.array([np.arange(0, M1a + 1), np.arange(-M1b, 0)]).flatten()

    M2a = np.floor((N2 - 1) / 2)
    M2b = np.floor(N2 / 2)
    kyrange = np.array([np.arange(0, M2a + 1), np.arange(-M2b, 0)]).flatten()

    kx, ky = np.meshgrid(kxrange, kyrange, indexing="ij")
    theta = (kx * dx / N1 + ky * dy / N2) * 2 * np.pi
    img_hat = img_hat * np.exp(1j * theta)
    img_shift = np.real(np.fft.ifftn(img_hat))

    return img_shift


def register_to_reference(image, image_ref):
    img_align = image.copy()
    N1 = image_ref.shape[0]
    N2 = image_ref.shape[1]

    M1 = int(np.ceil((N1 + 1) / 2))
    M2 = int(np.ceil((N2 + 1) / 2))

    tmp1 = np.abs(
        np.fft.ifftshift(
            np.fft.ifftn(np.fft.fftn(img_align) * np.conj(np.fft.fftn(image_ref)))
        )
    )
    tmp2 = np.abs(
        np.fft.ifftshift(
            np.fft.ifftn(
                np.conj(np.fft.fftn(img_align)) * np.conj(np.fft.fftn(image_ref))
            )
        )
    )

    ii1 = np.argmax(tmp1)
    max1 = tmp1.flatten()[ii1]

    ii2 = np.argmax(tmp2)
    max2 = tmp2.flatten()[ii2]

    if max1 > max2:
        ii = ii1
    else:
        ii = ii2
        img_align = np.fft.ifftn(np.conj(np.fft.fftn(img_align)))

    i1, i2 = np.unravel_index(ii, img_align.shape)

    img_align = np.roll(img_align, -(i1 - M1), 0)
    img_align = np.roll(img_align, -(i2 - M2), 1)

    if np.sum(img_align * image_ref) < 0:
        img_align = -img_align

    dxrange = np.arange(-1.0, 1.2, 0.2)
    dyrange = np.arange(-1.0, 1.2, 0.2)

    best_resid = np.inf
    best_image = img_align

    for dy in dyrange:
        for dx in dxrange:
            Atry = fracshift(img_align, dx, dy)
            resid = np.linalg.norm(Atry - image_ref)

            if resid < best_resid:
                best_image = Atry
                best_resid = resid

    return best_image
