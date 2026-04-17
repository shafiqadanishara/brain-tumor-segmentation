import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter, affine_transform


def augment(image, seg):
    """
    Full augmentation pipeline applied identically to image and mask.

    Args:
        image : (C, D, H, W) float numpy array  — all 4 modality channels
        seg   : (3, D, H, W) float numpy array  — binary WT, TC, ET channels

    Returns:
        image, seg — same shapes, augmented in-memory (originals on disk untouched)

    Steps:
        1. Random axis flips               — prob 0.5 per axis
        2. Rotation + per-axis scaling     — prob 0.3, scale range (0.65, 1.6)
        3. Elastic deformation             — prob 0.3
        4. Additive brightness             — prob 0.3  (image only)
        5. Gamma augmentation (aggressive) — prob 0.5  (image only)
    """

    # ---- 1. Random flips ----
    for axis in range(1, 4):   # D, H, W axes — skip channel dim
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=axis).copy()
            seg   = np.flip(seg,   axis=axis).copy()

    # ---- 2. Rotation + per-axis scaling (prob 0.3) ----
    if np.random.rand() < 0.3:
        angle_deg = np.random.uniform(-15, 15)
        angle_rad = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Independent scale factor per axis — range (0.65, 1.6)
        sx = np.random.uniform(0.65, 1.6)
        sy = np.random.uniform(0.65, 1.6)
        sz = np.random.uniform(0.65, 1.6)

        rot_scale = np.array([
            [ cos_a / sy,  sin_a / sz, 0        ],
            [-sin_a / sy,  cos_a / sz, 0        ],
            [          0,           0, 1.0 / sx ],
        ])

        C, D, H, W = image.shape
        center = np.array([D / 2, H / 2, W / 2])
        offset = center - rot_scale @ center

        aug_image = np.zeros_like(image)
        for c in range(C):
            aug_image[c] = affine_transform(
                image[c], rot_scale, offset=offset, order=1, mode='nearest'
            )

        aug_seg = np.zeros_like(seg)
        for c in range(seg.shape[0]):
            aug_seg[c] = affine_transform(
                seg[c], rot_scale, offset=offset, order=0, mode='nearest'
            )

        image = aug_image
        seg   = aug_seg

    # ---- 3. Elastic deformation (prob 0.3) ----
    if np.random.rand() < 0.3:
        C, D, H, W = image.shape
        sigma = 6.0    # smoothness of deformation field
        alpha = 80.0   # strength of deformation

        dz = gaussian_filter(np.random.randn(D, H, W), sigma) * alpha
        dy = gaussian_filter(np.random.randn(D, H, W), sigma) * alpha
        dx = gaussian_filter(np.random.randn(D, H, W), sigma) * alpha

        z, y, x = np.meshgrid(
            np.arange(D), np.arange(H), np.arange(W), indexing='ij'
        )
        coords = [
            np.clip(z + dz, 0, D - 1),
            np.clip(y + dy, 0, H - 1),
            np.clip(x + dx, 0, W - 1),
        ]

        aug_image = np.zeros_like(image)
        for c in range(C):
            aug_image[c] = map_coordinates(image[c], coords, order=1, mode='nearest')

        aug_seg = np.zeros_like(seg)
        for c in range(seg.shape[0]):
            aug_seg[c] = map_coordinates(seg[c], coords, order=0, mode='nearest')

        image = aug_image
        seg   = aug_seg

    # ---- 4. Additive brightness (prob 0.3, image only) ----
    if np.random.rand() < 0.3:
        offset = np.random.uniform(-0.1, 0.1)
        image  = image + offset

    # ---- 5. Gamma augmentation — aggressive range (prob 0.5, image only) ----
    if np.random.rand() < 0.5:
        gamma     = np.random.uniform(0.5, 2.0)
        image_min = image.min()
        image_max = image.max()
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min + 1e-8)
            image = np.power(np.clip(image, 0, 1), gamma)
            image = image * (image_max - image_min) + image_min

    return image, seg