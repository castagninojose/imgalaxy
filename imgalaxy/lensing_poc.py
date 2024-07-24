import matplotlib.pyplot as plt
import numpy as np
import skimage as ski

from imgalaxy.cfg import LENSING_POC_GALAXIES

CHANNEL = 'rgb'
GALAXIES = {n: LENSING_POC_GALAXIES.format(n, CHANNEL) for n in range(10)}


def contours_segment(
    galaxy_ix: int,
    level_threshold=0.41,
    use_rescaled=False,
):
    galaxy_rgb = plt.imread(GALAXIES[galaxy_ix])[:, :, :3]  # drop channel 4 (empty)
    size = galaxy_rgb.shape[0:2]
    galaxy = ski.color.rgb2gray(galaxy_rgb)

    rescaled = ski.exposure.rescale_intensity(
        galaxy_rgb, in_range=(0, 1), out_range=(-10, 10)
    )
    if use_rescaled:
        galaxy = ski.color.rgb2gray(rescaled)

    contours = dict()
    for level in [0.37, 0.41, 0.47, 0.53, 0.57]:
        contours[level] = ski.measure.find_contours(galaxy, level)

    fig, ax = plt.subplots(2, 2, figsize=(11, 11))
    ax[0, 0].imshow(galaxy_rgb)
    ax[0, 1].imshow(rescaled)
    ax[1, 0].imshow(galaxy)

    for contour in contours.values():
        for c in contour:
            ax[1, 0].plot(c[:, 1], c[:, 0], linewidth=2)

    if contours[level_threshold]:
        mask = 1 * ski.draw.polygon2mask(size, contours[level_threshold][0])
        for c in range(1, len(contours[level_threshold])):
            mask = mask + (
                1 * ski.draw.polygon2mask(size, contours[level_threshold][c])
            )
            mask = (
                1 * ski.draw.polygon2mask(size, contours[level_threshold][c])
            ) + mask
        ax[1, 1].imshow(mask)
    else:  # contours list empty for this threshold (level), plot empty background
        mask = np.full(galaxy_rgb.shape, (0, 0, 139), dtype=np.uint8)
        ax[1, 1].imshow(mask)

    fig.show()


def longest_contour_mask(
    galaxy_ix: int, level=0.47, use_rescaled=False, use_second=False
):
    galaxy_rgb = plt.imread(GALAXIES[galaxy_ix])[:, :, :3]  # drop channel 4 (empty)
    size = galaxy_rgb.shape[0:2]
    galaxy = ski.color.rgb2gray(galaxy_rgb)

    rescaled = ski.exposure.rescale_intensity(
        galaxy_rgb, in_range=(0, 1), out_range=(-10, 10)
    )
    if use_rescaled:
        galaxy = ski.color.rgb2gray(rescaled)

    contours = ski.measure.find_contours(galaxy, level)

    fig, ax = plt.subplots(2, 2, figsize=(11, 11))
    ax[0, 0].imshow(galaxy_rgb)
    ax[0, 1].imshow(rescaled)
    ax[1, 0].imshow(galaxy)

    longest_c = np.argmax([c.size for c in contours])
    if use_second:
        del contours[longest_c]
        longest_c = np.argmax([c.size for c in contours])

    ax[1, 0].plot(contours[longest_c][:, 1], contours[longest_c][:, 0], linewidth=2)

    mask = ski.draw.polygon2mask(size, contours[longest_c])
    ax[1, 1].imshow(mask)

    fig.show()


if __name__ == '__main__':
    contours_segment(0, level_threshold=0.57, use_rescaled=False)
    contours_segment(2, level_threshold=0.37, use_rescaled=False)
    contours_segment(3, level_threshold=0.53, use_rescaled=False)
    contours_segment(5, use_rescaled=False)
    contours_segment(7)

    longest_contour_mask(2, level=0.37, use_rescaled=False, use_second=True)
    longest_contour_mask(2, level=0.37, use_rescaled=False, use_second=True)
    longest_contour_mask(0, level=0.51, use_rescaled=True)
    longest_contour_mask(5, level=0.51, use_second=True)
    longest_contour_mask(7, level=0.51, use_rescaled=True, use_second=True)
