from astropy import convolution
from numpy.typing import NDArray
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import SourceFinder, make_2dgaussian_kernel

from imgalaxy.cfg import LENSING_POC_GALAXIES

BANDS = {0: 'G', 1: 'R', 2: 'I', 3: 'Z', 4: 'Y'}
CHANNEL = 'rgb'
GALAXIES = {n: LENSING_POC_GALAXIES.format(n, CHANNEL) for n in range(10)}
THRESHOLD_TYPES = ['yen', 'triangle', 'li', 'otsu', 'min', 'iso', 'sigma']


def segment_background(data: NDArray, thresh: float = 1.5, npixels: int = 10):
    bkg_estimator = MedianBackground()
    bkg = Background2D(
        data,
        (64, 64),
        filter_size=(7, 7),
        bkg_estimator=bkg_estimator,
        exclude_percentile=13.0,
    )
    data -= bkg.background

    threshold = thresh * bkg.background_rms
    kernel = make_2dgaussian_kernel(3.0, size=5)
    convolved_data = convolution.convolve(data, kernel)
    finder = SourceFinder(
        npixels=npixels, progress_bar=False, connectivity=8, nlevels=64
    )
    segment_map = finder(convolved_data, threshold)

    return segment_map
