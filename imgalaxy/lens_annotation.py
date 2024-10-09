"""Annotate simulated strong gravitational lenses images."""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import plotly.express as px
import streamlit as st
from astropy import convolution
from numpy.typing import NDArray
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import (
    SegmentationImage,
    SourceFinder,
    make_2dgaussian_kernel,
)

from imgalaxy.cfg import DES_DATA, DES_NO_SOURCE_DATA, LENSING_MASKS_DIR


def has_center(segmentation: NDArray, label: int) -> bool:
    """Boolean indicating if the center belongs to `label` in `segmentation` map."""
    mask: NDArray = segmentation == label
    centroids: List[Tuple] = [(31, 32), (32, 31), (32, 32), (33, 32), (32, 33)]
    for center in centroids:
        if mask[center]:
            return True

    return False


def remove_center_label(segmentation_map: SegmentationImage) -> NDArray:
    """Remove center segment (ie) the one that `has_center()`."""
    mask: NDArray = segmentation_map._data.copy()  # pylint: disable=protected-access
    for label in segmentation_map.labels:
        if has_center(mask, label):
            mask[mask == label] = 0
    return 1 * (mask > 0)


def keep_only_center(segmentation_map: SegmentationImage) -> NDArray:
    """Keep center segment (ie) the one that `has_center()`."""
    mask = segmentation_map._data.copy()  # pylint: disable=protected-access
    for label in segmentation_map.labels:
        if has_center(mask, label):
            mask[mask == label] = 1
        else:
            mask[mask == label] = 0
    return mask


def segment_background(
    data: NDArray, thresh: float = 1.5, npixels: int = 1, exclude_pct=12.5
):
    """Build segmentation map using photutils's Background2D and SourceFinder."""
    bkg = Background2D(
        data,
        (64, 64),
        filter_size=7,
        bkg_estimator=MedianBackground(),
        exclude_percentile=exclude_pct,
    )
    data -= bkg.background

    threshold = thresh * bkg.background_rms
    kernel = make_2dgaussian_kernel(3.0, size=5)
    convolved_data = convolution.convolve(data, kernel)
    finder = SourceFinder(
        npixels=npixels,
        connectivity=4,
        progress_bar=False,
        nlevels=32,
        mode='sinh',
        contrast=10 ** (-6),
    )
    segment_map = finder(convolved_data, threshold)

    return segment_map


def find_similar_masks(galaxy_ix):
    """
    Find all similar image and return the index of a randomly selected one.
    """
    index: int = 0
    image: NDArray = DES_NO_SOURCE_DATA[galaxy_ix]
    copies: list = []
    for i in range(4000):
        previous_image: NDArray = DES_NO_SOURCE_DATA[i]
        if np.allclose(previous_image, image, atol=0.5):
            copies.append(i)

    if copies:
        index = np.random.choice(copies)
    return index


def background_masks():
    """Streamlit dashboard app."""

    st.set_page_config(page_title="Lensing Masks", layout="wide")

    ix = st.number_input("Enter galaxy number", min_value=0, max_value=9999)
    background_image = DES_NO_SOURCE_DATA[ix].transpose(1, 2, 0)
    lens_mask_fp: Path = LENSING_MASKS_DIR / f"{ix}_lens_mask.npy"
    back_mask_fp: Path = LENSING_MASKS_DIR / f"{ix}_background_mask.npy"

    edit_masks = st.container()
    with edit_masks:
        lens_col, galaxy_col, background_col = st.columns(3)
        with lens_col:
            threshold = st.slider(
                "Select surface level",
                min_value=-1.0,
                max_value=75.0,
                value=14.14,
                key='lens_th',
            )
            segmentation_map = segment_background(
                background_image.sum(axis=2), threshold, exclude_pct=19.0
            )
            lens_mask = keep_only_center(segmentation_map)
            st.plotly_chart(px.imshow(lens_mask), theme=None)

        with background_col:
            threshold = st.slider(
                "Select surface level",
                min_value=-1.0,
                max_value=75.0,
                value=1.9,
                key='back_th',
            )
            segmentation_map = segment_background(
                background_image.sum(axis=2), threshold, exclude_pct=19.0
            )
            background_mask = remove_center_label(segmentation_map)
            st.plotly_chart(px.imshow(background_mask), theme=None)

        with galaxy_col:
            _, save_button_col, _ = st.columns([2, 3, 1])
            with save_button_col:
                if st.button("Save these masks", key='save_background'):
                    np.save(back_mask_fp, background_mask)
                    np.save(lens_mask_fp, lens_mask)
                    st.write(f"{back_mask_fp.stem} & {lens_mask_fp.stem} saved.")
                else:
                    st.write("")
                    st.write("")
            st.plotly_chart(px.imshow(background_image.sum(axis=2)), theme=None)

    mask_ix = find_similar_masks(ix)
    _, msg_col, _ = st.columns([4.5, 5, 2])
    with msg_col:
        st.write(r":point_down: $\textsf{\LARGE Similar image found}$ :point_down:")

    saved_lens_fp: Path = LENSING_MASKS_DIR / f"{mask_ix}_lens_mask.npy"
    saved_back_fp: Path = LENSING_MASKS_DIR / f"{mask_ix}_background_mask.npy"
    saved_back_mask = 1 * (np.load(saved_back_fp) > 0)  # avoid multi-labels
    saved_lens_mask = 1 * (np.load(saved_lens_fp) > 0)  # avoid multi-labels

    saved_masks_container = st.container()
    with saved_masks_container:
        saved_col1, saved_col2, saved_col3 = st.columns(3)
        with saved_col1:
            st.plotly_chart(px.imshow(saved_lens_mask), theme=None)
        with saved_col2:
            st.plotly_chart(px.imshow(background_image.sum(axis=2)), theme=None)
        with saved_col3:
            st.plotly_chart(px.imshow(saved_back_mask), theme=None)

    _, use_saved_button, _ = st.columns([5, 5, 2])
    with use_saved_button:
        if st.button(f"Reuse saved masks {mask_ix}", key='use_saved'):
            np.save(lens_mask_fp, saved_lens_mask)
            np.save(back_mask_fp, saved_back_mask)
            st.write(f"Acá no entramo nunca. Bueno... a veces si {mask_ix}")


def segment_source(
    data: NDArray,
    thresh1: float = 1.4,
    thresh2: float = 1.55,
    npixels: int = 1,
    exclude_pct=39.5,
):
    """Build segmentation map using photutils's Background2D and SourceFinder."""
    bkg = Background2D(
        data,
        (64, 64),
        filter_size=7,
        bkg_estimator=MedianBackground(),
        exclude_percentile=exclude_pct,
    )
    _data = data.copy()
    _data -= bkg.background
    threshold = thresh1 * bkg.background_rms
    kernel = make_2dgaussian_kernel(3.0, size=7)
    convolved_data = convolution.convolve(_data, kernel)
    finder = SourceFinder(
        npixels=npixels,
        connectivity=4,
        progress_bar=False,
        nlevels=32,
        mode='sinh',
        contrast=10 ** (-6),
        deblend=False,
    )
    source_map = finder(convolved_data, threshold)

    new_bkg = Background2D(
        data,
        (64, 64),
        filter_size=7,
        bkg_estimator=MedianBackground(),
        exclude_percentile=exclude_pct,
        mask=(source_map._data > 0),  # pylint: disable=protected-access
    )

    __data = data.copy()
    __data -= new_bkg.background

    threshold = thresh2 * new_bkg.background_rms
    kernel = make_2dgaussian_kernel(3.0, size=3)
    convolved_data = convolution.convolve(__data, kernel)
    finder = SourceFinder(
        npixels=npixels,
        connectivity=4,
        progress_bar=False,
        nlevels=32,
        mode='sinh',
        contrast=10 ** (-6),
        deblend=False,
    )
    mask = finder(convolved_data, threshold)

    return mask


def sources_masks():
    """Streamlit dashboard app."""

    st.set_page_config(page_title="Lensing Sources", layout="wide")

    ix = st.number_input("Enter galaxy number", min_value=0, max_value=9999)
    background_image = DES_NO_SOURCE_DATA[ix].transpose(1, 2, 0)
    background_and_source = DES_DATA[ix].transpose(1, 2, 0)
    source_image = background_and_source - background_image
    _source_img = source_image.copy()
    # _source_img[source_image < 19] = 0

    source_mask_fp: Path = LENSING_MASKS_DIR / f"{ix}_source_mask.npy"

    edit_mask = st.container()
    with edit_mask:
        galaxy_col, mask_col = st.columns(2)
        with mask_col:
            threshold1 = st.slider(
                "Select surface level",
                min_value=1.0,
                max_value=29.0,
                value=1.414,
                key='th1',
            )
            threshold2 = st.slider(
                "Select surface level",
                min_value=1.0,
                max_value=720.0,
                value=11.14,
                key='th2',
            )
            segmentation_map = segment_source(
                _source_img[:, :, 1:4].sum(axis=2),
                threshold1,
                threshold2,
                exclude_pct=39.0,
            )
            # mask = segmentation_map._data
            st.plotly_chart(
                px.imshow(segmentation_map, height=737, width=737),
                theme=None,
                use_container_width=True,
            )

        with galaxy_col:
            _, save_button_col, _ = st.columns([2, 3, 1])
            with save_button_col:
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                if st.button("Save this mask", key='save_source'):
                    np.save(source_mask_fp, segmentation_map)
                    st.write(f"{source_mask_fp.stem} saved.")
                else:
                    st.write("")
                    st.write("")
            # fig, ax = plt.subplots()
            # ax.imshow(ski.color.label2rgb(segmentation_map._data, _source_img[:, :, 1:4].sum(axis=2)))
            # st.pyplot(fig)

            st.plotly_chart(
                px.imshow(
                    source_image.sum(axis=2), height=737, width=737, binary_string=True
                ),
                use_container_width=True,
                theme=None,
            )


if __name__ == "__main__":
    # background_masks()
    sources_masks()
