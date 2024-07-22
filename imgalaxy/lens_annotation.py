"""Annotate simulated strong gravitational lenses images."""
from datetime import datetime
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
    return mask


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


def _main():
    """Streamlit dashboard app."""

    st.set_page_config(page_title="Lensing Masks", layout="wide")

    explore = st.container()
    with explore:
        galaxy_ix = st.number_input("Enter galaxy number", min_value=0, max_value=9999)
        background_image = DES_NO_SOURCE_DATA[galaxy_ix].transpose(1, 2, 0).mean(axis=2)
        _source_image = DES_DATA[galaxy_ix].transpose(1, 2, 0).mean(axis=2)
        source_image = _source_image - background_image

        lens_mask_fp: Path = LENSING_MASKS_DIR / f"{galaxy_ix}_lens_mask.npy"
        arcs_mask_fp: Path = LENSING_MASKS_DIR / f"{galaxy_ix}_arcs_mask.npy"
        back_mask_fp: Path = LENSING_MASKS_DIR / f"{galaxy_ix}_background_mask.npy"

        source_title, background_title = st.columns([2, 3])
        with source_title:
            _, title, _ = st.columns([1, 3, 1])
            with title:
                st.write(r"$\textsf{\large Source}$")

        with background_title:
            _, title, _ = st.columns([1, 3, 1])
            with title:
                st.write(r"$\textsf{\large Background}$")

        arcs_col, source_col, lens_col, galaxy_col, background_col = st.columns(5)

        with arcs_col:
            threshold = st.slider(
                "Select surface level",
                min_value=-1.0,
                max_value=80.0,
                value=4.9,
                key='arcs_th',
            )
            arcs_mask = source_image > threshold
            st.plotly_chart(
                px.imshow(arcs_mask).update_layout(coloraxis_showscale=False)
            )

        with source_col:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.plotly_chart(
                px.imshow(source_image).update_layout(coloraxis_showscale=False)
            )

        with galaxy_col:
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.write("")
            st.plotly_chart(
                px.imshow(background_image).update_layout(coloraxis_showscale=False)
            )

        with lens_col:
            threshold = st.slider(
                "Select surface level",
                min_value=-1.0,
                max_value=50.0,
                value=14.14,
                key='lens_th',
            )

            segmentation_map = segment_background(background_image, threshold)
            lens_mask = keep_only_center(segmentation_map)

            st.plotly_chart(
                px.imshow(lens_mask).update_layout(coloraxis_showscale=False)
            )
            last_modified = datetime.fromtimestamp(lens_mask_fp.stat().st_mtime)
            st.write(f"Last modified: {last_modified.strftime('%d/%-m, %H:%M')}")

        with background_col:
            threshold = st.slider(
                "Select surface level",
                min_value=-1.0,
                max_value=50.0,
                value=1.9,
                key='back_th',
            )

            segmentation_map = segment_background(background_image, threshold)
            background_mask = remove_center_label(segmentation_map)

            st.plotly_chart(
                px.imshow(background_mask).update_layout(coloraxis_showscale=False)
            )
            last_modified = datetime.fromtimestamp(lens_mask_fp.stat().st_mtime)
            st.write(f"Last modified: {last_modified.strftime('%d/%-m, %H:%M')}")

        _, save_button_col, _ = st.columns([5, 5, 2])
        with save_button_col:
            if st.button("Save these masks", key='save_background'):
                np.save(back_mask_fp, background_mask)
                np.save(lens_mask_fp, lens_mask)
                np.save(arcs_mask_fp, arcs_mask)


def main():
    """Streamlit dashboard app."""

    st.set_page_config(page_title="Lensing Masks", layout="wide")

    explore = st.container()
    with explore:
        galaxy_ix = st.number_input("Enter galaxy number", min_value=0, max_value=9999)
        image = DES_NO_SOURCE_DATA[galaxy_ix].transpose(1, 2, 0).mean(axis=2)

        lens_mask_fp: Path = LENSING_MASKS_DIR / f"{galaxy_ix}_lens_mask.npy"
        back_mask_fp: Path = LENSING_MASKS_DIR / f"{galaxy_ix}_background_mask.npy"

        lens_col, galaxy_col, background_col = st.columns(3)
        with galaxy_col:
            _, title_col, _ = st.columns([2.3, 6, 1])
            with title_col:
                st.write(r"$\textsf{\Large Original image}$")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")
                st.write("")

            st.plotly_chart(px.imshow(image))

        with lens_col:
            _, title_col, _ = st.columns([2.5, 6, 1])
            with title_col:
                st.write(r"$\textsf{\large Lens mask}$")

            (
                threshold_col,
                save_button_col,
            ) = st.columns([5, 1])
            with threshold_col:
                threshold = st.slider(
                    "Select surface level",
                    min_value=-1.0,
                    max_value=50.0,
                    value=14.14,
                    key='lens_th',
                )

            segmentation_map = segment_background(image, threshold)
            lens_mask = keep_only_center(segmentation_map)

            st.plotly_chart(px.imshow(lens_mask))
            last_modified = datetime.fromtimestamp(lens_mask_fp.stat().st_mtime)
            st.write(f"Last modified: {last_modified.strftime('%d/%-m, %H:%M')}")

        with background_col:
            _, title_col, _ = st.columns([2.5, 6, 1])
            with title_col:
                st.write(r"$\textsf{\large Background mask}$")
            threshold = st.slider(
                "Select surface level",
                min_value=-1.0,
                max_value=50.0,
                value=1.9,
                key='back_th',
            )

            segmentation_map = segment_background(image, threshold)
            background_mask = remove_center_label(segmentation_map)

            st.plotly_chart(px.imshow(background_mask))
            last_modified = datetime.fromtimestamp(lens_mask_fp.stat().st_mtime)
            st.write(f"Last modified: {last_modified.strftime('%d/%-m, %H:%M')}")

        _, save_button_col, _ = st.columns([5, 5, 2])
        with save_button_col:
            if st.button("Save these masks", key='save_background'):
                np.save(back_mask_fp, background_mask)
                np.save(lens_mask_fp, lens_mask)


if __name__ == "__main__":
    # main()
    _main()
