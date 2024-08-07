"""Configs."""
from pathlib import Path

import importlib_resources
from decouple import AutoConfig

PKG_PATH = importlib_resources.files("imgalaxy")
REPO_ROOT = PKG_PATH.parent

config = AutoConfig(search_path=REPO_ROOT)

RESOURCES_DIR = PKG_PATH / "resources"
DATA_DIR = RESOURCES_DIR / "data"
METADATA_DIR = RESOURCES_DIR / "metadata"
LOGS_DIR = RESOURCES_DIR / "logs"
MODELS_DIR = RESOURCES_DIR / "models"

Path(RESOURCES_DIR).mkdir(exist_ok=True, parents=True)
Path(DATA_DIR).mkdir(exist_ok=True, parents=True)
Path(METADATA_DIR).mkdir(exist_ok=True, parents=True)
Path(LOGS_DIR).mkdir(exist_ok=True, parents=True)
Path(MODELS_DIR).mkdir(exist_ok=True, parents=True)

BASE_URL = "https://data.sdss.org/sas/dr17/manga/morphology/galaxyzoo3d/v4_0_0/"
SHA1SUM_URL = "manga_morphology_galaxyzoo3d_v4_0_0.sha1sum"
METADATA_FILENAMES = [
    "gz3d_checksums.sha1sum",
    "gz3d_metadata.fits",
    "gz3d_galaxy_centers.fits",
    "gz3d_foreground_stars.fits",
]
CHECKSUMS_FILEPATH = METADATA_DIR / METADATA_FILENAMES[0]
GALAXIES_CACHE = [x for x in DATA_DIR.glob('*') if x.suffix == '.gz']  # downloaded

LENSING_POC_GALAXIES = str(DATA_DIR / "lensing_poc/galaxy_{}_{}.png")
