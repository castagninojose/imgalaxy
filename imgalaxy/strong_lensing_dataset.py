from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm

from imgalaxy.cfg import DES_DATA, LENSING_MASKS_DIR


class LensingDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version('1.0.0')

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description="Lensing dataset with one image and background, lens and source masks.",
            features=tfds.features.FeaturesDict(
                {
                    'galaxy_id': tfds.features.Text(),
                    'image': tfds.features.Tensor(shape=(64, 64, 5), dtype=np.float32),
                    'background': tfds.features.Image(shape=(64, 64, 1)),
                    'lens': tfds.features.Image(shape=(64, 64, 1)),
                    'source': tfds.features.Image(shape=(64, 64, 1)),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.extract('/hdd/lensing-dataset/')
        return {
            'train': self._generate_examples(data_dir),
        }

    def _generate_examples(self, path):
        # Replace with loading your real data
        images = np.load(path / 'images.npy')

        backgrounds = np.load(path / 'background_masks.npy')
        backgrounds = np.expand_dims(backgrounds, axis=-1)

        lenses = np.load(path / 'lens_masks.npy')
        lenses = np.expand_dims(lenses, axis=-1)

        sources = np.load(path / 'source_masks.npy')
        sources = np.expand_dims(sources, axis=-1)

        for i in range(len(images)):
            yield i, {
                'galaxy_id': f"{i}",
                'image': images[i],
                'background': backgrounds[i],
                'lens': lenses[i],
                'source': sources[i],
            }


def stack_images_and_masks(N: int = 9999, save_path: str = '/hdd/lensing-dataset/'):
    # Preallocate arrays
    images = np.zeros((N,) + (64, 64, 5), dtype=np.float32)
    lens_masks = np.zeros((N,) + (64, 64), dtype=np.uint8)
    background_masks = np.zeros((N,) + (64, 64), dtype=np.uint8)
    source_masks = np.zeros((N,) + (64, 64), dtype=np.uint8)

    # Stack arrays by index number
    for idx in tqdm(range(N), desc="Stacking files"):
        images[idx] = DES_DATA[idx].transpose(1, 2, 0).astype(np.float32)
        lens_masks[idx] = (np.load(LENSING_MASKS_DIR / f'{idx}_lens_mask.npy') > 0).astype(np.uint8)
        background_masks[idx] = (
            np.load(LENSING_MASKS_DIR / f'{idx}_background_mask.npy') > 0
        ).astype(np.uint8)
        source_masks[idx] = (np.load(LENSING_MASKS_DIR / f'{idx}_source_mask.npy') > 0).astype(
            np.uint8
        )

    output_dir = Path(save_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Saving Galaxies...")
    np.save(output_dir / 'images.npy', images)
    print("Saving Lenses...")
    np.save(output_dir / 'lens_masks.npy', lens_masks)
    print("Saving Backgrounds...")
    np.save(output_dir / 'background_masks.npy', background_masks)
    print("Saving Sources...")
    np.save(output_dir / 'source_masks.npy', source_masks)


if __name__ == '__main__':
    # stack_images_and_masks(999)
    builder = LensingDataset(data_dir='/hdd/lensing-dataset/')
    builder.download_and_prepare()
