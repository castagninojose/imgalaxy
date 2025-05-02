# pylint: disable=no-member
from typing import Callable, Union

import tensorflow as tf

from imgalaxy.constants import THRESHOLD


def binarize_mask(mask, threshold: int):
    return tf.where(mask < threshold, tf.zeros_like(mask), tf.ones_like(mask))


class BaseSegmentationPipeline:
    """
    A data pipeline class for preprocessing datasets for semantic segmentation models.

    Attributes
    ----------
    size : int
        The target size for resizing images and masks.
    mask_key : str
        The key to access the mask in the dataset examples.
    preprocess_input : callable, optional
        A function to preprocess the input images (intended to be use with preprocess_input functions
        from keras.applications).
    binary_threshold : int
        Votes threshold used to binarize the mask.
    sparse : bool
        Whether to output a sparse or a one-hot encoded mask.
    clip_votes_max : int
        The maximum value to clip the mask votes.
    batch_size : int
        The batch size for the dataset.
    shuffle_buffer_size : int
        The buffer size for shuffling the dataset.
    cache : bool
        Whether to cache the dataset.
    prefetch : bool, default=False.
        Whether to prefetch the dataset.

    Methods
    -------
    load_data(example):
        Loads and preprocesses the image and mask from a dataset example.
    resize(image, mask):
        Resizes the image and mask to the target size.
    __call__(ds):
        Applies the data pipeline to the given dataset.

    """

    def __init__(
        self,
        size: int,
        preprocess_input: Union[Callable, None] = None,
        binary_threshold: bool = False,
        sparse: bool = True,
        clip_votes_max: int = 6,
        batch_size: int = 32,
        shuffle_buffer_size: int = 1,
        cache: bool = True,
        prefetch: bool = True,
    ) -> None:
        self.size = size
        self.preprocess_input = preprocess_input
        self.binary_threshold = binary_threshold
        self.sparse = sparse
        self.clip_votes_max = clip_votes_max
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.cache = cache
        self.prefetch = prefetch

    def resize(self, image, mask):
        image = tf.image.resize(image, (self.size, self.size))
        mask = tf.image.resize(
            mask, (self.size, self.size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )
        return image, mask

    def __call__(self, ds):
        ds = ds.map(self.load_data, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(self.resize, num_parallel_calls=tf.data.AUTOTUNE)
        if self.cache:
            ds = ds.cache()
        if self.shuffle_buffer_size > 0:
            ds = ds.shuffle(buffer_size=self.shuffle_buffer_size)
        ds = ds.batch(self.batch_size)
        if self.prefetch:
            ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return ds


class GZ3DPipeline(BaseSegmentationPipeline):
    """
    Pipeline for training with Galaxy Zoo 3D dataset. Inherits from `BaseSegmentationPipeline`
    TODO May 2025: Add docstring
    """

    def load_data(self, example):
        image = example["image"]

        if self.preprocess_input:
            image = self.preprocess_input(image)
        else:
            image = tf.cast(image, tf.float16) / 255.0

        spiral_mask = example["spiral_mask"]
        bar_mask = example["bar_mask"]

        spiral_mask = tf.minimum(spiral_mask, self.clip_votes_max)
        bar_mask = tf.minimum(bar_mask, self.clip_votes_max)

        if self.binary_threshold:
            spiral_mask = binarize_mask(spiral_mask, THRESHOLD)
            bar_mask = binarize_mask(bar_mask, THRESHOLD)

        spiral_mask = tf.cast(spiral_mask, tf.int32)
        bar_mask = tf.cast(bar_mask, tf.int32)

        combined_mask = tf.zeros_like(bar_mask)
        combined_mask += tf.where(spiral_mask == 1, 1, 0)  # label spirals as 1
        combined_mask += tf.where(bar_mask == 1, 2, 0)  # label bars as 2
        # since these are added, pixels in both bars and spirals are labeled as 1 + 2 = 3.

        if not self.sparse:
            mask = tf.one_hot(tf.cast(combined_mask, tf.int32), depth=4)
            mask = tf.squeeze(mask, axis=2)
        else:
            mask = tf.cast(combined_mask, tf.int32)

        return image, mask


class LensingPipeline(BaseSegmentationPipeline):
    """Pipeline for strong gravitational lensing model. Inherits from `BaseSegmentationPipeline`"""

    def load_data(self, example):
        image = example["image"]

        if self.preprocess_input:
            image = self.preprocess_input(image)
        else:
            image = tf.cast(image, tf.float16) / 255.0

        source_mask = example["source"]
        lens_mask = example["lens"]
        background_mask = example["background"]

        source_mask = tf.cast(source_mask, tf.int32)
        lens_mask = tf.cast(lens_mask, tf.int32)
        background_mask = tf.cast(background_mask, tf.int32)

        combined_mask = tf.where(source_mask == 1, 1, 0)  # label source as 1
        combined_mask = tf.where(lens_mask == 1, 2, combined_mask)  # label lens as 2
        combined_mask = tf.where(background_mask == 1, 3, combined_mask)  # label background as 3

        if not self.sparse:
            mask = tf.one_hot(tf.cast(combined_mask, tf.int32), depth=4)
            mask = tf.squeeze(mask, axis=2)
        else:
            mask = tf.cast(combined_mask, tf.int32)

        return image, mask


class AugmentLayer(tf.keras.layers.Layer):
    """
    A custom Keras layer for applying augmentations to both images and masks. This layer ensures
    that the same augmentations are applied to both images and masks during training. The
    augmentations are specified as a list of functions

    Attributes
    ----------
    augmentations : list
        A list of augmentation functions or keras image augmentation layers to be applied to the
        images and masks.

    """

    def __init__(self, augmentations):
        super(AugmentLayer, self).__init__()
        self.augmentations = augmentations

    def call(self, images, masks, training=False):
        # Apply the same augmentations to both images and masks during training
        if training:
            img_channels = tf.shape(images)[-1]
            mask_channels = tf.shape(masks)[-1]
            float_masks = tf.cast(masks, tf.float16)
            images_masks = tf.concat([images, float_masks], axis=-1)

            for augmentation in self.augmentations:
                images_masks = augmentation(images_masks)

            images, masks = tf.split(images_masks, [img_channels, mask_channels], axis=-1)

        return images, masks


class AugmentedSegmentationModel(tf.keras.Model):
    """
    A custom Keras model that integrates data augmentation with a segmentation model. This model
    applies specified augmentations to both images and masks before passing them to the segmentation
    model during training.

    Attributes
    ----------
        augment_layer : AugmentLayer
            Layer that applies augmentations to images and masks.
        segmentation_model : tf.keras.Model
            The underlying segmentation model.

    Methods
    -------
        call(inputs, training=False):
            Forward pass of the model. Applies the segmentation model to the inputs.
        train_step(data):
            Custom training step that includes data augmentation and loss computation.

            Args
            ----
                data : tuple
                    A tuple containing images and masks.
            Returns
            -------
                dict
                    A dictionary containing the loss and other metrics.

    """

    def __init__(self, augmentations, segmentation_model):
        """
        Initializes the AugmentedSegmentationModel with the given augmentations and segmentation model.
        Args:
            augmentations (list): A list of augmentation functions or keras image augmentation layers to be applied to the images and masks.
            segmentation_model: The segmentation model to be used for image segmentation.
        """

        super(AugmentedSegmentationModel, self).__init__()
        self.augment_layer = AugmentLayer(augmentations)  # layer for images and masks
        self.segmentation_model = segmentation_model  # segmentation model

    def call(self, inputs, training=False):
        return self.segmentation_model(inputs, training=training)

    def train_step(self, data):
        images, masks = data
        with tf.GradientTape() as tape:
            images, masks = self.augment_layer(images, masks, training=True)
            predictions = self(images, training=True)
            loss = self.compute_loss(y=masks, y_pred=predictions)

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(masks, predictions)

        # Return a dictionary with loss and all metrics
        return {m.name: m.result() for m in self.metrics}
