"""Helper methods"""
# pylint: disable=no-member  # pylint keeps insisting that there's no tf.keras 🤷
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import colors
from numpy.typing import NDArray

import wandb

tf.config.run_functions_eagerly(True)


def log_predictions(ds_test, model, task: str = "galaxy_zoo3d", n: int = 3) -> None:
    """
    Samples up to n examples from ds_test, gets predictions from the model,
    and logs the original images, ground truth masks, and predicted masks to wandb.

    Args:
        ds_test (tf.data.Dataset): The test dataset.
        model (tf.keras.Model): The trained segmentation model.
        n (int): Number of samples to log.
    """

    for batch in ds_test.take(1):
        # Extract images and masks explicitly
        images = batch[0]
        true_masks = batch[1]
        break

    batch_size = tf.shape(images)[0]
    n = tf.minimum(n, batch_size).numpy()
    # indices = np.random.choice(batch_size, n, replace=False)  # use to subset samples from batch
    indices = range(batch_size)
    selected_images = tf.gather(images, indices)
    selected_true_masks = tf.gather(true_masks, indices)

    predictions = model.predict(selected_images)

    predicted_masks = np.argmax(predictions, axis=-1)

    # Convert true masks from one-hot encoding if necessary
    if selected_true_masks.shape[-1] > 1:
        selected_true_masks = np.argmax(selected_true_masks, axis=-1)

    cmap = colors.ListedColormap(['k', 'b', 'y', 'r'])
    for i in range(n):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        galaxy = selected_images[i].numpy()
        if task == 'lensing':
            galaxy = galaxy.mean(axis=-1)  # lensing images have 5 channels, need averaging to plot.
        axes[0].imshow(galaxy)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(predicted_masks[i], cmap=cmap)
        axes[1].set_title("Predicted Mask")
        axes[1].axis("off")

        axes[2].imshow(selected_true_masks[i], cmap=cmap)
        axes[2].set_title("Ground Truth Mask")
        axes[2].axis("off")

        wandb.log({"Prediction Sample": wandb.Image(fig)})

        plt.close(fig)  # Close figure to free memory


def log_training_examples(dataset, n: int = 5) -> None:
    """Log training examples to check if the masks were correctly generated."""
    for batch in dataset.take(1):
        # Extract images and masks explicitly
        images = batch[0]
        masks = batch[1]
        break

    batch_size: int = tf.shape(images)[0]
    size: int = tf.minimum(n, batch_size).numpy()
    indices: NDArray = np.random.choice(batch_size, size, replace=False)
    selected_images = tf.gather(images, indices)
    selected_masks = tf.gather(masks, indices)

    # Convert true masks from one-hot encoding if necessary
    if selected_masks.shape[-1] > 1:
        selected_masks = np.argmax(selected_masks, axis=-1)

    cmap = colors.ListedColormap(['k', 'b', 'y', 'r'])
    for i in range(n):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(selected_images[i])
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(selected_masks[i], cmap=cmap)
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")

        wandb.log({"Training Sample": wandb.Image(fig)})

        plt.close(fig)  # Close figure to free memory


class TimeCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        self.epochs = []
        self.timetaken = tf.timestamp()

    def on_epoch_end(self, epoch):
        self.times.append(tf.timestamp() - self.timetaken)
        self.epochs.append(epoch)

    def on_train_end(self):
        plt.xlabel('Epoch')
        plt.ylabel('Total time taken until an epoch in seconds')
        plt.plot(self.epochs, self.times, 'ro')
        for i in range(len(self.epochs)):
            j = self.times[i].numpy()
            if i == 0:
                plt.text(i, j, str(round(j, 3)))
            else:
                j_prev = self.times[i - 1].numpy()
                plt.text(i, j, str(round(j - j_prev, 3)))

        plt.savefig(datetime.now().strftime("%Y%m%d%H%M%S") + ".png")


def get_iou_configs(n_labels: int = 4, sparse: bool = True, compute_mean_iou: bool = False) -> list:
    """
    Generate a list of IoU metrics from tf.keras with the corresponding configurations for our pipeline.
    Will ignore class labeled as `0`, corresponding to the background.
    """
    metric_kwargs: Dict[str, Any] = {
        'num_classes': n_labels,
        'sparse_y_pred': False,
        'ignore_class': 0,
        'axis': -1,
    }
    metrics: list = []
    if compute_mean_iou:
        mean_kwargs: Dict[str, Any] = deepcopy(metric_kwargs)
        mean_kwargs['name'] = 'Mean_IoU'

        if not sparse:
            metrics.append(tf.keras.metrics.OneHotMeanIoU(**mean_kwargs))
        else:
            mean_kwargs['sparse_y_true'] = True
            metrics.append(tf.keras.metrics.MeanIoU(**mean_kwargs))

    for label in range(1, n_labels):
        metric_kwargs['target_class_ids'] = [label]
        metric_kwargs['name'] = f"IoU_{label}"

        if not sparse:
            metrics.append(tf.keras.metrics.OneHotIoU(**metric_kwargs))
        else:
            metric_kwargs['sparse_y_true'] = True
            metrics.append(tf.keras.metrics.IoU(**metric_kwargs))

    return metrics
