"""Helper methods"""

from datetime import datetime

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, jaccard_score

import wandb

tf.config.run_functions_eagerly(True)


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask


def evaluate_model(dataset, model, num=5):
    """Evaluate model after a run is completed."""
    # TODO: try tf.keras.Model.evaluate()
    if dataset:
        for image, mask in dataset:
            pred_mask = create_mask(model.predict(image))
            for ind in range(num):
                wandb.log(
                    {
                        "example": [
                            wandb.Image(image[ind]),
                            wandb.Image(mask[ind]),
                            wandb.Image(pred_mask[ind]),
                        ]
                    }
                )

                if np.amax(pred_mask[ind].numpy()) == 0:
                    print(2 * '\n')
                    continue
                else:
                    conf_matrix = confusion_matrix(
                        pred_mask[ind].numpy().reshape(-1),
                        mask[ind].numpy().reshape(-1),
                    )
                    jacc_score = jaccard_score(
                        pred_mask[ind].numpy().reshape(-1),
                        mask[ind].numpy().reshape(-1),
                    )
        return conf_matrix, jacc_score


def log_predictions(ds_test, model, n: int = 3) -> None:
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
    # indices = np.random.choice(batch_size, n, replace=False)
    indices = range(batch_size)
    selected_images = tf.gather(images, indices)
    selected_true_masks = tf.gather(true_masks, indices)

    predictions = model.predict(selected_images)

    predicted_masks = np.argmax(predictions, axis=-1)

    # Convert true masks from one-hot encoding if necessary
    if selected_true_masks.shape[-1] > 1:
        selected_true_masks = np.argmax(selected_true_masks, axis=-1)

    for i in range(n):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        galaxy = selected_images[i]  # convert back to RGB
        axes[0].imshow(galaxy)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(predicted_masks[i], cmap="viridis")
        axes[1].set_title("Predicted Mask")
        axes[1].axis("off")

        axes[2].imshow(selected_true_masks[i], cmap="viridis")
        axes[2].set_title("Ground Truth Mask")
        axes[2].axis("off")

        wandb.log({"Prediction Sample": wandb.Image(fig)})

        plt.close(fig)  # Close figure to free memory


def log_training_examples(dataset, n: int = 5):
    """Log training examples to check if the masks were correctly generated."""
    for batch in dataset.take(1):
        # Extract images and masks explicitly
        images = batch[0]
        masks = batch[1]
        break

    batch_size = tf.shape(images)[0]
    n = tf.minimum(n, batch_size).numpy()
    indices = np.random.choice(batch_size, n, replace=False)
    selected_images = tf.gather(images, indices)
    selected_masks = tf.gather(masks, indices)

    # Convert true masks from one-hot encoding if necessary
    if selected_masks.shape[-1] > 1:
        selected_masks = np.argmax(selected_masks, axis=-1)

    for i in range(n):
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(selected_images[i])
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        axes[1].imshow(selected_masks[i], cmap="viridis")
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")

        wandb.log({"Training Sample": wandb.Image(fig)})

        plt.close(fig)  # Close figure to free memory


def jaccard(y_true, y_pred):
    """Jaccard index to compute after each epoch."""
    tp = keras.metrics.TruePositives()
    fp = keras.metrics.FalsePositives()
    fn = keras.metrics.FalseNegatives()

    y_hats = tf.math.argmax(y_pred, axis=-1)
    tp.update_state(y_true, y_hats)
    fp.update_state(y_true, y_hats)
    fn.update_state(y_true, y_hats)

    score = tp.result() / (tp.result() + fp.result() + fn.result())

    return score.numpy()


def dice(y_true, y_pred):
    """Dice coefficient to compute after each epoch."""
    tp = keras.metrics.TruePositives()
    fp = keras.metrics.FalsePositives()
    fn = keras.metrics.FalseNegatives()

    y_hats = tf.math.argmax(y_pred, axis=-1)
    tp.update_state(y_true, y_hats)
    fp.update_state(y_true, y_hats)
    fn.update_state(y_true, y_hats)

    score = 2 * tp.result() / (2 * tp.result() + fp.result() + fn.result())

    return score.numpy()


class TimeCallback(tf.keras.callbacks.Callback):  # pylint: disable=no-member
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
