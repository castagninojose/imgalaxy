"""Helper methods"""

from datetime import datetime

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb
from sklearn.metrics import confusion_matrix, jaccard_score

tf.config.run_functions_eagerly(True)


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask


def evaluate_model(dataset, model, num=1):
    """Evaluate model after a run is completed."""
    # TODO: try tf.keras.Model.evaluate()
    if dataset:
        for image, mask in dataset:
            pred_mask = create_mask(model.predict(image))
            for ind in range(num):
                wandb.log({"example": [wandb.Image(image[ind]), wandb.Image(mask[ind]), wandb.Image(pred_mask[ind])]})
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
