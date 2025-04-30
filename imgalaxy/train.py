# pylint: disable=no-member  # pylint keeps insisting that there's no tf.keras 🤷
import click
import tensorflow as tf
import tensorflow_datasets as tfds
import yaml  # type: ignore
from keras_unet_collection import models
from tensorflow.keras import mixed_precision
from wandb.keras import WandbMetricsLogger

import wandb
from imgalaxy.cfg import MODELS_DIR, PKG_PATH
from imgalaxy.constants import IMAGE_SIZE, MIN_VOTE, NUM_EPOCHS
from imgalaxy.helpers import log_predictions
from imgalaxy.unet import AugmentedSegmentationModel, GZ3DPipeline

mixed_precision.set_global_policy("mixed_float16")


@click.command()
@click.option("--learning-rate", default=0.07, show_default=True, help="Learning rate.")
@click.option("--activation", default="ReLU", show_default=True, help="Activation function.")
@click.option("--batch-norm", default=False, show_default=True, help="Apply batch normalization.")
@click.option(
    "--out-activation", default="Softmax", show_default=True, help="Output activation function."
)
@click.option("--pool", default=False, show_default=True, help="Downsample strategy.")
@click.option("--unpool", default=False, show_default=True, help="Upsampling strategy.")
def train(
    learning_rate,
    activation,
    batch_norm,
    out_activation,
    pool,
    unpool,
):
    gpu = tf.config.list_physical_devices("GPU")[0]
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.optimizer.set_jit(True)
    with wandb.init(
        project="galaxy-segmentation-project",
        name="att_unet_spirals_bars_int",
        config={
            'group': "jose_spirals_bars_int",
            'learning_rate': learning_rate,
            'activation': activation,
            'batch_norm': batch_norm,
            'out_activation': out_activation,
        },
    ):
        segmentation_model = models.vnet_2d(
            (IMAGE_SIZE, IMAGE_SIZE, 3),
            n_labels=4,
            filter_num=[64, 128, 256, 512],
            activation=activation,
            output_activation=out_activation,
            pool=pool,
            unpool=unpool,
            name='vnet',
        )

        model = AugmentedSegmentationModel(
            augmentations=[
                tf.keras.layers.RandomFlip(mode="horizontal and vertical", seed=101),
                tf.keras.layers.RandomRotation(factor=(0, 1), seed=101),
                tf.keras.layers.RandomZoom(height_factor=(-0.2, +0.2)),
            ],
            segmentation_model=segmentation_model,
        )
        pipeline = GZ3DPipeline(
            size=IMAGE_SIZE,
            preprocess_input=None,
            binary_threshold=True,
            clip_votes_max=6,
            sparse=True,
            shuffle_buffer_size=1000,
            cache=False,
            prefetch=True,
        )

        ds_train, ds_val, ds_test = tfds.load(
            'galaxy_zoo3d', split=['train[:75%]', 'train[75%:90%]', 'train[90%:]']
        )

        ds_train = ds_train.filter(lambda x: tf.reduce_max(x['spiral_mask']) >= MIN_VOTE)
        ds_train = ds_train.filter(lambda x: tf.reduce_max(x['bar_mask']) >= MIN_VOTE)
        ds_val = ds_val.filter(lambda x: tf.reduce_max(x['spiral_mask']) >= MIN_VOTE)
        ds_val = ds_val.filter(lambda x: tf.reduce_max(x['bar_mask']) >= MIN_VOTE)
        ds_test = ds_test.filter(lambda x: tf.reduce_max(x['spiral_mask']) >= MIN_VOTE)
        ds_test = ds_test.filter(lambda x: tf.reduce_max(x['bar_mask']) >= MIN_VOTE)
        train_batches = pipeline(ds_train)
        val_batches = pipeline(ds_val)
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=[
                tf.keras.metrics.IoU(
                    num_classes=4,
                    target_class_ids=[1],
                    ignore_class=0,
                    sparse_y_true=True,
                    sparse_y_pred=False,
                    name="IoU_1",
                ),
                tf.keras.metrics.IoU(
                    num_classes=4,
                    target_class_ids=[2],
                    ignore_class=0,
                    sparse_y_true=True,
                    sparse_y_pred=False,
                    name="IoU_2",
                ),
                tf.keras.metrics.IoU(
                    num_classes=4,
                    target_class_ids=[3],
                    ignore_class=0,
                    sparse_y_true=True,
                    sparse_y_pred=False,
                    name="IoU_3",
                ),
                tf.keras.metrics.MeanIoU(
                    num_classes=4, sparse_y_true=False, sparse_y_pred=False, name="MeanIoU"
                ),
            ],
        )
        _ = model.fit(
            train_batches,
            epochs=NUM_EPOCHS,
            steps_per_epoch=(5000 // pipeline.batch_size),
            validation_steps=(5000 // pipeline.batch_size),
            validation_data=val_batches,
            callbacks=[
                WandbMetricsLogger(),
                tf.keras.callbacks.ModelCheckpoint(
                    MODELS_DIR / "best_att_comp.keras",
                    monitor='val_IoU_1',
                    save_best_only=True,
                    mode='max',
                ),
            ],
        )
        log_predictions(pipeline(ds_test), model, n=23)


if __name__ == '__main__':
    sweep_configs = yaml.safe_load((PKG_PATH / 'sweep_vnet.yaml').read_text())
    sweep_id = wandb.sweep(sweep=sweep_configs, project="galaxy-segmentation-project")
    wandb.agent(sweep_id, function=train)
    wandb.agent(f"ganegroup/galaxy-segmentation-project/{sweep_id}", function=train, count=29)
    # train()  # pylint: disable=no-value-for-parameter
