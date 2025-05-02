# pylint: disable=no-member  # pylint keeps insisting that there's no tf.keras 🤷
import click
import tensorflow as tf
import tensorflow_datasets as tfds
import yaml  # type: ignore  # pylint: disable=unused-import  # noqa: F401
from keras_unet_collection import models
from tensorflow.keras import mixed_precision
from wandb.keras import WandbMetricsLogger

import wandb
from imgalaxy.cfg import MODELS_DIR
from imgalaxy.constants import IMAGE_SIZE, MIN_VOTE, NUM_EPOCHS
from imgalaxy.helpers import log_predictions
from imgalaxy.unet import AugmentedSegmentationModel, GZ3DPipeline, LensingPipeline

mixed_precision.set_global_policy("mixed_float16")


@click.command()
@click.option("--learning-rate", default=1e-02, show_default=True, help="Learning rate.")
@click.option(
    "--activation",
    default="ReLU",
    show_default=True,
    help="Activation function. One of `keras_unet_collection.activations`.",
)
@click.option("--batch-norm", default=False, show_default=True, help="Apply batch normalization.")
@click.option(
    "--out-activation",
    default="Softmax",
    show_default=True,
    help="Output activation function. One of `keras_unet_collection.activations`.",
)
@click.option("--pool", default=False, show_default=False, help="Downsample strategy.")
@click.option("--unpool", default=False, show_default=False, help="Upsampling strategy.")
@click.option(
    "--task",
    default='galaxy_zoo3d',
    show_default=True,
    help="Segmentation task. Either 'lensing' or  'galaxy_zoo3d'.",
)
def train(learning_rate, activation, batch_norm, out_activation, pool, unpool, task):
    if task not in ['lensing', 'galaxy_zoo3d']:
        raise ValueError(f"Task must be one of 'lensing' or 'galaxy_zoo3d'. Instead got: {task}.")

    gpu = tf.config.list_physical_devices("GPU")[0]
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.optimizer.set_jit(True)
    with wandb.init(
        project="imgalaxy",  # f"{task}-segmentation-project" could be used
        name=f"unet_{task}",
        config={
            'group': f"jose_{task}",
            'learning_rate': learning_rate,
            'activation': activation,
            'batch_norm': batch_norm,
            'out_activation': out_activation,
        },
    ):
        channels = 3
        if task == 'lensing':
            channels: int = 5  # lensing images have 5 channels
            pipeline = LensingPipeline(size=64)  # TODO May 2025: avoid using magic number

        else:
            pipeline = GZ3DPipeline(size=IMAGE_SIZE, binary_threshold=True, clip_votes_max=6)

        segmentation_model = models.vnet_2d(
            (IMAGE_SIZE, IMAGE_SIZE, channels),
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

        ds_train, ds_val, ds_test = tfds.load(
            task, split=['train[:75%]', 'train[75%:90%]', 'train[90%:]']
        )

        if task == 'galaxy_zoo3d':
            # Make sure the selected galaxies have positives bar and spiral masks
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
            steps_per_epoch=(5000 // pipeline.batch_size),  # TODO May 2025: avoid magic number
            validation_steps=(5000 // pipeline.batch_size),
            validation_data=val_batches,
            callbacks=[
                WandbMetricsLogger(),
                tf.keras.callbacks.ModelCheckpoint(
                    MODELS_DIR / f"best_{task}.keras",
                    monitor='val_IoU_1',
                    save_best_only=True,
                    mode='max',
                ),
            ],
        )
        log_predictions(pipeline(ds_test), model, task, n=23)


def train_lensing(
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
        project="imgalaxy",
        name="vnet_lens_and_source",
        config={
            'group': "jose_lensing",
            'learning_rate': learning_rate,
            'activation': activation,
            'batch_norm': batch_norm,
            'out_activation': out_activation,
        },
    ):
        segmentation_model = models.vnet_2d(
            (IMAGE_SIZE, IMAGE_SIZE, 5),
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
        pipeline = LensingPipeline(
            size=IMAGE_SIZE,
            preprocess_input=None,
            binary_threshold=True,
            clip_votes_max=6,
            sparse=True,
            shuffle_buffer_size=1000,
            cache=True,
            prefetch=True,
        )

        ds_train, ds_val, ds_test = tfds.load(
            'lensing', split=['train[:75%]', 'train[75%:90%]', 'train[90%:]']
        )

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
            steps_per_epoch=(9999 // pipeline.batch_size),
            validation_steps=(9999 // pipeline.batch_size),
            validation_data=val_batches,
            callbacks=[
                WandbMetricsLogger(),
                tf.keras.callbacks.ModelCheckpoint(
                    MODELS_DIR / "best_vnet_lensing.keras",
                    monitor='val_IoU_1',
                    save_best_only=True,
                    mode='max',
                ),
            ],
        )
        log_predictions(pipeline(ds_test), model, n=23)


if __name__ == '__main__':
    # sweep_configs = yaml.safe_load((PKG_PATH / 'sweep_vnet.yaml').read_text())
    # sweep_id = wandb.sweep(sweep=sweep_configs, project="galaxy-segmentation-project")
    # wandb.agent(sweep_id, function=train)
    # wandb.agent(f"ganegroup/galaxy-segmentation-project/{sweep_id}", function=train, count=29)
    train()  # pylint: disable=no-value-for-parameter
