# pylint: disable=no-member  # pylint keeps insisting that there's no tf.keras 🤷
import click
import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
from keras_unet_collection import models
from tensorflow.keras import mixed_precision
from wandb.integration.keras import WandbMetricsLogger

from imgalaxy.cfg import MODELS_DIR
from imgalaxy.constants import IMAGE_SIZE, MIN_VOTE, NUM_EPOCHS
from imgalaxy.helpers import get_iou_configs, log_predictions
from imgalaxy.unet import AugmentedSegmentationModel, GZ3DPipeline, LensingPipeline

mixed_precision.set_global_policy("mixed_float16")
tf.get_logger().setLevel("ERROR")


def safe_cce(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    return tf.reduce_mean(
        tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)
    )


@click.command()
@click.option("--learning-rate", default=1e-04, show_default=True, help="Learning rate.")
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
@click.option("--loss-alpha", default=0.25, show_default=True, help="Weight balancing factor.")
@click.option("--loss-gamma", default=2.0, show_default=True, help="Focus parameter.")
@click.option("--label-smoothing", default=0.0, show_default=True, help="Label smoothing factor.")
@click.option(
    "--task",
    default='galaxy_zoo3d',
    show_default=True,
    help="Segmentation task. Either 'lensing' or  'galaxy_zoo3d'.",
)
@click.option(
    "--sparse/--one-hot",
    default=True,
    show_default=True,
    is_flag=True,
    help="Use sparse labels. If `False`, one-hot encoded labels are used instead.",
)
def train(
    learning_rate,
    activation,
    batch_norm,
    out_activation,
    pool,
    unpool,
    sparse,
    task,
    loss_alpha,
    loss_gamma,
    label_smoothing,
):
    if task not in ['lensing', 'galaxy_zoo3d']:
        raise ValueError(f"Task must be one of 'lensing' or 'galaxy_zoo3d'. Instead got: {task}.")

    gpu = tf.config.list_physical_devices("GPU")[0]
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.optimizer.set_jit(True)
    with wandb.init(
        project="imgalaxy",  # f"{task}-segmentation-project" could be used
        name=f"att_unet_{task}",
        config={
            'group': f"jose_{task}",
            'learning_rate': learning_rate,
            'activation': activation,
            'batch_norm': batch_norm,
            'out_activation': out_activation,
            'task': task,
            'sparse': sparse,
        },
    ):
        channels = 3
        if task == 'lensing':
            channels: int = 5  # lensing images have 5 channels
            pipeline = LensingPipeline(size=64, sparse=sparse)

        else:
            pipeline = GZ3DPipeline(
                size=IMAGE_SIZE, binary_threshold=True, clip_votes_max=6, sparse=sparse
            )

        # segmentation_model = models.vnet_2d(
        #     (pipeline.size, pipeline.size, channels),
        #     n_labels=4,
        #     filter_num=[64, 128, 256, 512],
        #     activation=activation,
        #     output_activation=out_activation,
        #     pool=pool,
        #     unpool=unpool,
        #     name='vnet',
        # )

        segmentation_model = models.att_unet_2d(
            (pipeline.size, pipeline.size, channels),
            n_labels=4,
            filter_num=[64, 128, 256, 512],
            activation=activation,
            output_activation="Softmax",
            pool=pool,
            unpool=unpool,
            name='att_unet',
        )

        model = AugmentedSegmentationModel(
            augmentations=[
                tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=101),
                tf.keras.layers.RandomRotation(factor=(0, 1), seed=101),
                tf.keras.layers.RandomZoom(height_factor=(-0.2, +0.2)),
            ],
            segmentation_model=segmentation_model,
        )

        ds_train, ds_val, ds_test = tfds.load(
            task, split=['train[:75%]', 'train[75%:90%]', 'train[90%:]']
        )

        if task == 'galaxy_zoo3d':
            # Make sure the selected galaxies have "positive" bar and spiral masks
            # We could increase the dataset size filtering only by spiral arms
            ds_train = ds_train.filter(lambda x: tf.reduce_max(x['spiral_mask']) >= MIN_VOTE)
            ds_train = ds_train.filter(lambda x: tf.reduce_max(x['bar_mask']) >= MIN_VOTE)
            ds_val = ds_val.filter(lambda x: tf.reduce_max(x['spiral_mask']) >= MIN_VOTE)
            ds_val = ds_val.filter(lambda x: tf.reduce_max(x['bar_mask']) >= MIN_VOTE)
            ds_test = ds_test.filter(lambda x: tf.reduce_max(x['spiral_mask']) >= MIN_VOTE)
            ds_test = ds_test.filter(lambda x: tf.reduce_max(x['bar_mask']) >= MIN_VOTE)

        train_batches = pipeline(ds_train).repeat()
        val_batches = pipeline(ds_val).repeat()

        if sparse:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        else:
            loss = tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=label_smoothing, from_logits=False, reduction="sum_over_batch_size"
            )

        model.compile(
            loss=loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
            metrics=get_iou_configs(sparse=False),
        )

        # --- DEBUG: manual validation loss probe ---
        val_batch = next(iter(val_batches))
        images_dbg, masks_dbg = val_batch

        preds_dbg = model(images_dbg, training=False)
        loss_dbg = model.compiled_loss(masks_dbg, preds_dbg)

        tf.print(
            "MANUAL VAL LOSS:",
            loss_dbg,
            "mask min/max:",
            tf.reduce_min(masks_dbg),
            tf.reduce_max(masks_dbg),
            "pred min/max:",
            tf.reduce_min(preds_dbg),
            tf.reduce_max(preds_dbg),
        )
        # --- DEBUG: manual validation loss probe ---

        _ = model.fit(
            train_batches,
            epochs=NUM_EPOCHS,
            steps_per_epoch=(5000 // pipeline.batch_size),  # TODO May 2025: avoid magic number
            validation_steps=(5000 // pipeline.batch_size),
            validation_data=val_batches,
            callbacks=[
                WandbMetricsLogger(),
                tf.keras.callbacks.ModelCheckpoint(
                    str(MODELS_DIR / f"best_{task}.keras"),
                    monitor='val_IoU_1',
                    save_best_only=True,
                    mode='max',
                ),
            ],
        )
        log_predictions(pipeline(ds_test), model, task, n=23)


if __name__ == '__main__':
    # sweep_id = wandb.sweep(sweep=UNET_SWEEP_CONFIGS, project="imgalaxy")
    # wandb.agent(sweep_id, function=train)
    # wandb.agent(f"ganegroup/galaxy-segmentation-project/{sweep_id}", function=train, count=29)
    train()  # pylint: disable=no-value-for-parameter
