# pylint: disable=no-member
import click
import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
import yaml  # type: ignore
from keras_unet_collection import models
from tensorflow.keras.applications.vgg16 import preprocess_input
from wandb.keras import WandbMetricsLogger

from imgalaxy.cfg import MODELS_DIR, PKG_PATH
from imgalaxy.constants import IMAGE_SIZE, MASK, MIN_VOTE, NUM_EPOCHS
from imgalaxy.helpers import check_augmented_images, evaluate_model
from imgalaxy.unet import AugmentedSegmentationModel, GZ3DPipeline


@click.command()
# @click.option(
#    "--loss", default="categorical_focal_crossentropy", show_default=True, help="Loss function."
# )
@click.option("--learning-rate", default=1e-02, show_default=True, help="Learning rate.")
@click.option("--activation", default="ReLU", show_default=True, help="Activation function.")
@click.option("--batch-norm", default=False, show_default=True, help="Apply batch normalization.")
@click.option(
    "--atten-activation", default="ReLU", show_default=True, help="Non-linear attention activation."
)
@click.option(
    "--out-activation", default="Softmax", show_default=True, help="Output activation function."
)
@click.option("--pool", default=True, show_default=True, help="Downsample strategy.")
@click.option("--unpool", default=True, show_default=True, help="Upsampling strategy.")
@click.option(
    "--stack-num-down",
    default=2,
    show_default=True,
    help="Number of convolutional layers per downsampling level/block.",
)
@click.option(
    "--stack-num-up",
    default=2,
    show_default=True,
    help="Number of convolutional layers (after concatenation) per upsampling level/block.",
)
@click.option(
    "--loss-alpha", default=0.25, show_default=True, help="Focusing parameter for loss function."
)
@click.option(
    "--loss-gamma", default=0.2, show_default=True, help="Focusing parameter for loss function."
)
@click.option("--loss-smoothing", default=0.0, show_default=True, help="Label smoothing.")
@click.option("--attention", default="add", show_default=True, help="Applied additive attention.")
def train(
    learning_rate,
    activation,
    batch_norm,
    atten_activation,
    out_activation,
    pool,
    unpool,
    stack_num_down,
    stack_num_up,
    loss_alpha,
    loss_gamma,
    loss_smoothing,
    attention,
):
    with wandb.init(
        project="galaxy-segmentation-project",
        name=f"attention_unet_{MASK}",
        config={
            'group': f"jose_{MASK}",
        },
    ):
        segmentation_model = models.att_unet_2d(
            (IMAGE_SIZE, IMAGE_SIZE, 3),
            filter_num=[64, 128, 256, 512, 1024],
            n_labels=2,
            stack_num_down=stack_num_down,
            stack_num_up=stack_num_up,
            activation=activation,
            atten_activation=atten_activation,
            attention=attention,
            output_activation=out_activation,
            batch_norm=batch_norm,
            pool=pool,
            unpool=unpool,
            backbone='VGG16',
            weights="imagenet",
            freeze_backbone=True,
            freeze_batch_norm=True,
            name='attunet',
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
            mask_key=MASK,
            preprocess_input=preprocess_input,
            binary_threshold=MIN_VOTE,
            clip_votes_max=6,
            sparse=False,
            shuffle_buffer_size=1000,
            cache=True,
            prefetch=True,
        )

        ds_train, ds_val, ds_test = tfds.load(
            'galaxy_zoo3d', split=['train[:75%]', 'train[75%:90%]', 'train[90%:]']
        )

        ds_train = ds_train.filter(lambda x: tf.reduce_max(x[MASK]) >= MIN_VOTE)
        ds_val = ds_val.filter(lambda x: tf.reduce_max(x[MASK]) >= MIN_VOTE)
        ds_test = ds_test.filter(lambda x: tf.reduce_max(x[MASK]) >= MIN_VOTE)
        train_batches = pipeline(ds_train)
        val_batches = pipeline(ds_val)
        loss = tf.keras.losses.CategoricalFocalCrossentropy(
            alpha=[loss_alpha, 1 - loss_alpha],
            gamma=loss_gamma,
            label_smoothing=loss_smoothing,
            from_logits=False,
        )
        model.compile(
            loss=loss,
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=[
                tf.keras.metrics.IoU(
                    num_classes=2,
                    target_class_ids=[0],
                    sparse_y_true=False,
                    sparse_y_pred=False,
                    name="IoU_0",
                ),
                tf.keras.metrics.IoU(
                    num_classes=2,
                    target_class_ids=[1],
                    sparse_y_true=False,
                    sparse_y_pred=False,
                    name="IoU_1",
                ),
                tf.keras.metrics.MeanIoU(
                    num_classes=2, sparse_y_true=False, sparse_y_pred=False, name="MeanIoU"
                ),
                tf.keras.losses.Dice(),
            ],
        )
        model_history = model.fit(
            train_batches,
            epochs=NUM_EPOCHS,
            # steps_per_epoch=STEPS_PER_EPOCH,
            # validation_steps=VALIDATION_STEPS,
            validation_data=val_batches,
            callbacks=[
                WandbMetricsLogger(),
                tf.keras.callbacks.ModelCheckpoint(
                    MODELS_DIR / "best_att_spirals.keras",
                    monitor='val_IoU_1',
                    save_best_only=True,
                    mode='max',
                ),
            ],
        )
        check_augmented_images(ds_train)
        evaluate_model(ds_test, model_history, num=3)


if __name__ == '__main__':
    sweep_configs = yaml.safe_load((PKG_PATH / 'sweep.yaml').read_text())
    sweep_id = wandb.sweep(sweep=sweep_configs, project="galaxy-segmentation-project")
    wandb.agent(sweep_id, function=train)
    wandb.agent(f"ganegroup/galaxy-segmentation-project/{sweep_id}", function=train, count=47)
    train()  # pylint: disable=no-value-for-parameter
