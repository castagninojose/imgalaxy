import click
import yaml

import wandb
from imgalaxy.cfg import PKG_PATH
from imgalaxy.constants import IMAGE_SIZE, NUM_EPOCHS, THRESHOLD
# from imgalaxy.helpers import check_augmented_images, evaluate_model
from imgalaxy.unet import UNet


@click.command()
@click.option(
    "--dropout-rate", default=0.3, show_default=True, help="Inner-layer dropout rate."
)
@click.option(
    "--mask",
    default='spiral_mask',
    show_default=True,
    help="Target mask to use for training.",
)
@click.option(
    "--learning-rate",
    default=0.0011,
    show_default=True,
    help="Learning rate for training.",
)
@click.option(
    "--num-epochs",
    default=NUM_EPOCHS,
    show_default=True,
    help="Total number of epochs (if no patience).",
)
@click.option(
    "--batch-size",
    default=32,
    show_default=True,
    help="Size of the batches used for train/test.",
)
@click.option(
    "--image-size",
    default=IMAGE_SIZE,
    show_default=True,
    help="Input images are resized for training.",
)
@click.option(
    "--n-filters",
    default=128,
    show_default=True,
    help="Base number of filters for convolutions.",
)
@click.option(
    "--min-vote",
    default=3,
    show_default=True,
    help="Min votes for a pixel to be positvely labeled.",
)
@click.option(
    "--batch-normalization",
    default=False,
    show_default=True,
    help="Batch normalization in each double convolution.",
)
@click.option(
    "--kernel-regularization",
    default=None,
    show_default=True,
    help="Use kernel regularization in convolution layers.",
)
@click.option(
    "--loss",
    default="sparse_categorical_crossentropy",
    show_default=True,
    help="Loss function for training.",
)
def train(
    loss,
    dropout_rate,
    num_epochs,
    learning_rate,
    batch_size,
    batch_normalization,
    kernel_regularization,
    image_size,
    n_filters,
    mask,
    min_vote,
):
    with wandb.init(
        project="galaxy-segmentation-project",
        name=f"prueba_jose_{mask}",
        config={
            'loss': loss,
            'dropout': dropout_rate,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'batch_normalization': batch_normalization,
            'kernel_regularization': kernel_regularization,
            'size': image_size,
            'n_filters': n_filters,
            'mask': mask,
            'min_vote': min_vote,
            'threshold': THRESHOLD,
            'group': f"jose_{mask}",
        },
    ):
        unet = UNet(
            loss=loss,
            dropout_rate=dropout_rate,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            batch_normalization=batch_normalization,
            kernel_regularization=kernel_regularization,
            image_size=image_size,
            n_filters=n_filters,
            mask=mask,
            min_vote=min_vote,
        )
        _, _, _ = unet.train_pipeline()
        # check_augmented_images(train_data)
        # evaluate_model(test_data, unet.unet_model, num=3)


if __name__ == '__main__':
    sweep_configs = yaml.safe_load((PKG_PATH / 'sweep.yaml').read_text())
    sweep_id = wandb.sweep(sweep=sweep_configs, project="galaxy-segmentation-project")
    wandb.agent(sweep_id, function=train)
    wandb.agent(
        f"ganegroup/galaxy-segmentation-project/{sweep_id}", function=train, count=47
    )
    #train()
