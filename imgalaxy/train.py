import click
import wandb
import yaml  # type: ignore

from imgalaxy.constants import IMAGE_SIZE, NUM_EPOCHS, THRESHOLD
from imgalaxy.helpers import check_augmented_images, evaluate_model
from imgalaxy.unet import AttentionUNet


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
@click.option(
    "--activation",
    default="ReLU",
    show_default=True,
    help="one of `tf.keras.layers` or `keras_unet_collection.activations` interfaces",
)
@click.option(
    "--attention",
    default="add",
    show_default=True,
    help="'add' for additive attention. 'multiply' for multiplicative attention.",
)
@click.option(
    "--atten-activation",
    default="ReLU",
    show_default=True,
    help="Nonlinear atteNtion activation. The 'sigma_1' in Oktay et al. 2018.",
)
@click.option(
    "--out-activation",
    default="Softmax",
    show_default=True,
    help="One of `tf.keras.layers` or `keras_unet_collection.activations` interface.",
)
@click.option(
    "--pool",
    default=False,
    show_default=True,
    help="True or 'max' for MaxPooling2D, 'ave' for AveragePooling2D. False for strided conv + batch normalization + activation.",
)
@click.option(
    "--unpool",
    default=False,
    show_default=True,
    help="True or 'bilinear' for Upsampling2D with bilinear interpolation, 'nearest' for Upsampling2D with nearest interpolation. False for Conv2DTranspose + batch norm + activation..",
)
@click.option(
    "--stack-num-down",
    default=2,
    show_default=True,
    help="Number of convolutional layers per downsampling level/block",
)
@click.option(
    "--stack-num-up",
    default=2,
    show_default=True,
    help="Number of convolutional layers (after concatenation) per upsampling level/block",
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
    activation,
    attention,
    atten_activation,
    out_activation,
    pool,
    unpool,
    stack_num_down,
    stack_num_up,
):
    with wandb.init(
        project="galaxy-segmentation-project",
        name=f"attention_unet_{mask}",
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
            'attention': attention,
            'activation': activation,
            'atten_activation': atten_activation,
            'output_activation': out_activation,
            'pool': False,
            'unpool': False,
            'stack_num_down': 2,
            'stack_num_up': 2,
            'group': f"jose_{mask}",
        },
    ):
        unet = AttentionUNet(
            backbone='ResNet101V2',
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
            attention=attention,
            activation=activation,
            atten_activation=atten_activation,
            output_activation=out_activation,
            pool=pool,
            unpool=unpool,
            stack_num_down=stack_num_down,
            stack_num_up=stack_num_up,
        )
        _, test_data, train_data = unet.train_pipeline()
        # check_augmented_images(train_data)
        evaluate_model(test_data, unet.unet_model, num=7)


if __name__ == '__main__':
    # sweep_configs = yaml.safe_load((PKG_PATH / 'sweep.yaml').read_text())
    # sweep_id = wandb.sweep(sweep=sweep_configs, project="galaxy-segmentation-project")
    # wandb.agent(sweep_id, function=train)
    # wandb.agent(
    #     f"ganegroup/galaxy-segmentation-project/{sweep_id}", function=train, count=47
    # )
    train()
