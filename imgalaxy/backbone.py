"""Models to use as a backbone for the UNet."""
from tensorflow.keras import Model
from keras import layers


class BackBoneModel:
    def __init__(
        self,
        image_size,
        n_filters,
        activation,
        out_activation,
        batch_normalization,
        pool,
        unpool,
        n_stack_down,
        n_stack_up,
    ):
        self.image_size = image_size
        self.n_filters = n_filters
        self.activation = activation
        self.out_activaction = out_activation
        self.batch_normalization = batch_normalization
        self.pool = pool
        self.unpool = unpool
        self.n_stack_down = n_stack_down
        self.n_stack_up = n_stack_up


BACKBONE_MODELS = [
    "CustomUNet",
    "U-net",
    "V-net",
    "Attention-Unet",
    "U-net++",
    "UNET 3+",
    "R2U-net",
    "ResUnet-a",
    "U^2-Net",
    "TransUNET",
    "Swin-UNET",
]

