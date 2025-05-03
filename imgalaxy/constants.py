"""Constants."""

RANDOM_SEED = 419

BUFFER_SIZE = 300
RUN_FROM = 'local'
NUM_EPOCHS = 29
IMAGE_SIZE = 128
MASK = 'spiral_mask'
MIN_VOTE = (
    3  # min votes that the most voted pixel of a mask must have to be a spiral arm (barred) galaxy
)
THRESHOLD = 3  # min votes that a pixel must have to be clasified as a spiral arm (bar)
PATIENCE = 10

UNET_SWEEP_CONFIGS = {
    "name": "VNet",
    "method": "bayes",
    "metric": {"goal": "minimize", "name": "val_IoU_1"},
    "parameters": {
        "learning_rate": {"distribution": "uniform", "min": 0.0001, "max": 0.1},
        "label_smoothing": {"distribution": "uniform", "min": 0.0, "max": 1.0},
        "activation": {"values": ["ReLU", "Softmax"]},
        "output_activation": {"values": ["ReLU", "Softmax", "Sigmoid"]},
        "batch_normalization": {"values": [True, False]},
    },
}
