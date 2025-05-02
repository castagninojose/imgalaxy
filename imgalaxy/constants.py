"""Constants."""

RANDOM_SEED = 419

BUFFER_SIZE = 300
RUN_FROM = 'local'
NUM_EPOCHS = 14
IMAGE_SIZE = 64
MASK = 'spiral_mask'
MIN_VOTE = (
    3  # min votes that the most voted pixel of a mask must have to be a spiral arm (barred) galaxy
)
THRESHOLD = 3  # min votes that a pixel must have to be clasified as a spiral arm (bar)
PATIENCE = 10
