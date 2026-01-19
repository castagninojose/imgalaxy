# pylint: disable=no-member  # pylint keeps insisting that there's no tf.keras 🤷
import click
import tensorflow as tf
import tensorflow_datasets as tfds
import wandb
from tensorflow.keras import mixed_precision
from wandb.integration.keras import WandbMetricsLogger

from imgalaxy.constants import IMAGE_SIZE, MIN_VOTE, NUM_EPOCHS
from imgalaxy.helpers import get_iou_configs, log_predictions
from imgalaxy.unet import AugmentedSegmentationModel, GZ3DPipeline, LensingPipeline, build_model

mixed_precision.set_global_policy("mixed_float16")
tf.get_logger().setLevel("ERROR")

BASE: dict = {
    "method": "bayes",
    "run_cap": 29,
    "metric": {"name": "epoch/val_IoU_1", "goal": "maximize"},
    "early_terminate": {"type": "hyperband", "min_iter": 11, "max_iter": 150},
    "parameters": {
        "learning_rate": {"distribution": "log_uniform_values", "min": 1e-5, "max": 5e-4},
        "label_smoothing": {"distribution": "uniform", "min": 0, "max": 1},
        'alpha': {"distribution": "uniform", "min": 0.05, "max": 0.8},
        'gamma': {"distribution": "uniform", "min": 0.5, "max": 5.0},
        "filter_num": {"values": [[32, 64, 128, 256], [64, 128, 256, 512]]},
        "activation": {"values": ["ReLU", "GELU"]},
        "batch_norm": {"values": [True, False]},
        "sparse": {"values": [True, False]},
    },
}

DEFAULT_CONFIG: dict = {
    "learning_rate": 1e-4,
    "filter_num": [32, 64, 128, 256],
    "activation": "ReLU",
    "batch_norm": True,
    "pool": "max",
    "unpool": True,
    "label_smoothing": 0.0,
    "sparse": True,
    "alpha": 0.25,
    "gamma": 1,
}


def with_overrides(base: dict, overrides: dict) -> dict:
    """Create run config by combining `base` and `overrides` dictionaries."""
    out: dict = {**base}
    out["parameters"] = {**base["parameters"], **overrides}
    return out


SWEEP_CONFIGS = {
    "att_unet": with_overrides(
        BASE, {"pool": {"values": ["max", "ave"]}, "unpool": {"values": [True]}}
    ),
    "res_unet": with_overrides(BASE, {"pool": {"values": ["max"]}, "unpool": {"values": [True]}}),
    "vnet": with_overrides(BASE, {"filter_num": {"values": [[64, 128, 256, 512]]}}),
    "trans_unet": with_overrides(
        BASE,
        {
            "filter_num": {"values": [[64, 128, 256], [64, 128, 256, 512]]},
            "learning_rate": {"distribution": "log_uniform_values", "min": 5e-5, "max": 1e-3},
        },
    ),
}


def train(
    *,
    task: str = 'lensing',
    model_name: str = 'att_unet',
    project: str = 'imgalaxy',
    sparse: bool = False,
) -> None:
    """
    Main training routine compatible with one-off runs and with hyperparameters tuning.

    Parameters
    ----------
    task : {'lensing', 'galaxy_zoo3d'}
        Segmentation task to train the model for.
    model_name : {'att_unet', 'res_unet', 'trans_unet', 'vnet'}
        Name of the model to use. See `keras_unet_collection`[^1] for more details.
    project : {'galaxy-segmentation-project', 'imgalaxy'}
        Wandb project name to track training and log results.
    sparse : bool
        Whether or not to use `sparse` labels. If `False`, one-hot encoding will be used.

    Returns
    -------
    No value

    References
    ----------
    [^1] https://github.com/yingkaisha/keras-unet-collection

    """
    if task not in ["lensing", "galaxy_zoo3d"]:
        raise ValueError(f"Invalid task: {task}")

    gpu = tf.config.list_physical_devices("GPU")[0]
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.optimizer.set_jit(True)

    with wandb.init(project=project, config=DEFAULT_CONFIG) as run:
        cfg = run.config
        cfg.task = task
        cfg.model = model_name
        run.name = f"{model_name}_{task}"

        if task == "lensing":
            channels = 5
            pipeline = LensingPipeline(size=64, sparse=sparse)
        else:
            channels = 3
            pipeline = GZ3DPipeline(
                size=IMAGE_SIZE,
                binary_threshold=True,
                clip_votes_max=6,
                sparse=sparse,
            )

        segmentation_model = build_model(
            name=model_name,
            input_shape=(pipeline.size, pipeline.size, channels),
            n_labels=4,
            cfg=cfg,
        )

        model = AugmentedSegmentationModel(
            augmentations=[
                tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=101),
                tf.keras.layers.RandomRotation(1.0, seed=101),
                tf.keras.layers.RandomZoom((-0.2, 0.2)),
            ],
            segmentation_model=segmentation_model,
        )

        ds_train, ds_val, ds_test = tfds.load(
            task, split=["train[:75%]", "train[75%:90%]", "train[90%:]"]
        )

        if task == "galaxy_zoo3d":
            for ds_name in ["ds_train", "ds_val", "ds_test"]:
                ds = locals()[ds_name]
                ds = ds.filter(lambda x: tf.reduce_max(x["spiral_mask"]) >= MIN_VOTE)
                ds = ds.filter(lambda x: tf.reduce_max(x["bar_mask"]) >= MIN_VOTE)
                locals()[ds_name] = ds

        train_batches = pipeline(ds_train).repeat()
        val_batches = pipeline(ds_val).repeat()

        if sparse:
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        else:
            # loss = tf.keras.losses.CategoricalCrossentropy(
            #     label_smoothing=cfg.label_smoothing, from_logits=False,
            # )
            loss = tf.keras.losses.CategoricalFocalCrossentropy(
                label_smoothing=cfg.label_smoothing,
                alpha=cfg.alpha,
                gamma=cfg.gamma,
                from_logits=False,
            )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate, clipnorm=1.0),
            loss=loss,
            metrics=get_iou_configs(sparse=False),
        )

        model.fit(
            train_batches,
            epochs=NUM_EPOCHS,
            steps_per_epoch=5000 // pipeline.batch_size,
            validation_data=val_batches,
            validation_steps=5000 // pipeline.batch_size,
            callbacks=[
                WandbMetricsLogger(),
                # tf.keras.callbacks.ModelCheckpoint(
                #     str(MODELS_DIR / f"best_{task}_{model_name}.keras"),
                #     monitor="val_IoU_1",
                #     save_best_only=True,
                #     mode="max",
                # ),
            ],
        )

        log_predictions(pipeline(ds_test), model, task, n=23)


@click.command()
@click.option("--task", required=True, type=click.Choice(["lensing", "galaxy_zoo3d"]))
@click.option(
    "--model",
    required=True,
    default="att_unet",
    type=click.Choice(["att_unet", "res_unet", "vnet", "trans_unet"]),
)
@click.option(
    "--sweep", is_flag=True, default=False, show_default=True, help="Launch a wandb sweep."
)
@click.option(
    "--sparse/--one-hot",
    default=True,
    show_default=True,
    is_flag=True,
    help="Use sparse labels. If `False`, one-hot encoded labels are used instead.",
)
def main(task, model, sweep, sparse):
    """CLI wrapper method for `train.py`."""
    if task == 'lensing':
        project = 'imgalaxy'
    else:
        project = 'galaxy-segmentation-project'

    if sweep:
        sweep_config = SWEEP_CONFIGS[model]
        sweep_id = wandb.sweep(sweep_config, project=project)
        wandb.agent(
            sweep_id,
            function=lambda: train(task=task, model_name=model, sparse=sparse, project=project),
        )
    else:
        train(task=task, model_name=model, sparse=sparse, project=project)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
