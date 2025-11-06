from typing import List, Optional, Sequence
from pathlib import Path

import hydra
from omegaconf import OmegaConf, DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from train.utils import utils

log = utils.get_logger(__name__)

import torch
import torch.distributed
import wandb
import pandas as pd
import tempfile
import os
import subprocess


def log_predictions(predictions, datamodule, num_batches, name):
        """
        One issue is that this can only be run with trainer.devices=1 right now. So restart with that after training completes.
        Add resume=True to your training script.
        """
        
        # iterate over batches
        from collections import defaultdict
        gathered_Data = defaultdict(list)
        for i, (batch, preds) in enumerate(zip(datamodule.test_dataloader(), predictions)):
            print(f"batch: {i}")
            if i == num_batches:
                break

            input_ids_batch = batch[0]
            label_ids_batch = batch[1]
            metadata_batch = batch[2]
            doc_index = metadata_batch['doc_index']
            doc_offset = metadata_batch['doc_offset']
            sample_index = metadata_batch['sample_index']
            source_index = metadata_batch['source_index']

            # preds
            preds_batch = preds[0] # (batch_size, seq_len, vocab_size)
            probs = preds_batch.softmax(dim=-1)
            pred_probs, preds = probs.max(dim=-1)
            try:
                label_probs = probs.gather(dim=-1, index=label_ids_batch.unsqueeze(-1)).squeeze(-1)
            except:
                print(f"{i}: {label_ids_batch.shape}")
                print(f"Failed to log predictions...")

            # gather data
            gathered_Data['input_id'].extend(input_ids_batch.cpu().numpy())
            gathered_Data['label_id'].extend(label_ids_batch.cpu().numpy())
            gathered_Data['pred_id'].extend(preds.cpu().numpy())
            gathered_Data['label_prob'].extend(label_probs.float().cpu().numpy())
            gathered_Data['pred_prob'].extend(pred_probs.float().cpu().numpy())
            gathered_Data['doc_index'].extend(doc_index.cpu().numpy())
            gathered_Data['doc_offset'].extend(doc_offset.cpu().numpy())
            gathered_Data['sample_index'].extend(sample_index.cpu().numpy())
            gathered_Data['source_index'].extend(source_index.cpu().numpy())

        df = pd.DataFrame(gathered_Data)
        # print the type of each column
        for col in df.columns:
            print(f"{col}: {type(df[col][0])}")

        print(f"Uploading table with name {name} and {len(df)} rows to wandb.")

        with tempfile.TemporaryDirectory(prefix="/work/sabri_data") as tmp_dir:
            path = os.path.join(tmp_dir, f"{name}.feather")
            df.to_feather(path)
            artifact = wandb.Artifact(
                f"run-{wandb.run.id}-{name}", type="feather"
            )
            artifact.add_file(path)
            wandb.log_artifact(artifact)

            # need to intermittently cleanup cache
            subprocess.run(["wandb", "artifact", "cache", "cleanup", "1g"])


def last_modification_time(path):
    """Including files / directory 1-level below the path
    """
    path = Path(path)
    if path.is_file():
        return path.stat().st_mtime
    elif path.is_dir():
        return max(child.stat().st_mtime for child in path.iterdir())
    else:
        return None


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.

    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # We want to add fields to config so need to call OmegaConf.set_struct
    OmegaConf.set_struct(config, False)
    # Init lightning model
    model: LightningModule = hydra.utils.instantiate(config.task, cfg=config, _recursive_=False)
    datamodule: LightningDataModule = model._datamodule


    # Init lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if cb_conf is not None and "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if lg_conf is not None and "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    ckpt_cfg = {}
    if config.get('resume'):
        try:
            checkpoint_path = Path(config.callbacks.model_checkpoint.dirpath)
            print("HERERE", config.callbacks.model_checkpoint.dirpath, checkpoint_path.is_dir())
            if checkpoint_path.is_dir():
                last_ckpt = checkpoint_path / 'last.ckpt'
                autosave_ckpt = checkpoint_path / '.pl_auto_save.ckpt'
                if not (last_ckpt.exists() or autosave_ckpt.exists()):
                    raise FileNotFoundError("Resume requires either last.ckpt or .pl_autosave.ckpt")
                if ((not last_ckpt.exists())
                    or (autosave_ckpt.exists()
                       and last_modification_time(autosave_ckpt) > last_modification_time(last_ckpt))):
                    # autosave_ckpt = autosave_ckpt.replace(autosave_ckpt.with_name('.pl_auto_save_loaded.ckpt'))
                    checkpoint_path = autosave_ckpt
                else:
                    checkpoint_path = last_ckpt
            # DeepSpeed's checkpoint is a directory, not a file
            if checkpoint_path.is_file() or checkpoint_path.is_dir():
                ckpt_cfg = {'ckpt_path': str(checkpoint_path)}
            else:
                log.info(f'Checkpoint file {str(checkpoint_path)} not found. Will start training from scratch')
        except (KeyError, FileNotFoundError):
            pass

    # Configure ddp automatically
    n_devices = config.trainer.get('devices', 1)
    if isinstance(n_devices, Sequence):  # trainer.devices could be [1, 3] for example
        n_devices = len(n_devices)
    if n_devices > 1 and config.trainer.get('strategy', None) is None:
        config.trainer.strategy = dict(
            _target_='pytorch_lightning.strategies.DDPStrategy',
            find_unused_parameters=False,    # FLAG
            gradient_as_bucket_view=True,  # https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#ddp-optimizations
        )

    # Init lightning trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger)
    
    do_test = getattr(config, "do_test", True)
    do_predict = getattr(config, "do_predict", False)

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, **ckpt_cfg)

    # Evaluate model on test set, using the best model achieved during training
    if do_test and config.get("test_after_training") and not config.trainer.get("fast_dev_run"):
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule)

    # Log predictions
    if do_predict:
        log.info("Starting predictions!")
        indices = range(len(datamodule.test_dataloader()))
        print(f"len(indices): {len(indices)}")
        batch_size = config.datamodule.batch_size_eval
        num_batches = config.datamodule.num_predict_batches_eval
        chunked_indices = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
        predictions = []
        for i, chunk_index in enumerate(chunked_indices):
            if i == num_batches:
                break
            datamodule.indices = chunk_index
            chunk_predictions = trainer.predict(model=model, datamodule=datamodule, return_predictions=True)
            try:
                chunk_predictions = [p[0].cpu() for p in chunk_predictions]
            except:
                chunk_predictions = [p['logits'].cpu() for p in chunk_predictions]
            predictions.append(chunk_predictions)
        log_predictions(predictions, datamodule, num_batches, name=config.expt_name)

    # Make sure everything closed properly
    log.info("Finalizing!")
    utils.finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    if not config.trainer.get("fast_dev_run"):
        log.info(f"Best model ckpt: {trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]

