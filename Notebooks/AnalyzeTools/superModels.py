import warnings, torch
warnings.filterwarnings("ignore")

import pytorch_lightning as pl
pl.seed_everything(123)

from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, DeepAR, RecurrentNetwork, GroupNormalizer
from pytorch_forecasting.metrics import MAPE, NormalDistributionLoss, QuantileLoss, SMAPE

from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from .prepare import retriveBestModelPath

def RNN(
    data, 
    training_cutoff, 
    target, 
    group, 
    max_encoder_length, 
    max_prediction_length, 
    time_varying_known_categoricals,
    time_varying_unknown_categoricals, 
    time_varying_known_reals, 
    batch_size,
    saving_dir,
    cell='LSTM'

):
    best_model_path = retriveBestModelPath(saving_dir)

    data[time_varying_known_categoricals] = data[time_varying_known_categoricals].astype(str).astype("category")
    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=target,
        group_ids=group,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_unknown_categoricals=time_varying_unknown_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=[target],
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, 
        data, 
        min_prediction_idx=training.index.time.max() + 1,
        stop_randomization=True
    )

    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    if not best_model_path:
        early_stop_callback = EarlyStopping(monitor="val_loss", verbose=False, mode="min")
        lr_logger = LearningRateMonitor()

        trainer = pl.Trainer(
            max_epochs=100,
            gpus=0,
            weights_summary='top',
            callbacks=[lr_logger, early_stop_callback],
            log_every_n_steps=10,
            default_root_dir=saving_dir,
        )

        model = RecurrentNetwork.from_dataset(
            training,
            cell_type=cell,
            hidden_size=128,
            rnn_layers=1,
            dropout=0.1,
            output_size=1,
            loss=MAPE(),
            log_interval=10
        )

        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )
    
    best_model_path = retriveBestModelPath(saving_dir)
    best_model = RecurrentNetwork.load_from_checkpoint(best_model_path, cell_type=cell)

    return best_model, val_dataloader

def TFT(
    data, 
    training_cutoff, 
    target, 
    group, 
    max_encoder_length, 
    max_prediction_length, 
    time_varying_known_categoricals,
    time_varying_unknown_categoricals, 
    time_varying_known_reals, 
    time_varying_unknown_reals,
    batch_size,
    saving_dir,
):
    best_model_path = retriveBestModelPath(saving_dir)

    data[time_varying_known_categoricals] = data[time_varying_known_categoricals].astype(str).astype("category")
    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=target,
        group_ids=group,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_unknown_categoricals=time_varying_unknown_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, 
        data, 
        min_prediction_idx=training.index.time.max() + 1,
        stop_randomization=True
    )

    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    if not best_model_path:
        early_stop_callback = EarlyStopping(monitor="val_loss", verbose=False, mode="min")
        lr_logger = LearningRateMonitor()  # log the learning rate

        trainer = pl.Trainer(
            max_epochs=100,
            gpus=0,
            weights_summary="top",
            callbacks=[lr_logger, early_stop_callback],
            log_every_n_steps=10,
            default_root_dir=saving_dir,
        )

        tft = TemporalFusionTransformer.from_dataset(
            training,
            hidden_size=128,
            attention_head_size=4,
            dropout=0.1,
            output_size=1,# 7 quantiles by default
            loss=MAPE(),
            log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        )
        # print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
    
    best_model_path = retriveBestModelPath(saving_dir)
    tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    
    return tft, val_dataloader

def DEEPAR(
    data, 
    training_cutoff, 
    target, 
    group, 
    max_encoder_length, 
    max_prediction_length, 
    time_varying_known_categoricals,
    time_varying_unknown_categoricals, 
    time_varying_known_reals, 
    batch_size,
    saving_dir,
):
    best_model_path = retriveBestModelPath(saving_dir)

    data[time_varying_known_categoricals] = data[time_varying_known_categoricals].astype(str).astype("category")
    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=target,
        group_ids=group,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_unknown_categoricals=time_varying_unknown_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=[target],
        target_normalizer=GroupNormalizer(groups=group),
        add_relative_time_idx=False,
        add_target_scales=True,
        randomize_length=None,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, 
        data, 
        min_prediction_idx=training.index.time.max() + 1,
        stop_randomization=True
    )

    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    if not best_model_path:
        early_stop_callback = EarlyStopping(monitor="val_loss", verbose=False, mode="min")
        lr_logger = LearningRateMonitor()  # log the learning rate

        trainer = pl.Trainer(
            max_epochs=100,
            gpus=0,
            weights_summary="top",
            callbacks=[lr_logger, early_stop_callback],
            log_every_n_steps=10,
            default_root_dir=saving_dir,
        )

        deep_ar = DeepAR.from_dataset(
            training,
            hidden_size=128,
            dropout=0.1,
            loss=NormalDistributionLoss(),
            log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
            log_val_interval=3,
        )
        # print(f"Number of parameters in network: {deep_ar.size()/1e3:.1f}k")

        trainer.fit(
            deep_ar,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

    best_model_path = retriveBestModelPath(saving_dir)
    deep_ar = DeepAR.load_from_checkpoint(best_model_path)

    return deep_ar, val_dataloader