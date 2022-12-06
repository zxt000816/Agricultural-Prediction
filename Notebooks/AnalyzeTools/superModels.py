import warnings, torch, shutil
warnings.filterwarnings("ignore")

import pytorch_lightning as pl
pl.seed_everything(123)

from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, DeepAR, RecurrentNetwork, GroupNormalizer
from pytorch_forecasting.metrics import MAPE, NormalDistributionLoss, QuantileLoss, SMAPE

from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from .prepare import retriveBestModelPath, save_best_params
from .paramsTuner import optimize_hyperparameters_for_RNN, optimize_hyperparameters_for_DeepAR

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
    cell='LSTM',
    training_params={},
):
    best_model_path = retriveBestModelPath(saving_dir)
    data[time_varying_known_categoricals] = data[time_varying_known_categoricals].astype(str).astype("category")
    max_epochs = training_params.get('max_epochs') if training_params.get('max_epochs') else 100
    n_trials = training_params.get('n_trials') if training_params.get('n_trials') else 30
    
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
        categorical_encoders={
            "month": NaNLabelEncoder().fit(data.month),
            "week": NaNLabelEncoder().fit(data.week)
        },
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, 
        data, 
        min_prediction_idx=training.index.time.max() + 1 + max_encoder_length - 1,
        stop_randomization=True
    )

    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)
    
    if not best_model_path:
        study = optimize_hyperparameters_for_RNN(
            train_dataloader,
            val_dataloader,
            model_path=saving_dir,
            n_trials=n_trials,
            max_epochs=max_epochs,
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=(8, 128),
            learning_rate_range=(0.001, 0.1),
            dropout_range=(0.1, 0.3),
            trainer_kwargs=dict(limit_train_batches=30),
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=False,
            log_dir=saving_dir
        )

        best_params = study.best_params
        shutil.rmtree(saving_dir)

        early_stop_callback = EarlyStopping(monitor="val_loss", verbose=False, mode="min")
        lr_logger = LearningRateMonitor()

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=0,
            enable_model_summary=False,
            callbacks=[lr_logger, early_stop_callback],
            log_every_n_steps=10,
            check_val_every_n_epoch=3,
            default_root_dir=saving_dir,
        )

        model = RecurrentNetwork.from_dataset(
            training,
            cell_type=cell,
            hidden_size=best_params.get('hidden_size'),
            rnn_layers=best_params.get('rnn_layers'),
            learning_rate=best_params.get('learning_rate'),
            dropout=best_params.get('dropout'),
            output_size=1,
            loss=MAPE(),
            log_interval=10
        )
    
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

        save_best_params(best_params, f"{saving_dir}/Best_{cell}.json")
    
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
    predict_type,
    **params,
):
    if predict_type == 'single' or predict_type == 'multiple':
        loss_function = MAPE()
        output_size = 1
    elif predict_type == 'multiple_interval':
        loss_function = QuantileLoss()
        output_size = params.get('output_size') if params.get('output_size') else 7
    
    best_model_path = retriveBestModelPath(saving_dir)
    data[time_varying_known_categoricals] = data[time_varying_known_categoricals].astype(str).astype("category")
    
    max_epochs = params.get('max_epochs') if params.get('max_epochs') else 50
    n_trials = params.get('n_trials') if params.get('n_trials') else 10
    hidden_size_range = params.get('hidden_size_range') if params.get('hidden_size_range') else (8, 128)
    hidden_continuous_size_range = params.get('hidden_continuous_size_range') if params.get('hidden_continuous_size_range') else (8, 128)
    attention_head_size_range = params.get('attention_head_size_range') if params.get('attention_head_size_range') else (1, 4)

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
        categorical_encoders={
            "month": NaNLabelEncoder().fit(data.month),
            "week": NaNLabelEncoder().fit(data.week)
        },
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, 
        data, 
        min_prediction_idx=training.index.time.max() + 1 + max_encoder_length - 1,
        stop_randomization=True
    )

    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    if not best_model_path:
        # create study
        study = optimize_hyperparameters(
            train_dataloader,
            val_dataloader,
            model_path=saving_dir,
            n_trials=n_trials,
            max_epochs=max_epochs,
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=hidden_size_range,
            hidden_continuous_size_range=hidden_continuous_size_range,
            attention_head_size_range=attention_head_size_range,
            learning_rate_range=(0.001, 0.1),
            dropout_range=(0.1, 0.3),
            trainer_kwargs=dict(limit_train_batches=30),
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
            log_dir=saving_dir
        )

        best_params = study.best_params
        shutil.rmtree(saving_dir)

        early_stop_callback = EarlyStopping(monitor="val_loss", verbose=False, mode="min")
        lr_logger = LearningRateMonitor()  # log the learning rate

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=0,
            enable_model_summary=False,
            callbacks=[lr_logger, early_stop_callback],
            log_every_n_steps=10,
            check_val_every_n_epoch=3,
            default_root_dir=saving_dir,
        )

        tft = TemporalFusionTransformer.from_dataset(
            training,
            hidden_size=best_params.get('hidden_size'),
            attention_head_size=best_params.get('attention_head_size'),
            hidden_continuous_size=best_params.get('hidden_continuous_size'),
            learning_rate=best_params.get('learning_rate'),
            dropout=best_params.get('dropout'),
            output_size=output_size,# 7 quantiles by default
            loss=loss_function,
            log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        )
        # print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

        trainer.fit(
            tft,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        save_best_params(best_params, f"{saving_dir}/Best_TFT.json")
    
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
    training_params={},
):
    best_model_path = retriveBestModelPath(saving_dir)
    data[time_varying_known_categoricals] = data[time_varying_known_categoricals].astype(str).astype("category")
    max_epochs = training_params.get('max_epochs') if training_params.get('max_epochs') else 100
    n_trials = training_params.get('n_trials') if training_params.get('n_trials') else 30

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
        categorical_encoders={
            "month": NaNLabelEncoder().fit(data.month),
            "week": NaNLabelEncoder().fit(data.week)
        },
    )

    validation = TimeSeriesDataSet.from_dataset(
        training, 
        data, 
        min_prediction_idx=training.index.time.max() + 1 + max_encoder_length - 1,
        stop_randomization=True
    )

    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    if not best_model_path:

        # create study
        study = optimize_hyperparameters_for_DeepAR(
            train_dataloader,
            val_dataloader,
            model_path=saving_dir,
            n_trials=n_trials,
            max_epochs=max_epochs,
            gradient_clip_val_range=(0.01, 1.0),
            hidden_size_range=(8, 128),
            learning_rate_range=(0.001, 0.1),
            dropout_range=(0.1, 0.3),
            trainer_kwargs=dict(limit_train_batches=30),
            reduce_on_plateau_patience=4,
            use_learning_rate_finder=False,
            log_dir=saving_dir
        )

        best_params = study.best_params
        shutil.rmtree(saving_dir)

        early_stop_callback = EarlyStopping(monitor="val_loss", verbose=False, mode="min")
        lr_logger = LearningRateMonitor()  # log the learning rate

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            gpus=0,
            enable_model_summary=False,
            callbacks=[lr_logger, early_stop_callback],
            log_every_n_steps=10,
            check_val_every_n_epoch=3, 
            default_root_dir=saving_dir,
        )

        deep_ar = DeepAR.from_dataset(
            training,
            hidden_size=best_params.get("hidden_size"),
            rnn_layers=best_params.get('rnn_layers'),
            learning_rate=best_params.get('learning_rate'),
            dropout=best_params.get("dropout"),
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

        save_best_params(best_params, f"{saving_dir}/Best_DeepAR.json")

    best_model_path = retriveBestModelPath(saving_dir)
    deep_ar = DeepAR.load_from_checkpoint(best_model_path)

    return deep_ar, val_dataloader