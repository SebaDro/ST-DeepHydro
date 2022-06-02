import logging
from libs import config
from libs import ioutils
from libs import models
from libs import monitoring
from libs import processing
from libs import evaluation
from libs import dataloader
import os

logger = logging.getLogger(__name__)


def run_data_preparation(cfg: config.Config, data_loader: dataloader.HydroDataLoader,
                     processor: processing.AbstractProcessor, basin: str):
    logger.info(f"Load '{cfg.data_config.forcings_cfg.data_type}' forcings and "
                f"'{cfg.data_config.streamflow_cfg.data_type}' streamflow data.")
    logger.debug(f"Split datasets: training start date {cfg.data_config.training_cfg.start_date}, "
                 f"training end date {cfg.data_config.training_cfg.end_date}; "
                 f"validation start date {cfg.data_config.validation_cfg.start_date}, "
                 f"validation end date {cfg.data_config.validation_cfg.end_date}; "
                 f"test start date {cfg.data_config.test_cfg.start_date}, "
                 f"test end date {cfg.data_config.test_cfg.end_date}")
    ds_train = data_loader.load_single_dataset(cfg.data_config.training_cfg.start_date,
                                               cfg.data_config.training_cfg.end_date,
                                               basin)
    ds_validation = data_loader.load_single_dataset(cfg.data_config.validation_cfg.start_date,
                                                    cfg.data_config.validation_cfg.end_date,
                                                    basin)
    ds_test = data_loader.load_single_dataset(cfg.data_config.test_cfg.start_date,
                                              cfg.data_config.test_cfg.end_date, basin)

    logger.info("Preprocess datasets.")
    processor.fit(ds_train)
    ds_train = processor.process(ds_train)
    ds_validation = processor.process(ds_validation)
    ds_test = processor.process(ds_test)

    return ds_train, ds_validation, ds_test


def run_build_and_compile_model(cfg: config.Config, ds_train, ds_validation, work_dir, basin):
    logger.info(f"Build model of type '{cfg.model_config.model_type}'.")
    model = models.factory(cfg.model_config)
    model.build(input_shape=(cfg.model_config.timesteps,) + ds_train.get_input_shape())
    logger.debug(model.model.summary())

    out_dir = os.path.join(work_dir, basin)
    logger.info(f"Configure training monitor: output_dir={out_dir}, "
                f"save_checkpoints={cfg.general_config.save_checkpoints}, "
                f"save_model={cfg.general_config.save_model}")
    monitor = monitoring.TrainingMonitor(out_dir, cfg.general_config.save_checkpoints,
                                         cfg.general_config.save_model)

    logger.info("Start fitting model to training data...")
    model.compile_and_fit(ds_train, ds_validation, monitor=monitor)
    logger.info("Finished fitting model to training data.")
    return model


def run_evaluation(model, ds_test, processor, eval_res, basin):
    logger.info("Start evaluating model...")
    ds_prediction = model.predict(ds_test, basin, as_dataset=True, remove_nan=False)
    result = model.evaluate(ds_test, True, basin)

    ds_prediction = processor.rescale(ds_prediction)
    ds_observation = processor.rescale(ds_test.timeseries)

    eval_res.calc_evaluation(ds_observation, ds_prediction, "streamflow", basin, True)
    eval_res.append_evaluation_results(result)
    logger.info("Finished evaluating model.")


def run_training(cfg: config.Config):
    work_dir = ioutils.create_out_dir(cfg.general_config.output_dir, cfg.general_config.name)
    logger.info(f"Created working directory '{work_dir}'.")

    data_loader = dataloader.HydroDataLoader.from_config(cfg.data_config)
    processor = processing.DefaultDatasetProcessor()
    eval_res = evaluation.Evaluation()

    with open(cfg.data_config.basins_file, 'r') as file:
        basins = [line.strip() for line in file.readlines()]

    logger.info(f"Training will be run for {len(basins)}.")
    logger.debug(f"Basins: {basins}.")

    for basin in basins:
        logger.info(f"Start data preparation for basin {basin}.")
        try:
            ds_train, ds_validation, ds_test = run_data_preparation(cfg, data_loader, processor, basin)
            model = run_build_and_compile_model(cfg, ds_train, ds_validation, work_dir, basin)
            run_evaluation(model, ds_test, processor, eval_res, basin)
        except ValueError:
            logging.exception(f"Training stopped for basin {basin} due to non valid timeseries samples.")
        except Exception:
            logging.exception(f"Training stopped for basin {basin} due to an unexpected error that occurred during"
                              f"training.")
    res_out_path = eval_res.save(cfg.general_config.output_dir)
    logger.info(f"Stored evaluation results '{res_out_path}'.")
