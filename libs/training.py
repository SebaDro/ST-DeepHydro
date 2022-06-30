import logging
import os

from libs import common
from libs import config
from libs import ioutils
from libs import models
from libs import monitoring
from libs import processing
from libs import evaluation
from libs import dataloader
from typing import List


logger = logging.getLogger(__name__)


def run_data_preparation(cfg: config.Config, data_loader_list: List[dataloader.HydroDataLoader],
                         processor_list: List[processing.AbstractProcessor], basin: str):
    for f_cfg in cfg.data_config.forcings_cfg:
        logger.info(f"Load '{f_cfg}' forcings.")
    logger.info(f"Load '{cfg.data_config.streamflow_cfg.data_type}' streamflow data.")
    logger.debug(f"Split datasets: training start date {cfg.data_config.training_cfg.start_date}, "
                 f"training end date {cfg.data_config.training_cfg.end_date}; "
                 f"validation start date {cfg.data_config.validation_cfg.start_date}, "
                 f"validation end date {cfg.data_config.validation_cfg.end_date}; "
                 f"test start date {cfg.data_config.test_cfg.start_date}, "
                 f"test end date {cfg.data_config.test_cfg.end_date}")
    ds_train_list = []
    ds_validation_list = []
    ds_test_list = []
    try:
        for i, data_loader in enumerate(data_loader_list):
            ds_train = data_loader.load_single_dataset(cfg.data_config.training_cfg.start_date,
                                                       cfg.data_config.training_cfg.end_date,
                                                       basin)
            ds_validation = data_loader.load_single_dataset(cfg.data_config.validation_cfg.start_date,
                                                            cfg.data_config.validation_cfg.end_date,
                                                            basin)
            ds_test = data_loader.load_single_dataset(cfg.data_config.test_cfg.start_date,
                                                      cfg.data_config.test_cfg.end_date, basin)
            processor = processor_list[i]
            logger.info(f"Preprocess datasets using '{type(processor)}'.")
            processor.fit(ds_train)
            ds_train_list.append(processor.process(ds_train))
            ds_validation_list.append(processor.process(ds_validation))
            ds_test_list.append(processor.process(ds_test))
        return ds_train_list, ds_validation_list, ds_test_list
    except KeyError as e:
        raise common.DataPreparationError("Error during data preparation due to failing variable access.") from e
    except ValueError as e:
        raise common.DataPreparationError("Error during data preparation due to an incorrect value.") from e
    except TypeError as e:
        raise common.DataPreparationError("Error during data preparation due to an incorrect value type.") from e
    except IOError as e:
        raise common.DataPreparationError("Error during data preparation while trying to read a file.") from e


def run_build_and_compile_model(cfg: config.Config, ds_train_list, ds_validation_list, out_dir=None):
    logger.info(f"Build model of type '{cfg.model_config.model_type}'.")
    try:
        model = models.factory(cfg.model_config)
        shape_list = [(cfg.model_config.timesteps[i],) + ds.get_input_shape() for i, ds in enumerate(ds_train_list)]
        model.build(input_shape=shape_list) if len(shape_list) > 1 else model.build(input_shape=shape_list[0])
        model.model.summary(print_fn=logger.debug)

        monitor = None
        if out_dir is not None:
            logger.info(f"Configure training monitor: output_dir={out_dir}, "
                        f"save_checkpoints={cfg.general_config.save_checkpoints}, "
                        f"save_model={cfg.general_config.save_model}")
            monitor = monitoring.TrainingMonitor(out_dir, cfg.general_config.save_checkpoints,
                                                 cfg.general_config.save_model)

        logger.info("Start fitting model to training data...")
        model.compile_and_fit(ds_train_list, ds_validation_list, monitor=monitor)
        logger.info("Finished fitting model to training data.")
        return model
    except ValueError as e:
        raise common.TrainingError("Error during model fitting due to an incorrect value.") from e
    except TypeError as e:
        raise common.TrainingError("Error during model fitting due to an incorrect value type.") from e
    except ArithmeticError as e:
        raise common.TrainingError("Error during model fitting due to an unsupported calculation.") from e


def run_evaluation(model, ds_test_list, processor_list, target_var, basin) -> evaluation.Evaluation:
    logger.info("Start evaluating model...")
    ds_prediction = model.predict(ds_test_list, basin, as_dataset=True, remove_nan=False)
    result = model.evaluate(ds_test_list, True, basin)

    # Rescale predictions and observation to the origin uom
    processor = processor_list[0]
    ds_prediction = processor.rescale(ds_prediction)
    ds_observation = processor.rescale(ds_test_list[0].timeseries)

    ds_eval = evaluation.calc_evaluation(ds_observation, ds_prediction, target_var, basin)

    eval_res = evaluation.Evaluation()
    eval_res.append_evaluation_results(result)
    eval_res.append_evaluation_results(ds_eval)
    logger.info(f"Finished evaluating model. NSE={eval_res.ds_results.sel(basin=basin).nse.values}")
    return eval_res


def run_training(cfg: config.Config, dry_run):
    work_dir = None
    if not dry_run:
        work_dir = ioutils.create_out_dir(cfg.general_config.output_dir, cfg.general_config.name)
        logger.info(f"Created working directory '{work_dir}'.")

    data_loader_list = []
    processor_list = []
    for f_cfg in cfg.data_config.forcings_cfg:
        data_loader = dataloader.HydroDataLoader.from_config(cfg.data_config.basins_file, f_cfg, cfg.data_config.streamflow_cfg)
        data_loader_list.append(data_loader)
        processor = processing.DefaultDatasetProcessor()
        processor_list.append(processor)
    common_eval_res = evaluation.Evaluation()

    with open(cfg.data_config.basins_file, 'r') as file:
        basins = [line.strip() for line in file.readlines()]

    logger.info(f"Training will be run for {len(basins)} basins.")
    logger.debug(f"Basins: {basins}.")

    for basin in basins:
        logger.info(f"Start training for basin {basin}.")
        out_dir = None
        if work_dir is not None:
            out_dir = os.path.join(work_dir, basin)
        try:
            logger.info(f"Prepare data for basin {basin}.")
            ds_train_list, ds_validation_list, ds_test_list = run_data_preparation(cfg, data_loader_list, processor_list, basin)
            model = run_build_and_compile_model(cfg, ds_train_list, ds_validation_list, out_dir)
            eval_res = run_evaluation(model, ds_test_list, processor_list, cfg.data_config.streamflow_cfg.variables[0], basin)

            if not dry_run:
                if cfg.general_config.save_model:
                    storage_path = model.save_model(out_dir)
                    logger.info(f"Stored model: '{storage_path}'.")
                common_eval_res.append_evaluation_results(eval_res)
                res_out_path = eval_res.save(out_dir)
                logger.info(f"Stored evaluation results: '{res_out_path}'.")
            logger.info(f"Successfully finished training for basin {basin}.")
        except config.ConfigError:
            logger.exception(f"Training stopped for basin {basin} due to incorrect configuration parameters.")
        except common.DataPreparationError:
            logger.exception(f"Training stopped for basin {basin} due to an error that occurred while preparing the"
                             f" datasets.")
        except common.TrainingError:
            logger.exception(f"Training stopped for basin {basin} due to an unexpected error that occurred during"
                             f" fitting the model.")
        except common.EvaluationError:
            logger.exception(f"Training stopped for basin {basin} due to an unexpected error that occurred during"
                             f" evaluating the model.")
    if not dry_run:
        res_out_path = common_eval_res.save(work_dir)
        logger.info(f"Stored evaluation results for all basins: '{res_out_path}'.")
