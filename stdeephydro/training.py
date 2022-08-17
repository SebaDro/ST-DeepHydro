import logging
import os

from stdeephydro import common
from stdeephydro import config
from stdeephydro import dataset
from stdeephydro import ioutils
from stdeephydro import models
from stdeephydro import monitoring
from stdeephydro import processing
from stdeephydro import evaluation
from stdeephydro import dataloader
from typing import List, Tuple

logger = logging.getLogger(__name__)


def run_data_preparation(cfg: config.Config, data_loader_list: List[dataloader.HydroDataLoader],
                         processor_list: List[processing.AbstractProcessor], basin: str, out_dir: str, dry_run: bool)\
        -> Tuple[List[dataset.HydroDataset], List[dataset.HydroDataset], List[dataset.HydroDataset]]:
    """
    Performs data loading and data preprocessing steps for training, validation and test data

    Parameters
    ----------
    cfg: config.Config
        Training runtime configuration
    data_loader_list: List of dataloader.HydroDataLoader
        Used for loading training, validation and test datasets
    processor_list: List of dataloader.HydroDataLoader
        Used for processing training, validation and test datasets
    basin: List of str
        List of basin IDs
    out_dir: str
        Path to the working directory, which will be used to persist scaling parameters.
    dry_run: bool
        Indicates a dry run. If true, scaling parameters of a fitted processor won't be stored.
    Returns
    -------
    Tuple[List[dataset.HydroDataset], List[dataset.HydroDataset], List[dataset.HydroDataset]]
        One or more datasets that will be used for model training and evaluation

    """
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
            if not dry_run:
                processor.scaling_params[0].to_netcdf(os.path.join(out_dir, f"scaling_params_{i:02d}_min.nc"),
                                                      engine="h5netcdf")
                processor.scaling_params[1].to_netcdf(os.path.join(out_dir, f"scaling_params_{i:02d}_max.nc"),
                                                      engine="h5netcdf")
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


def run_model_training(cfg: config.Config, ds_train_list: List[dataset.HydroDataset],
                       ds_validation_list: List[dataset.HydroDataset], out_dir=None) -> models.AbstractModel:
    """
    Builds, compiles and fits a model to a training dataset. Some model architectures require multiple inputs for
    training and validation. Therefore, a list of datasets have to be passed to this method.

    Parameters
    ----------
    cfg: config.Config
        Model training runtime config
    ds_train_list: List of dataset.HydroDataset
        Datasets used as inputs for model training
    ds_validation_list: List of dataset.HydroDataset
        Datasets used for model validation during training
    out_dir: str
        Path to the directory, which will be used for storing intermediate results (e.g., training checkpoints) during
        training

    Returns
    -------
    models.AbstractModel
        The trained model
    """
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


def run_evaluation(model, ds_test_list: List[dataset.HydroDataset], processor_list: List[processing.AbstractProcessor],
                   target_var: str, basin: str) -> evaluation.Evaluation:
    """
    Performs model evaluation on a test dataset. Some model architectures require multiple inputs. Therefore, a list of
    datasets have to be passed to this method which will be used as test inputs.

    Parameters
    ----------
    model: models.Model
        A trained model that should be evaluated
    ds_test_list: List of dataset.HydroDataset
        Datasets used as test inputs for model evaluation
    processor_list: List of processing.AbstractProcessor
        Used for processing each dataset
    target_var: str
        Name of the prediction target variable
    basin: str
        ID of the basin for which the model will be evaluated

    Returns
    -------
    evaluation.Evaluation
        Evaluation metrics

    """
    logger.info("Start evaluating model...")
    try:
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
    except ValueError as e:
        raise common.EvaluationError("Error during model evaluation due to an incorrect value.") from e


def run_training_and_evaluation(cfg: config.Config, dry_run: bool):
    """
    Runs model training and evaluation for the purpose of a rainfall runoff prediction in multiple basins. For each
    basin a separate model will be trained and evaluated. Evaluation results are stored within separate NetCDF files for
    each basin and in an overall NetCDF file that folds evaluation metrics for all basins.

    Parameters
    ----------
    cfg: config.Config
        Runtime configuration for model training and evaluation
    dry_run: bool
        Indicates a dry run. If true, no intermediate or evaluation results are stored.

    """
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
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        try:
            logger.info(f"Prepare data for basin {basin}.")
            ds_train_list, ds_validation_list, ds_test_list = run_data_preparation(cfg, data_loader_list, processor_list,
                                                                                   basin, out_dir, dry_run)
            model = run_model_training(cfg, ds_train_list, ds_validation_list, out_dir)
            eval_res = run_evaluation(model, ds_test_list, processor_list, cfg.data_config.streamflow_cfg.variables[0],
                                      basin)

            if not dry_run:
                if cfg.general_config.save_model:
                    storage_path = model.save_model(out_dir, True)
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
            logger.exception(f"Evaluation stopped for basin {basin} due to an unexpected error that occurred during"
                             f" evaluating the model.")
    if not dry_run:
        res_out_path = common_eval_res.save(work_dir)
        logger.info(f"Stored evaluation results for all basins: '{res_out_path}'.")
