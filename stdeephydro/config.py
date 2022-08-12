import logging
import yaml
from typing import Union, List

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    pass


class GeneralConfig:
    """
    Holds general configuration parameters that control some runtime configuration parameters

    Parameters
    ----------
    name: str
        Name for the run
    output_dir: str
        Path to the directory that is used for storing outputs
    save_checkpoints: bool
        Indicates whether to store checkpoints during training or not. If true, tf.keras.callbacks.ModelCheckpoint
        is used for training.
    save_model: bool
        Indicates whether to save the model after training or not.
    log_tensorboard_events: bool
        Indicates whether to log events during training for Tensorboard analysis. If true,
        tf.keras.callbacks.TensorBoard is used for model training
    logging_config: str
        Path to a logging configuration file
    """

    def __init__(self, name, output_dir, save_checkpoints, save_model, log_tensorboard_events, logging_config,
                 seed=None):
        self.__name = name
        self.__output_dir = output_dir
        self.__save_checkpoints = save_checkpoints
        self.__save_model = save_model
        self.__log_tensorboard_events = log_tensorboard_events
        self.__logging_config = logging_config
        self.__seed = seed

    @property
    def name(self):
        return self.__name

    @property
    def output_dir(self):
        return self.__output_dir
    
    @property
    def save_checkpoints(self):
        return self.__save_checkpoints
    
    @property
    def save_model(self):
        return self.__save_model

    @property
    def log_tensorboard_events(self):
        return self.__log_tensorboard_events

    @property
    def logging_config(self):
        return self.__logging_config

    @property
    def seed(self):
        return self.__seed


class DatasetConfig:
    """
    Holds parameters that define a dataset timespan

    Parameters
    ----------
    start_date: str
        Start date string in the format yyyy-MM-dd
    end_date: str
        End date string in the format yyyy-MM-dd
    """

    def __init__(self, start_date: str, end_date: str):
        self.__start_date = start_date
        self.__end_date = end_date

    @property
    def start_date(self):
        return self.__start_date

    @property
    def end_date(self):
        return self.__end_date


class DataTypeConfig:
    """
    Configuration parameters that define a certain dataset type (e.g., CAMELS-US or Daymet)

    Parameters
    ----------
    data_dir: str
        Path to the data directory
    data_type: str
        Data type. Supported: 'Daymet', 'CAMELS-US'
    variables
        List of variable names

    """

    def __init__(self, data_dir: str, data_type: str, variables: list):
        self.__data_dir = data_dir
        self.__data_type = data_type
        self.__variables = variables

    @property
    def data_dir(self):
        return self.__data_dir

    @property
    def data_type(self):
        return self.__data_type

    @property
    def variables(self):
        return self.__variables


class DataConfig:
    """
    Holds configuration parameters required for creating training, validation and test datasets from forcings and
    streamflow data.

    Parameters
    ----------
    basins_file: str
        Path to basins file
    forcings_cfg: List of DataTypeConfig
        Configuration parameters for one or more forcings datasets
    streamflow_cfg: DataTypeConfig
        Configuration parameters for forcings data
    training_cfg: DatasetConfig
        Configuration parameters for the training dataset
    validation_cfg: DatasetConfig
        Configuration parameters for the validation dataset
    test_cfg: DatasetConfig
        Configuration parameters for the test dataset
    """
    def __init__(self, basins_file: str, forcings_cfg: List[DataTypeConfig], streamflow_cfg: DataTypeConfig,
                 training_cfg: DatasetConfig, validation_cfg: DatasetConfig, test_cfg: DatasetConfig):
        self.__basins_file = basins_file
        self.__forcings_cfg = forcings_cfg
        self.__streamflow_cfg = streamflow_cfg
        self.__training_cfg = training_cfg
        self.__validation_cfg = validation_cfg
        self.__test_cfg = test_cfg

    @property
    def basins_file(self):
        return self.__basins_file

    @property
    def forcings_cfg(self):
        return self.__forcings_cfg

    @property
    def streamflow_cfg(self):
        return self.__streamflow_cfg

    @property
    def training_cfg(self):
        return self.__training_cfg

    @property
    def validation_cfg(self):
        return self.__validation_cfg

    @property
    def test_cfg(self):
        return self.__test_cfg


class ModelConfig:
    """
    Configuration parameters that define the model architecture and control the training process

    Parameters
    ----------
    model_type: str
        Type of the model. Supported: 'lstm', 'cnn-lstm', 'multi-cnn-lstm', 'convlstm', 'conv3d'
    timesteps: List of int
        Timesteps for each input dataset
    offset: int:
        Prediction offset for the target variable(s)
    loss: List of str
        List of loss functions to use for training
    metrics: List of str
        List of metrics for validation and evaluation
    optimizer: str
        Optimizer to use for training
    epochs: int
        Number of training epochs
    batch_size: int
        Batch size to use for training
    multi_output: bool
        Indicates whether the model should predict multiple target variables or only one
    params: dict
        Additional model specific configuration parameters. E.g., a combined CNN-LSTM models required other
        config parameter then a simple LSTM model.
    """
    def __init__(self, model_type: str, timesteps: Union[int, List[int]], offset: int, loss: list, metrics: list,
                 optimizer: str, epochs: int, batch_size: int, multi_output: bool, params: dict = None):
        self.__model_type = model_type
        self.__timesteps = timesteps
        self.__offset = offset
        self.__loss = loss
        self.__metrics = metrics
        self.__optimizer = optimizer
        self.__epochs = epochs
        self.__batch_size = batch_size
        self.__multi_output = multi_output
        self.__params = params

    @property
    def model_type(self):
        return self.__model_type

    @property
    def timesteps(self):
        return self.__timesteps

    @property
    def offset(self):
        return self.__offset

    @property
    def loss(self):
        return self.__loss

    @property
    def metrics(self):
        return self.__metrics

    @property
    def optimizer(self):
        return self.__optimizer

    @property
    def epochs(self):
        return self.__epochs

    @property
    def batch_size(self):
        return self.__batch_size

    @property
    def multi_output(self):
        return self.__multi_output

    @property
    def params(self):
        return self.__params


class Config:
    """
    Wrapper class for configuration parameters

    Parameters
    ----------
    general_config: GeneralConfig
        General configuration parameters
    data_config: DataConfig
        Dataset related configuration parameters
    model_config: ModelConfig
        Model related configuration parameters
    """

    def __init__(self, general_config: GeneralConfig, data_config: DataConfig, model_config: ModelConfig):
        self.__general_config = general_config
        self.__data_config = data_config
        self.__model_config = model_config

    @classmethod
    def from_dict(cls, cfg_dict: dict):
        """
        Creates a Config instance from a dictionary.

        Parameters
        ----------
        cfg_dict: dict
            Dictionary containing config parameters

        Returns
        -------
        Config
            Instantiates a Config object

        """
        try:
            return cls(create_general_config(cfg_dict["general"]), create_data_config(cfg_dict["data"]),
                       create_model_config(cfg_dict["model"]))
        except KeyError as e:
            raise ConfigError("Could not create configuration due to invalid config parameters.") from e

    @property
    def general_config(self):
        return self.__general_config

    @property
    def data_config(self):
        return self.__data_config

    @property
    def model_config(self):
        return self.__model_config


def create_general_config(cfg: dict) -> GeneralConfig:
    """
    Creates a GeneralConfig instance from a dictionary

    Parameters
    ----------
    cfg: dict
        Dict that holds dataset specific config parameters

    Returns
    -------
    GeneralConfig
        Object containing general config parameters that

    """
    seed = cfg["seed"] if "seed" in cfg else None
    return GeneralConfig(cfg["name"], cfg["outputDir"], cfg["saveModel"], cfg["saveCheckpoints"],
                         cfg["logTensorboardEvents"], cfg["loggingConfig"], seed)


def create_dataset_config(cfg: dict) -> DatasetConfig:
    """
    Create a DatasetConfig instance from a dictionary

    Parameters
    ----------
    cfg: dict
        Dict that holds dataset specific config parameters

    Returns
    -------
    DatasetConfig
        Object containing config parameters that define a certain (training, validation or test) dataset

    """
    return DatasetConfig(cfg["startDate"], cfg["endDate"])


def create_dataype_config(cfg: dict) -> DataTypeConfig:
    """
    Create a DataTypeConfig instance from a dictionary

    Parameters
    ----------
    cfg: dict
        Dict that holds specific config parameters for a reading data of a certain type (forcings or streamflow)

    Returns
    -------
    DataTypeConfig
        Object containing config parameters that define a certain data type which can be used for reading it from
        a file

    """
    return DataTypeConfig(cfg["dir"], cfg["type"], cfg["variables"])


def create_data_config(cfg: dict) -> DataConfig:
    """
    Create a DataConfig instance from a dictionary

    Parameters
    ----------
    cfg: dict
        Dict that holds the data config parameters

    Returns
    -------
    DataConfig
        Object containing config parameters controlling the reading of streamflow and forcing datasets

    """
    return DataConfig(cfg["basinsFile"], [create_dataype_config(c) for c in cfg["forcings"]],
                      create_dataype_config(cfg["streamflow"]), create_dataset_config(cfg["training"]),
                      create_dataset_config(cfg["validation"]), create_dataset_config(cfg["test"]))


def create_model_config(cfg: dict) -> ModelConfig:
    """
    Reads parameters from a config dict related to model specific configurations

    Parameters
    ----------
    cfg: dict
        Dict that holds the config parameters

    Returns
    -------
    ModelConfig
        Object containing config parameters controlling model architecture

    """
    return ModelConfig(cfg["type"], cfg["timesteps"], cfg["offset"], cfg["loss"], cfg["metrics"],
                       cfg["optimizer"], cfg["epochs"], cfg["batchSize"], cfg["multiOutput"], cfg["params"])


def create_config(cfg: dict) -> Config:
    """
    Reads parameters from a config dict

    Parameters
    ----------
    cfg: dict
        Dict that holds the config parameters

    Returns
    -------
    Config
        Object containing config parameters

    """
    try:
        return Config(create_general_config(cfg["general"]), create_data_config(cfg["data"]),
                      create_model_config(cfg["model"]))
    except KeyError as e:
        raise ConfigError("Could not create configuration due to invalid config parameters.") from e


def read_config(path: str) -> dict:
    """
    Reads configuration parameters from a *.yml file.

    Parameters
    ----------
    path: str
        Path to the configuration file

    Returns
    -------
    dict
        Dictionary containing config parameters

    """

    try:
        with open(path, 'r') as stream:
            config = yaml.safe_load(stream)
            return config
    except yaml.YAMLError as e:
        raise ConfigError("Could not read configuration due to invalid formating.") from e
    except KeyError as e:
        raise ConfigError("Could not read configuration due to missing config parameters.") from e
    except IOError as e:
        raise ConfigError("Error while trying toe read configuration file.") from e

