import logging
import yaml

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    pass


class GeneralConfig:
    def __init__(self, name, output_dir, save_checkpoints, save_model):
        self.__name = name
        self.__output_dir = output_dir
        self.__save_checkpoints = save_checkpoints
        self.__save_model = save_model

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


class DatasetConfig:
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
    """
    def __init__(self, basins_file: str, forcings_cfg: DataTypeConfig, streamflow_cfg: DataTypeConfig,
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
    def __init__(self, model_type: str, timesteps: int, offset: int, loss: list, metrics: list, optimizer: str,
                 epochs: int, batch_size: int, multi_output: bool, params: dict = None):
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

    Parameters
    ----------
    cfg: dict
        Dict that holds dataset specific config parameters

    Returns
    -------
    GeneralConfig
        Object containing general config parameters that

    """
    return GeneralConfig(cfg["name"], cfg["outputDir"], cfg["saveModel"], cfg["saveCheckpoints"])


def create_dataset_config(cfg: dict) -> DatasetConfig:
    """

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
    Reads parameters from a config dict related to data specific configurations.

    Parameters
    ----------
    cfg: dict
        Dict that holds the data config parameters

    Returns
    -------
    DataConfig
        Object containing config parameters controlling the reading of streamflow and forcing datasets

    """
    return DataConfig(cfg["basinsFile"], create_dataype_config(cfg["forcings"]),
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

