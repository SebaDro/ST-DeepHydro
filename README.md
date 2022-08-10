# ST-DeepHydro
Python library for spatio-temporal aware hydrological modelling (especially, rainfall-runoff modelling) using deep learning.

This library facilitates the training of Neural Networks for spatio-temporal timeseries prediction. It is based on the
Deep Learning library Tensorflow and aims to support hydrological use cases. For this purpose, the library implements
different Neural Network architectures, with a special focus on learning spatio-temporal processes within catchments.
Model types comprise lumped and distributed models, which enable training on aggregated meteorological timeseries data
as well as spatially distributed forcings such as gridded datasets. In addition, the library comes with various data
loading mechanisms for common hydrometeorological datasets, such as Daymet and CAMELS-US, as well useful preprocessing
utilities for handling spatio-temporal data. 

To train and evaluate some models on your own hydrometeorological datasets for one or more catchments,this library
comes with a simple command line tool. You also can use this library for implementing your own deep learning
applications by using the already implemented models and data loading classes. The library is designed in a way that
also facilitates additional model or data loading and processing pipelines. To get started, just follow the documentation
below.

The library is inspired by the great NeuralHydrology package [[1]](#1) which has been used for various research aspects
regarding hydrological modelling. However, since NeuralHydrology mainly focuses on lumped models, the ST DeepHydro
package addresses needs for spatial distributed modelling.

## Get Started
### Requirements
To be prepared for using the _stdeephydro_ package for your own purposes, you first have to set up you local environment
by installing all required dependencies. The easiest way to do so is creating a virtual environment by using Conda.
+Make sure, your have installed [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or even
[Anaconda](https://docs.anaconda.com/) and create a new environment using the _environment.yml_ file that comes
with this repository:
```commandline
conda env create -f environment.yml
```

If you'd rather like to create a virtual environment with [venv](https://docs.python.org/3/library/venv.html) or 
[virtualenv](https://virtualenv.pypa.io/en/latest/), you can also use the _requirements.txt_ that is shipped
with this repository. Just create a virtual environment with your preferred tool and install all dependencies:
```commandline
python3 -m pip install -r requirements.txt 
```

### Installation
This package simply can be installed by using pip. Up to now, the package has not been published and uploaded to
PyPi, yet. However, you can install the latest version of the package, which is based on the master branch:
```commandline
python3 -m pip install git+https://github.com/SebaDro/st-deep-hydro.git
```

It is also possible to clone this repository to your local machine and develop your own models and data loading routines.
Finally, you can install the package from your local copy:
```commandline
python3 -m pip install -e .
```

The installation also makes a bash script (_run_training_) available within your environment.

## Data
The ST-DeepHydro library mainly focuses on training models for rainfall runoff predictions by using hydrometeorological
datasets. For this purpose, a variety of datasets are suitable to be used as training data. Though, especially CAMELS-US
and Daymet datasets has been widely proven as appropriate input datasets for hydrological modelling.

For loading different types of hydrometeorological datasets the [stdeephydro.dataloader](./stdeephydro/dataloader.py)
module comes with various dataloader implementations.


### CAMELS-US
The CAMELS-US dataset contains hydrometeorological timeseries data for 671 basins in the continuous United States [[2]](#2).
Meteorological products contain basin aggregated daily forcings from three different data sources (Daymet, Maurer and NLDAS). 
Daily streamflow data for 671 gauges comes from the United States Geological Survey National Water Information System.
You simply can download this large-sample dataset from the [NCAR website](https://ral.ucar.edu/solutions/products/camels).

To load CAMELS-US datasets use the `CamelsUsStreamflowDataLoader` and `CamelsUsForcingsDataLoader` classes. See their
documentation for further usage information. 

### Daymet
Daymet data contain gridded estimates of daily weather and climatology parameters at a 1 km x 1 km raster for North
America, Hawaii, and Puerto Rico [[3](#3), [4](#4)]. Daymet Version 3 and Version 4 data are provided by (ORNL DAAC)[https://daymet.ornl.gov/]
and can be via ORNL DAAC's Thematic Real-time Environmental Distributed Data Services (THREDDS). To download these
datasets for your preferred region and prepare it for model training, you might want to use the
[Daymet PyProcessing](https://github.com/SebaDro/daymet-pyprocessing) toolset.

The `DaymetDataLoader` class is able to load 1-dimensional (temporally distributed) as well as 2-dimensional 
(raster-based, spatio-temporally distributed) Daymet NetCDF data.

See it's documentation for further details.

## Models
To train neural networks for timeseries forecasting the ST-DeepHydro library implements different network architectures,
based on the Deep Learning framework Tensorflow. Although, these networks are intended to model rainfall-runoff
in river catchments, other hydrological modelling use-cases are conceivable. Model types comprise lumped and distributed
models. While lumped models are trained on aggregated meteorological timeseries data, distributed models have a special
focus on learning spatio-temporal catchment processes. This makes them suitable to be trained on spatio-temporal 
hydrometeorological datasets, i.e. timeseries of raster data.

The [stdeephydro.models](./stdeephydro/models.py) module contains all model implementations. Here, you will find a
variety of Tensorflow models for different use cases and data types.

### LSTM
`stdeephydro.models.LstmModel` builds a classical LSTM model, which is able to learn hydrological processes within
catchment areas from aggregated hydrometeorological input datasets. The model is applicable for rainfall-runoff
timeseries forecasting by predicting gauge streamflow.

The Tensorflow model comprises one or more stacked (hidden) LSTM layers with a fully connected layer on top for
predicting one or more target variables from timeseries inputs. 

#### LSTM Attributes:
Required values for `cfg.params`:
- `lstm`:
  - `hiddenLayers`: number of LSTM layers (int)
  - `units`: units for each LSTM layer (list of int, with the same length as hiddenLayers)
  - `dropout`: dropout for each LSTM layer (list of float, with the same length as hiddenLayers)

**Example:**
```yml
params:
  lstm:
    hiddenLayers: 2
    units:
      - 32
      - 32
    dropout:
      - 0.1
      - 0
```

### CNN-LSTM 
`stdeephydro.models.CnnLstmModel` builds a combination of [Convolutional Neural Network (CNN)](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)
and [Long short-term memory (LSTM)](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM) Tensorflow model.
The neural network architecture addresses needs to learn spatio-temporal processes within catchments from spatially
distributed (raster-based) timeseries data. Therefore, this model type can be trained on meteorological raster data to
forecast gauge streamflow or any other hydrological variables within river catchments. 

The idea of this model architecture is to extract features from a timeseries of 2-dimensional raster data by convolutional
operations at first. The extracted timeseries features then are passed to a stack of LSTM layer to predict one
or more target variables.

#### CNN-LSTM Attributes:
Required values for `cfg.params`:
- `cnn`:
  - `hiddenLayers`: number of time-distributed Conv2D layers (int). After each Conv2D layer follows a MaxPooling2D layer,
    except the last Conv2D layer, which has a GlobalMaxPooling2D on top.
  - `filters`: number of filters for each Conv2D layer (list of int, with the same length as hiddenLayers)
- `lstm`:
  - `hiddenLayers`: number of LSTM layers (int)
  - `units`: units for each LSTM layer (list of int, with the same length as hiddenLayers)
  - `dropout`: dropout for each LSTM layer (list of float, with the same length as hiddenLayers)

**Example:**
```yml
params:
  cnn:
    hiddenLayers: 3
    filters:
      - 8
      - 16
      - 32
  lstm:
    hiddenLayers: 2
    units:
      - 32
      - 32
    dropout:
      - 0.1
      - 0
```

### Multi Input CNN-LSTM
The `stdeephydro.models.MultiInputCnnLstmModel` class concatenates a combination of Convolutional Neural Network (CNN)
and Long short-term memory (LSTM), CNN-LSTM, with a classical LSTM Tensorflow model. With this architecture design the
neural network is able to process two input datasets that differ in its spatio-temporal dimensions. Hence, it is possible
to train the model with lumped meteorological long-term timeseries data as well as spatially-distributed short-term raster
data. 

The idea of this model is to enhance the capability of a classical LSTM model to predict target variables from
one-dimensional timeseries data but also considering spatial distributed timeseries data that are processed by a
CNN-LSTM part of the model. This approach adds enhanced spatial information to the model and limits computational efforts
for training the model at the same time.

#### Multi input CNN-LSTM Attributes:
Required values for `cfg.params`:
- cnn:
  - hiddenLayers: number of time-distributed Conv2D layers for the CNN-LSTM part of the model (int). After each Conv2D
    layer follows a MaxPooling2D layer, except the last Conv2D layer, which has a  GlobalMaxPooling2D on top.
  - filters: number of filters for each time-distributed Conv2D layer (list of int, with the same length as 
    hiddenLayers)
- lstm:
  - hiddenLayers: number of LSTM layers for both the LSTM and CNN-LSTM part of the model (int)
  - units: units for each LSTM layer (list of int, with the same length as hiddenLayers)
  - dropout: dropout for each LSTM layer (list of float, with the same length as hiddenLayers)

**Example:**
```yml
params:
  cnn:
    hiddenLayers: 3
    filters:
      - 8
      - 16
      - 32
  lstm:
    hiddenLayers: 2
    units:
      - 32
      - 32
    dropout:
      - 0.1
      - 0
```

### ConvLSTM
The `stdeephydro.models.ConvLstmModel` class builds a Convolutional LSTM model that maily builds up on
[Tensorflow ConvLSTM2D layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ConvLSTM2D). This architecture
is able to predict one or more target variables based on spatially distributed timeseries data. The neural network
processes timeseries of raster data with a stack of LSTM layers that perform convolutional operations by using
input-to-state and state-to-state transitions.

Up to now, the ConvLSTM model can be trained on meteorological raster data to predict one-dimensional or any other
hydrological variables, such as the CNN-LSTM model can do. However, originally ConvLSTM layers are intended for building
models that are able to produce raster-based predictions. This maybe implemented for future releases to support relevant
hydrological use cases.

#### ConvLSTM Attributes:
Required values for `cfg.params`:
- `cnn`:
  - `hiddenLayers`: number of ConvLSTM2D layers (int). After each ConvLSTM2D layer follows a MaxPooling3D layer, except
    the last ConvLSTM2D layer, which has a GlobalMaxPooling2D on top.
  - `filters`: number of filters for each Conv2D layer (list of int, with the same length as hiddenLayers)

**Example:**
```yml
cnn:
  hiddenLayers: 3
  filters:
    - 8
    - 16
    - 32
```

### Conv3D
The `stdeephydro.models.Conv3DModel` builds a Tensorflow model based multiple stacked [Conv3D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv3D)
and [MaxPooling3D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool3D) layers. Usually, intended to
process video data, it applies convolutional and max pooling operations on the spatial as well as the temporal dimension
of the input data. The model can be trained on a timeseries of meteorological raster data to predict gauge streamflow or
any other hydrological variables.

#### Conv3D Attributes:
Required values for `cfg.params`:
- `cnn`:
  - `hiddenLayers`: number of Conv3D layers (int). After each Conv3D layer follows a MaxPooling3D layer, except the last
    Conv3D layer, which has a GlobalMaxPooling3D on top.
  - `filters`: number of filters for each Conv3D layer (list of int, with the same length as hiddenLayers)

**Example:**
```yml
cnn:
  hiddenLayers: 3
  filters:
    - 8
    - 16
    - 32
```

## How to use
### Training
Model training and evaluating for multiple basins can be simply performed by using the _run_training_ command line tool.
This tool will be automatically available in your environment when installing this package.

Just run `python.exe .\run_training.py .\config\your-training-config.yml` to perform training according to your own
configuration.

For testing purposes you can add the `--dryrun` flag to this call: `python.exe .\run_training.py --dryrun .\config\your-training-config.yml`.
This causes no results such as model checkpoints or evaluation metrics to be stored during the run.

#### Data preparation
To train a model, one of the supported datasets mentioned in the [Data section](#data) is required. So, before you
start with model training, make sure that you have properly prepared all the datasets you want to use for training:
1. Download one or more of the supported datasets. You'll need a streamflow datasets and a forcings dataset.
2. Place all datasets within a separate folder. It is required, that for each basin you want to train a model for, a
corresponding dataset file exists int the data folder, which has the basin ID in its file name.
3. Forcing files and streamflow files should be placed in different folders, if your datasets does not contain both
variable sets jointly.
4. Create a text file (e.b. "basins.txt") that lists all basins you want to perform model training and evaluating on.
Each basin ID must be on a separate line. There is an [example file](./config/basins.txt) within this repository. Make
sure, that the data folder contains corresponding files for each basin you list in the basins file.
#### Configuration
Several training aspects, such as the neural network architecture, dataset types or number of training epochs, can be
customized by providing a configuration file. The [./config](./config) folder comes with several examples for such a file.
In addition, this section describes all configuration parameters that are supported at the moment.

##### General Parameters
General configuration parameters must be defined under the `general`key:  

| Config Parameter       | Type      | Description                                                                                                                                                                                          |
|------------------------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| _name_                 | `string`  | Name of the experiment. This name will be used for prefixing some of the outputs.                                                                                                                    |
| _logTensorboardEvents_ | `boolean` | Indicates whether to log events during training for Tensorboard or not.                                                                                                                              |
| _loggingConfig_        | `string`  | Path to a logging configuration file. This must be a YAML file according to the [Python logging dictionary schema](https://docs.python.org/3/library/logging.config.html#logging-config-dictschema). |
| _outputDir_            | `string`  | Path to a directory, which will be used for storing outputs such as the trained model, checkpoints and evaluation results.                                                                           |
| _saveCheckpoints_      | `boolean` | Indicates whether to save training checkpoints or not.                                                                                                                                               |
| _saveModel_            | `boolean` | Indicates whether to store the trained model or not.                                                                                                                                                 |


##### Data Parameters
The `data` key contains several definitions for the hydrometeorological datasets, which should be used for model 
training, validation and testing in your experiments.

| Config Parameter       | Type     | Description                                                                                                                           |
|------------------------|----------|---------------------------------------------------------------------------------------------------------------------------------------|
| _forcings.dir_         | `string` | Path to a directory, which contains forcings datasets.                                                                                |
| _forcings.type_        | `string` | Type of the forcings datasets. Currently supported: `daymet`, `camels-us`                                                             |
| _forcings.variables_   | `array`  | List of forcing variables, which should be considered for training the model                                                          |
| _streamflow.dir_       | `string` | Path to a directory, which contains streamflow datasets.                                                                              |
| _streamflow.type_      | `string` | Type of the forcings datasets. Currently supported: `camels-us`                                                                       |
| _streamflow.variables_ | `array`  | List of streamflow variables, which should be considered for training the model. Actually, only one variable is supported, up to now. |
| _training.startDate_   | `string` | Start of the training period (ISO 8601 date string in the format yyyy-MM-dd)                                                          |
| _training.endDate_     | `string` | End of the training period (ISO 8601 date string in the format yyyy-MM-dd)                                                            |
| _validation.startDate_ | `string` | Start of the validation period (ISO 8601 date string in the format yyyy-MM-dd)                                                        |
| _validation.endDate_   | `string` | End of the validation period (ISO 8601 date string in the format yyyy-MM-dd)                                                          |
| _test.startDate_       | `string` | Start of the testing period (ISO 8601 date string in the format yyyy-MM-dd)                                                           |
| _test.endDate_         | `string` | End of the testing period (ISO 8601 date string in the format yyyy-MM-dd)                                                             |

##### Model Parameters
The `model` configuration section contains several parameters that define the model architecture and control the
training process.

| Config Parameter | Type      | Description                                                                                                                                                                                                                                                                                                                                                                                     |
|------------------|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| _type_           | `string`  | Name of the model type that will be trained. Currently supported: `lstm`, `cnn-lstm`, `multi-cnn-lstm`, `convlstm`, `conv3d`                                                                                                                                                                                                                                                                    |
| _timesteps_      | `array`   | Timesteps that will be used for creating the input (forcings) timeseries windows. E.g., if timesteps are 10, the last 10 days of forcings values will be used as inputs for model training to predict the target variable with a defined offset. If you train a model that accepts multiple inputs, such as as the `multi-cnn-lstm` model, you have to define a timesteps value for each input. |
| _offset_         | `int`     | Offset between inputs (forcings) and target (streamflow). An offset of 1 means that forcings for the last n-days will be taken as input and the streamflow for n + 1 will be taken as target.                                                                                                                                                                                                   |
| _loss_           | `array`   | List of loss functions to use for training. Name of the used loss functions must refer to [Tensorflow supported loss functions](https://www.tensorflow.org/api_docs/python/tf/keras/losses).                                                                                                                                                                                                    |
| _metrics_        | `array`   | List of metrics used for validation and evaluation. Name of the used metrics must refer to [Tensorflow supported metrics](https://www.tensorflow.org/api_docs/python/tf/keras/metrics).                                                                                                                                                                                                         |
| _optimizer_      | `string`  | Defines an optimizer that will be used for training. Must be one of [Tensorflow supported optimizers](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers).                                                                                                                                                                                                                          |
| _epochs_         | `int`     | Number of training epochs                                                                                                                                                                                                                                                                                                                                                                       |
| _batchSize_      | `int`     | Batch size to use for training                                                                                                                                                                                                                                                                                                                                                                  |
| _multiOutput_    | `boolean` | Indicates whether the model should predict multiple target variables at once or only one (currently, not supported)                                                                                                                                                                                                                                                                             |
| _params_         | `dict`    | Additional model specific configuration parameters. Which parameters can be defined depends on the `type` parameter value. Supported parameters for each model type are listed in the [models section](#models)                                                                                                                                                                                 |

## References
<a id="1">[1]</a>
Kratzert, F., Gauch, M., Nearing, G., & Klotz, D. (2022). NeuralHydrology — A Python library for Deep Learning research in
hydrology. In: _Journal of Open Source Software_, 7(71), 4050. https://doi.org/10.21105/joss.04050

<a id="2">[2]</a>
Newman, A., Sampson, K., Clark, M. P., Bock, A., Viger, R. J., Blodgett, D. (2014). _A large-sample
watershed-scale hydrometeorological dataset for the contiguous USA_. Boulder, CO: UCAR/NCAR. https://dx.doi.org/10.5065/D6MW2F4D

<a id="3">[3]</a>
Thornton, P.E., M.M. Thornton, B.W. Mayer, Y. Wei, R. Devarakonda, R.S. Vose, and R.B. Cook. 2016. _Daymet: Daily Surface
Weather Data on a 1-km Grid for North America, Version 3_. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/1328

<a id="4">[4]</a>
Thornton, M.M., R. Shrestha, Y. Wei, P.E. Thornton, S. Kao, and B.E. Wilson. 2020. _Daymet: Daily Surface Weather Data 
on a 1-km Grid for North America, Version 4_. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/1840