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
This package mainly focuses on training models for rainfall runoff predictions by using the hydro-meteorological datasets.
Therefore, a variety of datasets are suitable to be used as training data. Though, especially CAMELS-US and Daymet
datasets has been widely proven as appropriate input datasets for hydrological modelling. 

### CAMELS-US
The CAMELS-US dataset contains hydro-meteorological timeseries data for 671 basins in the continuous United States [[2]](#2).
Meteorological products contain basin aggregated daily forcings from three different data sources (Daymet, Maurer and NLDAS). 
Daily streamflow data for 671 gauges comes from the United States Geological Survey National Water Information System.
You simply can download this large-sample dataset from the [NCAR website](https://ral.ucar.edu/solutions/products/camels).

### Daymet
Daymet data contain gridded estimates of daily weather and climatology parameters at a 1 km x 1 km raster for North
America, Hawaii, and Puerto Rico [[3](#3), [4](#4)]. Daymet Version 3 and Version 4 data are provided by (ORNL DAAC)[https://daymet.ornl.gov/]
and can be via ORNL DAAC's Thematic Real-time Environmental Distributed Data Services (THREDDS). To download these
datasets for your preferred region and prepare it for model training, you might want to use the
[Daymet PyProcessing](https://github.com/SebaDro/daymet-pyprocessing) toolset.

## Models
TBD

## Training
TBD

## References
<a id="1">[1]</a>
Kratzert, F., Gauch, M., Nearing, G., & Klotz, D. (2022). NeuralHydrology — A Python library for Deep Learning research in
hydrology. In: _Journal of Open Source Software_, 7(71), 4050. https://doi.org/10.21105/joss.04050

<a id="2">[2]</a>
Newman, A., Sampson, K., Clark, M. P., Bock, A., Viger, R. J., Blodgett, D. (2014). _A large-sample
watershed-scalehydrometeorological dataset for the contiguous USA_. Boulder, CO: UCAR/NCAR. https://dx.doi.org/10.5065/D6MW2F4D

<a id="3">[3]</a>
Thornton, P.E., M.M. Thornton, B.W. Mayer, Y. Wei, R. Devarakonda, R.S. Vose, and R.B. Cook. 2016. _Daymet: Daily Surface
Weather Data on a 1-km Grid for North America, Version 3_. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/1328

<a id="4">[4]</a>
Thornton, M.M., R. Shrestha, Y. Wei, P.E. Thornton, S. Kao, and B.E. Wilson. 2020. _Daymet: Daily Surface Weather Data 
on a 1-km Grid for North America, Version 4_. ORNL DAAC, Oak Ridge, Tennessee, USA. https://doi.org/10.3334/ORNLDAAC/1840

