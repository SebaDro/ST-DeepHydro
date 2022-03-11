import copy
import logging
import xarray as xr

from libs import dataset


logger = logging.getLogger(__name__)


def merge_observations_and_predictions(ds_observation: xr.Dataset, ds_prediction: xr.Dataset,
                                       use_pred_timeframe: bool = True) -> xr.Dataset:
    """
    Merges two xarray.Datasets which contain observation and prediction timeseries data. Only variables that are
    present within the prediction dataset will be considered for subsetting the observation dataset. For each
    variable the resulting xarray.Dataset contains two variables, each one named by the original variable but with
    a prefix indicating observations or predictions. E.g. if the prediction dataset contains a variable named
    "streamflow", the resulting dataset contains two variables ""streamflow_obs" and "streamflow_pred".
    By default the predictions timeframe will be used for merging both datasets.

    Parameters
    ----------
    ds_observation: xarray.Dataset
        Dataset that contains observations
    ds_prediction
        Dataset that contains predictions
    use_pred_timeframe: bool
        If True the prediction dataset timeframe should be used for merging both datasets. Otherwise the observation
        dataset timeframe will be preserved.

    Returns
    -------
    xarray.Dataset
        Dataset containing merged observation and prediction timeseries

    """
    variables = list(ds_prediction.keys())

    if use_pred_timeframe:
        start_date = ds_prediction.time[0]
        end_date = ds_prediction.time[-1]
    else:
        start_date = ds_observation.time[0]
        end_date = ds_observation.time[-1]

    ds_obs = ds_observation[variables].sel(time=slice(start_date, end_date))
    ds_obs = ds_obs.rename(dict((param, param + "_obs") for param in variables))
    ds_prediction = ds_prediction.rename(dict((param, param + "_pred") for param in variables))

    return xr.merge([ds_prediction, ds_obs], join="left") \
        if use_pred_timeframe \
        else xr.merge([ds_prediction, ds_obs], join="right")


class AbstractProcessor:
    def __init__(self, scaling_params: tuple = None):
        self.__scaling_params = scaling_params

    @property
    def scaling_params(self):
        return self.__scaling_params

    @scaling_params.setter
    def scaling_params(self, value):
        self.__scaling_params = value

    def fit(self, ds: dataset.HydroDataset):
        pass

    def process_and_fit(self, ds: dataset.HydroDataset) -> dataset.HydroDataset:
        pass

    def process(self, ds: dataset.HydroDataset) -> dataset.HydroDataset:
        pass

    def scale(self, ds: xr.Dataset):
        """
        Performs a min/max scaling on all variables of a xarray.Dataset. If the current processor instance has been
        fit to a dataset, scaling will be done by using the minimum and maximum parameters from the fitting dataset.
        Else, minimum and maximum parameters will be calculated from the given xarray.Dataset.

        Parameters
        ----------
        ds: xarray.Dataset
            Timeseries data which will be scaled.
        """
        if self.scaling_params is None:
            min_params = ds.min()
            max_params = ds.max()
        else:
            min_params, max_params = self.scaling_params
        return (ds - min_params) / (max_params - min_params)

    def rescale(self, ds: xr.Dataset):
        min_params, max_params = self.scaling_params
        return ds * (max_params - min_params) + min_params


class DefaultDatasetProcessor(AbstractProcessor):
    def __init__(self, scaling_params: tuple = None):
        """
        Initializes a DefaultDatasetProcessor instance that peforms several default processing steps on timeseries data
        wrapped by a dataset.AbstractDataset instance.

        Parameters
        ----------
        scaling_params: tuple
            Parameters that should be used for performing min-max-sacling on the timeseries data.
        """
        super().__init__(scaling_params)

    def fit(self, ds: dataset.HydroDataset):
        """
        Fits the processor to a dataset which usually should be the training dataset. Fitting means, the processor will
        derive various parameters from the specified dataset which will be used for several subsequent processing steps.
        Usually, you will fit the processor on the training data to use the derived parameters for processing the
        validation and test datasets.

        Up to now, this method will derive the following parameters:
        - Minimum and maximum values for each variable, which will be used for performing a min-max-scalin.

        Parameters
        ----------
        ds: dataset.HydroDataset
            Dataset that holds timeseries data as xarray.Dataset

        """
        self.__fit_scaling_params(ds)

    def process(self, ds: dataset.HydroDataset):
        """
        Performs several processing steps on a dataset.LumpedDataset.

        Note, that it will use parameters that have been
        derived while fitting the processor to a dataset using the fit function. If this function have not been called
        before, it will automatically derive the same parameters form the specified dataset. This will lead to
        misleading results if you aim to process validation and test datsets by using processing parameters derived from
        a training dataset. Hence, it is strongly recommended to first call fit() on a dedicated dataset-

        Parameters
        ----------
        ds: dataset.HydroDataset
            Dataset that will be processed

        Returns
        -------
            The resulting dataset.BasinDataset after performing various processing steps on it

        """
        if self.scaling_params is None:
            logging.warning("Processor has not been fit to a dataset before. Thus, it will be fitted to the provided "
                            "dataset.")
            self.__fit_scaling_params(ds)
        ds = copy.copy(ds)
        ds.timeseries = self.scale(ds.timeseries)
        return ds

    def __fit_scaling_params(self, ds: dataset.HydroDataset):
        self.scaling_params = (ds.timeseries.min(), ds.timeseries.max())
