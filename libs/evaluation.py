from typing import Union
import logging
import xarray as xr
from libs import processing

logger = logging.getLogger(__name__)


def nse_metric(observations: xr.DataArray, predictions: xr.DataArray, as_dataset: bool = False, basin: str = None)\
        -> Union[float, xr.Dataset]:
    """
    Calculates the Nash-Sutcliffe Efficiency (NSE) metric from streamflow observations and predictions.

    Parameters
    ----------
    observations: xarray:DataArray
        DataArray containing the streamflow observations (true values)
    predictions: xarray:DataArray
        DataArray containing the streamflow model predictions
    as_dataset: bool
        Indicates whether the calculated NSE metric should be returned as raw value or as xarray.Dataset indexed by the
        basin ID.
    basin: str
        ID of the basin to calculate the NSE metric for
    Returns
    -------
    Union[float, xr.Dataset]
        Nash-Sutcliffe Efficiency (NSE) metric either as raw float value or basin indexed xarray.Dataset

    """
    nse = 1 - xr.DataArray.sum((predictions - observations) ** 2) / xr.DataArray.sum(
        (observations - observations.mean()) ** 2)
    if as_dataset:
        return xr.Dataset(dict(nse=(["basin"], [nse])), coords=dict(basin=[basin]))
    else:
        return nse


class Evaluation:

    def __init__(self):
        self.__ds_results = xr.Dataset()

    @property
    def ds_results(self):
        return self.__ds_results

    def append_evaluation_results(self, ds: xr.Dataset):
        """
        Appends new evaluation results to the existing ones

        Parameters
        ----------
        ds: xarray.Dataset
            Dataset containing aligned observation and prediction timeseries data as well as basin indexed NSE metrics

        """
        self.__ds_results = xr.merge([self.__ds_results, ds])

    def calc_evaluation(self, ds_obs: xr.Dataset, ds_pred: xr.Dataset, target_var: str, basin: str,
                        append: bool = True) -> Union[None, xr.Dataset]:
        """
        Prepares observations and predictions and calculates evaluation metrics for a certain basin and target variable.
        Therefore, xarray.Datasets containing observations and predictions are merged to cover the same timespan in a
        first step. Following, the NSE metric is calculated and the result appended to already existing results

        Parameters
        ----------
        ds_obs: xarray.Dataset
            Dataset containing obsrvations
        ds_pred: xarray.Dataset
            Dataset containing obsrvations
        target_var: str
            Name of the target variable, which will be used to create target_obs and target_pred variables in the
            common dataset.
        basin: str
            ID of the basin
        append: bool
            Indicates whether the evaluation result should be appended to the existing ones of this instance or returned
            without appending.

        Returns
        -------
        Union[None, xr.Dataset]
            None or a dataset containing the evaluation results in accordance to the 'append' parameter

        """
        ds_res = processing.merge_observations_and_predictions(ds_obs, ds_pred, True)
        observations = ds_res[target_var + "_obs"]
        predictions = ds_res[target_var + "_pred"]

        ds_nse = nse_metric(observations, predictions, as_dataset=True, basin=basin)
        ds_res = xr.merge([ds_res, ds_nse])
        if append:
            self.append_evaluation_results(ds_res)
        else:
            return ds_res
