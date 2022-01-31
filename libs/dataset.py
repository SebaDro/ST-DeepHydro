import xarray as xr


class AbstractDataset:
    def __init__(self, timeseries: xr.Dataset):
        self.__timeseries = timeseries

    @property
    def timeseries(self):
        return self.__timeseries

    @timeseries.setter
    def timeseries(self, value):
        self.__timeseries = value

    def normalize(self, min_params: xr.Dataset, max_params: xr.Dataset):
        pass


class LumpedDataset(AbstractDataset):
    def __init__(self, timeseries: xr.Dataset, feature_variables: list, target_variables: list,
                 start_date: str = None, end_date: str = None):
        """
        Initializes a LumpedDataset instance, which wraps lumped forcings and streamflow timeseries for one or more
        basins as xarray.Dataset. Lumped means, the forcings and streamflow variables are only distributed in time but
        not in space. Forcings and streamflow values are store as variables.

        Parameters
        ----------
        timeseries: xarray.Dataset
            A xarray.Dataset that holds forcings and streamflow timeseries data. Tha variables are distribted in time
            but not in space. Dimension coordinates must be `time` for the temporal dimension and `basin` for separating
            the timeseries datasets for each basin.
        feature_variables: list
            List of names that represent the feature variables (e.g. 'temp', 'prec') within the xarray.Dataset
        target_variables
            List of names that represent the target variables (e.g. `streamflow`) within the xarray.Dataset.
        start_date:
            Start date of the timeseries
        end_date
            End date of the timeseries
        """
        self.__feature_cols = feature_variables
        self.__target_cols = target_variables
        self.__start_date = start_date
        self.__end_date = end_date
        super().__init__(timeseries)

    def normalize(self, min_params: xr.Dataset = None, max_params: xr.Dataset = None):
        """
        Performs a min/max scaling on all variables that are present within a xarray.Dataset.

        Parameters
        ----------
        min_params: xarray.Dataset
            Minimum parameters as reduced xarray.Dataset that will be used for scaling. If None, this parameter will be
            calculated from the current dataset.
        max_params: xarray.Dataset
            Maximum parameters as reduced xarray.Dataset that will be used for scaling. If None, this parameter will be
            calculated from the current dataset.
        """
        min_params = self.timeseries.min() if min_params is None else min_params
        max_params = self.timeseries.max() if max_params is None else max_params

        self.timeseries = (self.timeseries - min_params) / (max_params - min_params)
