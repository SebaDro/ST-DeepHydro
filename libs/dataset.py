import xarray as xr


class AbstractDataset:
    def __init__(self, timeseries: xr.Dataset, feature_variables: list, target_variables: list,
                 start_date: str = None, end_date: str = None):
        self.__timeseries = timeseries
        self.__feature_cols = feature_variables
        self.__target_cols = target_variables
        self.__start_date = start_date
        self.__end_date = end_date

    @property
    def timeseries(self):
        return self.__timeseries

    @timeseries.setter
    def timeseries(self, value):
        self.__timeseries = value

    @property
    def feature_cols(self):
        return self.__feature_cols

    @property
    def target_cols(self):
        return self.__target_cols

    @property
    def start_date(self):
        return self.__start_date

    @property
    def end_date(self):
        return self.__end_date

    def get_input_shape(self):
        """
        Determines the input shape considering all feature variables as well as additional dimensions. Only basind and
        time dimension will be ignored which are both not relevant for the input shape.

        Returns
        -------
        tuple:
            Input shape

        """
        dim_indices = [dim for dim in self.timeseries[self.feature_cols].to_array().dims if
                       dim not in ["variable", "basin", "time"]]
        dim_size = tuple(self.timeseries[dim].size for dim in dim_indices)
        return dim_size + (len(self.feature_cols),)


class LumpedDataset(AbstractDataset):
    def __init__(self, timeseries: xr.Dataset, feature_variables: list, target_variables: list,
                 start_date: str = None, end_date: str = None):
        """
        Initializes a LumpedDataset instance, which wraps lumped forcings and streamflow timeseries for one or more
        basins as xarray.Dataset. Lumped means, the forcings and streamflow variables are only distributed in time but
        not in space. Forcings and streamflow values are stored as variables.

        Parameters
        ----------
        timeseries: xarray.Dataset
            A xarray.Dataset that holds forcings and streamflow timeseries data. The variables are distributed in time
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
        super().__init__(timeseries, feature_variables, target_variables, start_date, end_date)


class DistributedDataset(AbstractDataset):
    def __init__(self, timeseries: xr.Dataset, feature_variables: list, target_variables: list,
                 start_date: str = None, end_date: str = None):
        """
        Initializes a DistributedDataset instance, which wraps forcings and streamflow timeseries data for one basin as
        xarray.Dataset. Forcings and streamflow values are stored as variables. Forcing variables are distrbuted in time
        and space which means they have an x, y and time dimension. The streamflow variables are only distributed in
        time but not in space. Hence, they only have a time dimension.

        Parameters
        ----------
        timeseries: xarray.Dataset
            A xarray.Dataset that holds forcings and streamflow timeseries data. Dimension coordinates for forcings must
            be 'x' and 'y' for the spatial dimension and 'time' for the temporal dimension. Streamflow only has a `time`
             dimension. For convenience, the dataset is also indexed by a 'basin' dimension.
        feature_variables: list
            List of names that represent the feature variables (e.g. 'temp', 'prec') within the xarray.Dataset
        target_variables
            List of names that represent the target variables (e.g. `streamflow`) within the xarray.Dataset.
        start_date:
            Start date of the timeseries
        end_date
            End date of the timeseries
        """
        super().__init__(timeseries, feature_variables, target_variables, start_date, end_date)
