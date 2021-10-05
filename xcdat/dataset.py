"""Dataset module for functions related to an xarray.Dataset."""
from typing import Any, Dict, Hashable, List, Optional, Union

import pandas as pd
import xarray as xr

from xcdat import bounds  # noqa: F401
from xcdat.logger import setup_custom_logger

logger = setup_custom_logger(__name__)


def open_dataset(
    path: str, data_var: Optional[str] = None, **kwargs: Dict[str, Any]
) -> xr.Dataset:
    """Wrapper for ``xarray.open_dataset`` that applies common operations.

    Operations include:

    - If the dataset a time dimension, decode both CF and non-CF time units
      (months and years).
    - Generate bounds for supported coordinates if they don't exist.
    - Limit the Dataset to a single variable, since XCDAT operations are
      performed on a single variable at a time.

    Parameters
    ----------
    path : str
        Path to Dataset.
    data_var: Optional[str], optional
        The data variable to keep in the Dataset, by default None.
    kwargs : Dict[str, Any]
        Additional arguments passed on to ``xarray.open_dataset``.

        - Visit the xarray docs for accepted arguments [1]_.
        - ``decode_times`` defaults to ``False`` to allow for the manual
          decoding of non-CF time units.

    Returns
    -------
    xr.Dataset
        Dataset after applying operations.

    Notes
    -----
    ``xarray.open_dataset`` opens the file with read-only access. When you
    modify values of a Dataset, even one linked to files on disk, only the
    in-memory copy you are manipulating in xarray is modified: the original file
    on disk is never touched.

    References
    ----------

    .. [1] https://xarray.pydata.org/en/stable/generated/xarray.open_dataset.html

    Examples
    --------
    Import and call module:

    >>> from xcdat.dataset import open_dataset
    >>> ds = open_dataset("file_path")

    Keep a single variable in the Dataset:

    >>> from xcdat.dataset import open_dataset
    >>> ds = open_dataset("file_path", keep_vars="tas")

    Keep multiple variables in the Dataset:

    >>> from xcdat.dataset import open_dataset
    >>> ds = open_dataset("file_path", keep_vars=["ts", "tas"])
    """
    # NOTE: Using decode_times=False may add incorrect units for existing time
    # bounds (becomes "days since 1970-01-01  00:00:00").
    ds = xr.open_dataset(path, decode_times=False, **kwargs)
    ds = infer_or_keep_var(ds, data_var)

    if ds.cf.dims.get("T") is not None:
        ds = decode_time_units(ds)

    ds = ds.bounds.fill_missing()
    return ds


def open_mfdataset(
    paths: Union[str, List[str]],
    data_var: Optional[str] = None,
    **kwargs: Dict[str, Any],
) -> xr.Dataset:
    """Wrapper for ``xarray.open_mfdataset`` that applies common operations.

    Operations include:

    - If the dataset a time dimension, decode both CF and non-CF time units
      (months and years).
    - Generate bounds for supported coordinates if they don't exist.
    - Limit the Dataset to a single variable. XCDAT operations are performed on
      a single variable.

    Parameters
    ----------
    path : Union[str, List[str]]
        Either a string glob in the form ``"path/to/my/files/*.nc"`` or an
        explicit list of files to open. Paths can be given as strings or as
        pathlib Paths. If concatenation along more than one dimension is desired,
        then ``paths`` must be a nested list-of-lists (see ``combine_nested``
        for details). (A string glob will be expanded to a 1-dimensional list.)
    data_var: Optional[str], optional
        The data variable to keep in the Dataset, by default None.
    kwargs : Dict[str, Any]
        Additional arguments passed on to ``xarray.open_mfdataset`` and/or
        ``xarray.open_dataset``.

        - Visit the xarray docs for accepted arguments, [2]_ and [3]_.
        - ``decode_times`` defaults to ``False`` to allow for the manual
          decoding of non-CF time units.

    Returns
    -------
    xr.Dataset
        Dataset after applying operations.

    Notes
    -----
    ``xarray.open_mfdataset`` opens the file with read-only access. When you
    modify values of a Dataset, even one linked to files on disk, only the
    in-memory copy you are manipulating in xarray is modified: the original file
    on disk is never touched.

    References
    ----------

    .. [2] https://xarray.pydata.org/en/stable/generated/xarray.open_mfdataset.html
    .. [3] https://xarray.pydata.org/en/stable/generated/xarray.open_dataset.html

    Examples
    --------
    Import and call module:

    >>> from xcdat.dataset import open_mfdataset
    >>> ds = open_mfdataset(["file_path1", "file_path2"])

    Keep a single variable in the Dataset:

    >>> from xcdat.dataset import open_dataset
    >>> ds = open_mfdataset(["file_path1", "file_path2"], keep_vars="tas")

    Keep multiple variables in the Dataset:

    >>> from xcdat.dataset import open_dataset
    >>> ds = open_mfdataset(["file_path1", "file_path2"], keep_vars=["ts", "tas"])
    """
    # NOTE: Using decode_times=False may add incorrect units for existing time
    # bounds (becomes "days since 1970-01-01  00:00:00").
    ds = xr.open_mfdataset(paths, decode_times=False, **kwargs)
    ds = infer_or_keep_var(ds, data_var)

    if ds.cf.dims.get("T") is not None:
        ds = decode_time_units(ds)

    ds = ds.bounds.fill_missing()
    return ds


def infer_or_keep_var(
    dataset: xr.Dataset, data_var: Optional[str] = None
) -> xr.Dataset:
    """Infer or keep a specific data variable in the Dataset.

    If a ``data_var`` is not specified, this method checks the number of
    regular (non-bounds) data variables in the Dataset. If there a single
    regular data variable, then it will add an XCDAT inference tag to it.

    If a ``data_var`` is specified, this method checks if ``data_var`` exists
    in the Dataset and if it is a regular data var. If those checks pass,
    it will perform the appropriate Dataset subsetting.

    Parameters
    ----------
    dataset : xr.Dataset
        The Dataset.
    data_var: Optional[str], optional
        The regular data variable to keep in the Dataset, by default None.

    Returns
    -------
    xr.Dataset
        The Dataset.

    Raises
    ------
    KeyError
        If the specified data variable is not found in the Dataset.
    KeyError
        If the user specifies a bounds variable to keep (this method does that
        by default).
    """
    ds = dataset.copy()
    all_vars = ds.data_vars.keys()
    bounds_vars = ds.bounds.names
    regular_vars: List[Hashable] = list(set(all_vars) ^ set(bounds_vars))

    if len(regular_vars) == 0:
        logger.warning("This dataset only contains bounds data variables.")
    elif len(regular_vars) == 1:
        ds.attrs["xcdat_infer"] = regular_vars[0]
    elif len(regular_vars) > 1:
        logger.info(
            "This dataset contains more than one regular data variable. If "
            "desired, pass the `data_var` kwarg to reduce down to one regular data var."
        )

    if data_var is not None:
        if data_var not in all_vars:
            raise KeyError(
                f"The data variable '{data_var}' does not exist in the dataset."
            )
        if data_var in bounds_vars:
            raise KeyError("Please specify a regular (non-bounds) data variable.")

        ds = dataset[[data_var] + bounds_vars]
        ds.attrs["xcdat_infer"] = data_var

    return ds


def decode_time_units(dataset: xr.Dataset):
    """Decodes both CF and non-CF compliant time units.

    ``xarray`` uses the ``cftime`` module, which only supports CF compliant
    time units [4]_. As a result, opening datasets with non-CF compliant
    time units (months and years) will throw an error if ``decode_times=True``.

    This function works around this issue by first checking if the time units
    are CF or non-CF compliant. Datasets with CF compliant time units are passed
    to ``xarray.decode_cf``. Datasets with non-CF compliant time units are
    manually decoded by extracting the units and reference date, which are used
    to generate an array of datetime values.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset with non-decoded CF/non-CF compliant time units.

    Returns
    -------
    xr.Dataset
        Dataset with decoded time units.

    Notes
    -----
    .. [4] https://unidata.github.io/cftime/api.html#cftime.num2date

    Examples
    --------

    Decode non-CF compliant time units in a Dataset:

    >>> from xcdat.dataset import decode_time_units
    >>> ds = xr.open_dataset("file_path", decode_times=False)
    >>> ds.time
    <xarray.DataArray 'time' (time: 3)>
    array([0, 1, 2])
    Coordinates:
    * time     (time) int64 0 1 2
    Attributes:
        units:          years since 2000-01-01
        bounds:         time_bnds
        axis:           T
        long_name:      time
        standard_name:  time
    >>> ds = decode_time_units(ds)
    >>> ds.time
    <xarray.DataArray 'time' (time: 3)>
    array(['2000-01-01T00:00:00.000000000', '2001-01-01T00:00:00.000000000',
        '2002-01-01T00:00:00.000000000'], dtype='datetime64[ns]')
    Coordinates:
    * time     (time) datetime64[ns] 2000-01-01 2001-01-01 2002-01-01
    Attributes:
        units:          years since 2000-01-01
        bounds:         time_bnds
        axis:           T
        long_name:      time
        standard_name:  time

    View time coordinate encoding information:

    >>> ds.time.encoding
    {'source': None, 'dtype': dtype('int64'), 'original_shape': (3,), 'units':
    'years since 2000-01-01', 'calendar': 'proleptic_gregorian'}
    """
    time = dataset["time"]
    units_attr = time.attrs.get("units")

    if units_attr is None:
        raise KeyError(
            "No 'units' attribute found for time coordinate. Make sure to open "
            "the dataset with `decode_times=False`."
        )

    units, reference_date = units_attr.split(" since ")
    non_cf_units_to_freq = {"months": "MS", "years": "YS"}

    cf_compliant = units not in non_cf_units_to_freq.keys()
    if cf_compliant:
        dataset = xr.decode_cf(dataset, decode_times=True)
    else:
        # NOTE: The "calendar" attribute for units consisting of "months" or
        # "years" is not factored when generating date ranges. The number of
        # days in a month is not factored.
        decoded_time = xr.DataArray(
            data=pd.date_range(
                start=reference_date,
                periods=time.size,
                freq=non_cf_units_to_freq[units],
            ),
            dims=["time"],
            attrs=dataset["time"].attrs,
        )
        decoded_time.encoding = {
            "source": dataset.encoding.get("source"),
            "dtype": time.dtype,
            "original_shape": decoded_time.shape,
            "units": units_attr,
            # pandas.date_range() returns "proleptic_gregorian" by default
            "calendar": "proleptic_gregorian",
        }

        dataset = dataset.assign_coords({"time": decoded_time})
    return dataset
