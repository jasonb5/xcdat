"""Functions related to calculating climatology cycles and departures."""

from typing import Dict, Optional, Union, get_args

import numpy as np
import xarray as xr
from typing_extensions import Literal

from xcdat.logger import setup_custom_logger

logger = setup_custom_logger("root")

# PERIODS
# =======
# Type alias representing climatology periods for the ``frequency`` param.
Period = Literal["month", "season", "year"]
# Tuple for available period groups.
PERIODS = get_args(Period)

# MONTHS
# ======
# Type alias representing months for the ``frequency`` param.
Month = Literal[
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEPT", "OCT", "NOV", "DEC"
]
# Tuple for available months.
MONTHS = get_args(Month)
# Maps str representation of months to integer for xarray operations.
MONTHS_TO_INT = dict(zip(MONTHS, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)))

# SEASONS
# =======
# Type alias representing seasons for the ``frequency`` param.
Season = Literal["DJF", "MAM", "JJA", "SON"]
# Tuple for available seasons.
SEASONS = get_args(Season)

# ALL FREQUENCIES
# ===============
# Type alias representing available ``frequency`` param options.
Frequency = Union[Period, Month, Season]
#: Tuple of available frequencies for the ``frequency`` param.
FREQUENCIES = PERIODS + MONTHS + SEASONS

# DATETIME COMPONENTS
# ===================
# Type alias representing xarray DateTime components.
DateTimeComponent = Literal["time.month", "time.season", "time.year"]
# Maps available frequencies to xarray DateTime components for xarray operations.
FREQUENCIES_TO_DATETIME: Dict[str, DateTimeComponent] = {
    **{period: f"time.{period}" for period in PERIODS},  # type: ignore
    **{month: "time.month" for month in MONTHS},
    **{season: "time.season" for season in SEASONS},
}

# DJF CLIMATOLOGY SPECIFIC
# ========================
# Type alias for DJF as seasonally continuous ("scd") or discontinuous ("sdd")
DJFType = Literal["scd", "sdd"]
# Tuple of "scd" and "sdd" for `djf_type` param.
DJF_TYPES = get_args(DJFType)


def climatology(
    ds: xr.Dataset,
    frequency: Frequency,
    is_weighted: bool = True,
    djf_type: DJFType = "scd",
) -> xr.Dataset:
    """Calculates a Dataset's climatology cycle for all data variables.

    The "time" dimension and any existing bounds variables are preserved in the
    dataset.

    # TODO: Daily climatology

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to calculate climatology cycle.
    frequency : Frequency
        The frequency of time to group by. Available aliases:

        - ``"month"`` for monthly climatologies.
        - ``"year"`` for annual climatologies.
        - ``"season"` for seasonal climatologies.
        - ``"JAN", "FEB", ..., or "DEC"`` for specific month climatology.
          - Averages the month across all seasons.
        - ``"DJF", "MAM", "JJA", or "SON"`` for specific season climatology.
          - Average the season across all years.

        Refer to ``FREQUENCIES`` for a complete list of available options.
    is_weighted : bool, optional
        Perform grouping using weighted averages, by default True.
        Time bounds, leap years, and month lengths are considered.
    djf_type : DJFType, optional
        Whether DJF climatology contains a seasonally continuous or
        discontinuous December, by default ``"scd"``.

        - ``"scd"`` for seasonally continuous December.
        - ``"sdd"`` or ``None`` for seasonally discontinuous December.

        Seasonally continuous December (``"scd"``) refers to continuity between
        December and January. DJF starts on the first year Dec and second year
        Jan/Feb, and ending on the second to last year Dec and last year Jan +
        Feb). Incomplete seasons are dropped (first year Jan/ Feb and last year
        Dec).

        - Example Date Range: Jan/2015 - Dec/2017
        - Start -> Dec/2015, Jan/2016, Feb/2016
        - End -> Dec/2016, Jan/2017, Feb/2017
        - Dropped incomplete seasons -> Jan/2015, Feb/2015, and Dec/2017

        Seasonally discontinuous December (``"sdd"``) refers to discontinuity
        between Feb and Dec. DJF starts on the first year Jan/Feb/Dec, and
        ending on the last year Jan/Feb/Dec. This is the default xarray behavior
        when grouping by season.

        - Example Date Range: Jan/2015 - Dec/2017
        - Start -> Jan/2015, Feb/2015, Dec/2015
        - End -> Jan/2017, Feb/2017, Dec/2017

    Returns
    -------
    xr.Dataset
        Climatology cycle for all data variables for a frequency of time.

    Raises
    ------
    ValueError
        If incorrect ``frequency`` argument is passed.
    KeyError
        If the dataset does not have "time" coordinates.

    Examples
    --------
    Import:

    >>> import xarray as xr
    >>> from xcdat.climatology import climatology, departure
    >>> ds = xr.open_dataset("file_path")

    Get monthly, seasonal, or annual weighted climatology:

    >>> ds_climo_monthly = climatology(ds, "month")
    >>> ds_climo_seasonal = climatology(ds, "season")
    >>> ds_climo_annual = climatology(ds, "year")

    Get monthly, seasonal, or annual unweighted climatology:

    >>> ds_climo_monthly = climatology(ds, "month", is_weighted=False)
    >>> ds_climo_seasonal = climatology(ds, "season", is_weighted=False)
    >>> ds_climo_annual = climatology(ds, "year", is_weighted=False)

    Access attribute for info on climatology operation:

    >>> ds_climo_monthly.calculation_info
    {'type': 'climatology', 'frequency': 'month', 'is_weighted': True}
    >>> ds_climo_monthly.attrs["calculation_info"]
    {'type': 'climatology', 'frequency': 'month', 'is_weighted': True}
    """
    if ds.get("time") is None:
        raise KeyError(
            "This dataset does not have 'time' coordinates. Cannot calculate climatology."
        )

    if frequency not in FREQUENCIES:
        raise ValueError(
            f"Incorrect `frequency` argument. Supported frequencies include: {', '.join(FREQUENCIES)}."
        )

    if djf_type not in DJF_TYPES:
        raise ValueError(
            f"Incorrect `djf_type` argument. Supported DJF types include: {', '.join(DJF_TYPES)}"
        )

    ds_copy = ds.copy(deep=True)
    ds_climatology = _group_data(
        ds_copy, "climatology", frequency, is_weighted, djf_type
    )
    return ds_climatology


def departure(ds_base: xr.Dataset, ds_climatology: xr.Dataset) -> xr.Dataset:
    """Calculates departures for a given climatology.

    First, the base dataset is grouped using the same frequency and weights (if
    weighted) as the climatology dataset. After grouping, it iterates over the
    dataset to get the difference between non-bounds variables in the base
    dataset and the climatology dataset. Bounds variables are preserved.

    Parameters
    ----------
    ds_base : xr.Dataset
        The base dataset.
    ds_climatology : xr.Dataset
        A climatology dataset.

    Returns
    -------
    xr.Dataset
        The climatology departure between the base and climatology datasets.

    Examples
    --------
    Import:

    >>> import xarray as xr
    >>> from xcdat.climatology import climatology, departure

    Get departure for any time frequency:

    >>> ds = xr.open_dataset("file_path")
    >>> ds_climo_monthly = climatology(ds, "month")
    >>> ds_departure = departure(ds, ds_climo_monthly)

    Access attribute for info on departure operation:

    >>> ds_climo_monthly.calculation_info
    {'type': 'departure', 'frequency': 'month', 'is_weighted': True}
    >>> ds_climo_monthly.attrs["calculation_info"]
    {'type': 'departure', 'frequency': 'month', 'is_weighted': True}
    """
    frequency = ds_climatology.attrs["calculation_info"]["frequency"]
    is_weighted = ds_climatology.attrs["calculation_info"]["is_weighted"]
    djf_type = ds_climatology.attrs["calculation_info"].get("djf_type")

    ds_departure = _group_data(
        ds_base.copy(deep=True), "departure", frequency, is_weighted, djf_type
    )

    for key in ds_departure.data_vars.keys():
        if "_bnds" not in str(key):
            ds_departure[key] = ds_departure[key] - ds_climatology[key]

    return ds_departure


def _group_data(
    ds: xr.Dataset,
    calculation_type: Literal["climatology", "departure"],
    frequency: Frequency,
    is_weighted: bool,
    djf_type: Optional[DJFType] = None,
) -> xr.Dataset:
    """Groups data variables by a frequency to get their averages.

    It iterates over each non-bounds variable and groups them. After grouping,
    attributes are added to the dataset to describe the operation performed.
    This distinguishes datasets that have been manipulated from their original source.

    This "time" dimension and any existing bounds variables are preserved in the
    dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to perform group operation on.
    calculation_type : Literal["climatology", "departure"]
        The calculation type.
    frequency : Frequency
        The frequency of time to group on.
    is_weighted : bool
        Perform grouping using weighted averages.
    djf_type : Optional[DJFType], optional
        Whether DJF climatology contains a seasonally continuous or
        discontinuous December.

    Returns
    -------
    xr.Dataset
        The dataset with grouped data variables.
    """
    if frequency in MONTHS + SEASONS:
        ds = _subset_dataset(ds, frequency)

    weights = calculate_weights(ds, frequency) if is_weighted else None
    datetime_component: DateTimeComponent = FREQUENCIES_TO_DATETIME[frequency]
    for key in ds.data_vars.keys():
        if "_bnds" not in str(key):
            data_var = ds[key]

            if is_weighted:
                data_var *= weights

            # For DJF, scd uses a rolling window grouping operation to start on
            # the first year Dec, while DJF sdd uses the default groupby operation
            # TODO: Make sure seasonal and specific seasons can support daily, sub-monthly, etc.
            if frequency == "DJF" and djf_type == "scd":
                ds[key] = data_var.rolling(min_periods=3, center=True, time=3).sum(
                    dim="time"
                )
                ds[key] = ds[key].groupby("time.year").sum(dim="time")
            else:
                ds[key] = data_var.groupby(datetime_component).sum(dim="time")

    ds = _add_attributes(ds, calculation_type, frequency, is_weighted, djf_type)
    return ds


def _subset_dataset(ds: xr.Dataset, frequency: Frequency) -> xr.Dataset:
    """Subsets data by a time frequency.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to subset.
    frequency : Frequency
        The frequency to subset with.

    Returns
    -------
    xr.Dataset
        The subsetted dataset.
    """
    datetime_component: DateTimeComponent = FREQUENCIES_TO_DATETIME[frequency]

    if frequency in MONTHS:
        month_int = MONTHS_TO_INT[frequency]
        ds = ds.where(ds[datetime_component] == month_int, drop=True)
    elif frequency in SEASONS:
        ds = ds.where(ds[datetime_component] == frequency, drop=True)

    return ds


def calculate_weights(ds: xr.Dataset, frequency: Frequency) -> xr.DataArray:
    """Calculates weights for a Dataset based on a frequency of time.

    Time bounds, leap years and number of days for each month are considered
    during grouping.

    If the sum of the weights does not equal 1.0, an error will be thrown.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to calculate weights for.
    frequency : Frequency
        The frequency of time to group on.
        Refer to ``FREQUENCIES`` for a complete list of available options.

    Returns
    -------
    xr.DataArray
        The weights based on a frequency of time.
    """
    months_lengths = _get_months_lengths(ds)
    datetime_component: DateTimeComponent = FREQUENCIES_TO_DATETIME[frequency]

    weights: xr.DataArray = (
        months_lengths.groupby(datetime_component)
        / months_lengths.groupby(datetime_component).sum()
    )
    _validate_weights(ds, weights, datetime_component)

    return weights


def _get_months_lengths(ds: xr.Dataset) -> xr.DataArray:
    """Get the months' lengths based on the time coordinates of a dataset.

    If time bounds exist, it will be used to generate the months' lengths. This
    allows for a robust calculation of weights because different datasets could
    record their time differently (e.g., at beginning/end/middle of each time
    interval).

    If time bounds do not exist, use the time variable (which may be less
    accurate based on the previously described time recording differences).
    # TODO: Generate time bounds if they don't exist?

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to get months' lengths from.

    Returns
    -------
    xr.DataArray
        The months' lengths for the dataset.
    """
    time_bounds = ds.get("time_bnds")

    if time_bounds is not None:
        logger.info("Using existing time bounds to calculate weights.")
        months_lengths = (time_bounds[:, 1] - time_bounds[:, 0]).dt.days
    else:
        logger.info("No time bounds found, using time to calculate weights.")
        months_lengths = ds.time.dt.days_in_month

    return months_lengths


def _validate_weights(
    ds: xr.Dataset, weights: xr.DataArray, datetime_component: DateTimeComponent
):
    """Validate that the sum of the weights for a dataset equals 1.0.

    It generates the number of frequency groups after grouping by a frequency.
    For example, if generating weights on a monthly basis, there are 12 groups
    for the 12 months in year.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to validate weights for.
    weights : xr.DataArray
        The weights based on a frequency of time.
    datetime_component : DateTimeComponent
        The frequency of time to group by in xarray datetime component notation.
    """
    frequency_groups = len(ds.time.groupby(datetime_component).count())

    expected_sum = np.ones(frequency_groups)
    actual_sum = weights.groupby(datetime_component).sum().values

    np.testing.assert_allclose(actual_sum, expected_sum)


def _add_attributes(
    ds: xr.Dataset,
    calculation_type: Literal["climatology", "departure"],
    frequency: Frequency,
    is_weighted: bool,
    djf_type: Optional[DJFType],
) -> xr.Dataset:
    """Adds calculation information attributes to a dataset for reference.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset.
    calculation_type : Literal["climatology", "departure"],
        The calculation type.
    frequency : Frequency
        The frequency of time.
    is_weighted : bool
        Whether to calculation was weighted or not.
    djf_type : Optional[DJFType]
        If frequency is "DJF", whether seasonally continuous or discontinuous
        December.

    Returns
    -------
    xr.Dataset
        The dataset with new calculation_info dict attribute.
    """
    ds.attrs.update(
        {
            "calculation_info": {
                "type": calculation_type,
                "frequency": frequency,
                "is_weighted": is_weighted,
            },
        }
    )

    if frequency == "DJF":
        ds.attrs.update({"djf_type": djf_type})

    return ds
