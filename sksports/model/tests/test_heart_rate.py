"""Tests the module used to estimate the heart-rate."""

# Authors: Aart Goossens
#          Guillaume Lemaitre
# License: MIT

import pytest
import numpy as np
import pandas as pd

from sksports.model import exp_heart_rate_model
from sksports.model import HeartRateRegressor

# POWER = np.array([100] * 10 + [240] * 100 + [160] * 100)
# HEART_RATE = exp_heart_rate_model(POWER, 100, 180, 0.40, 3e-5, 20, 35)
# HEART_RATE = HEART_RATE + np.random.random(HEART_RATE.size)


@pytest.fixture()
def power_arr():
    return np.array([100] * 10 + [240] * 100 + [160] * 100)


@pytest.fixture(params=[
    (np.asarray, None),
    (pd.DataFrame, None),
    (pd.DataFrame, pd.date_range('1/1/2011', periods=power_arr().size,
                                 freq='s')),
    (pd.DataFrame, pd.timedelta_range(start='0 day', periods=power_arr().size,
                                      freq='s'))])
def power(request, power_arr):
    power = request.param[0](power_arr.reshape(-1, 1))
    if request.param[1] is not None:
        power.index = request.param[1]
    return power


@pytest.fixture(params=[
    (np.asarray, None),
    (pd.Series, None),
    (pd.Series, pd.date_range('1/1/2011', periods=power_arr().size,
                              freq='s')),
    (pd.Series, pd.timedelta_range(start='0 day', periods=power_arr().size,
                                   freq='s'))])
def heart_rate(request, power_arr):
    heart_rate = request.param[0](
        exp_heart_rate_model(power_arr, 100, 180, 0.40, 3e-5, 20, 35) +
        np.random.random(power_arr.size))
    if request.param[1] is not None:
        heart_rate.index = request.param[1]
    return heart_rate


def test_exp_heart_rate_model(power_arr):
    interval = [10, 1010, 2010]
    hr_pred = exp_heart_rate_model(power_arr, 75, 200, 0.30, 3e-5, 24, 30)

    diff_hr_pred = np.diff(hr_pred)
    assert np.all(diff_hr_pred[interval[1]:interval[2]] > 0)
    assert np.all(diff_hr_pred[interval[2]:] < 0)


def test_heart_rate_regressor(power, heart_rate):
    reg = HeartRateRegressor()
    reg.fit(power, heart_rate)

    assert reg.hr_start_ == pytest.approx(100, 1.0)
    assert reg.hr_max_ == pytest.approx(180, 1.0)
    assert reg.hr_slope_ == pytest.approx(0.40, 0.1)

    hr_pred = reg.predict(power)
    exp_pred_type = np.ndarray if isinstance(power, np.ndarray) else pd.Series
    assert type(hr_pred) == exp_pred_type

    assert reg.score(power, heart_rate) > 0.95
