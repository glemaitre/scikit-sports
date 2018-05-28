"""Tests the module used to estimate the heart-rate."""

# Authors: Aart Goossens
#          Guillaume Lemaitre
# License: MIT

import pytest
import numpy as np
import pandas as pd

from sksports.model import exp_heart_rate_model
from sksports.model import HeartRateRegressor

POWER = np.array([100] * 10 + [240] * 100 + [160] * 100)
HEART_RATE = exp_heart_rate_model(POWER, 100, 180, 0.40, 3e-5, 20, 35)
HEART_RATE = HEART_RATE + np.random.random(HEART_RATE.size)


def test_exp_heart_rate_model():
    interval = [10, 1010, 2010]
    hr_pred = exp_heart_rate_model(POWER, 75, 200, 0.30, 3e-5, 24, 30)

    diff_hr_pred = np.diff(hr_pred)
    assert np.all(diff_hr_pred[interval[1]:interval[2]] > 0)
    assert np.all(diff_hr_pred[interval[2]:] < 0)


@pytest.mark.parametrize(
    "power, heart_rate, output_type",
    [(POWER.reshape(-1, 1), HEART_RATE, np.ndarray),
     (pd.DataFrame(
         POWER,
         index=pd.date_range('1/1/2011', periods=POWER.size, freq='s')),
      pd.Series(
          HEART_RATE,
          index=pd.date_range('1/1/2011', periods=POWER.size, freq='s')),
      pd.Series),
     (POWER.reshape(-1, 1),
      pd.Series(
          HEART_RATE,
          index=pd.date_range('1/1/2011', periods=POWER.size, freq='s')),
      np.ndarray)]
)
def test_heart_rate_regressor(power, heart_rate, output_type):
    reg = HeartRateRegressor()
    reg.fit(power, heart_rate)

    assert reg.hr_start_ == pytest.approx(100, 1.0)
    assert reg.hr_max_ == pytest.approx(180, 1.0)
    assert reg.hr_slope_ == pytest.approx(0.40, 0.01)

    hr_pred = reg.predict(power)
    assert isinstance(hr_pred, output_type)
