"""Module to estimate heart rate from data."""

# Authors: Aart Goossens
#          Guillaume Lemaitre
# License: MIT

from __future__ import division

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted


def exp_heart_rate_model(power, hr_start, hr_max, hr_slope, hr_drift,
                         rate_growth, rate_decay):
    """Model heart-rate from power date based on exponential
    physiological-based model.

    The model used is based on [1]_.

    Read more in the :ref:`User Guide <heartrate_inference>`.

    Parameters
    ----------
    power : ndarray, shape (n_samples,)
        The power data in watts.

    hr_start : float
        Initial heart-rate frequency.

    hr_max : float
        Heart-rate frequency of the athlete at maximum.

    hr_slope : float
        Slope considered if the model is linear.

    hr_drift : float
        Attenuation of the heart-rate other time.

    rate_growth : float
        Growth rate of the exponential when the power increases.

    rate_decay : float
        Decay rate of the exponential when the power decreases.

    Returns
    -------
    hr_pred : ndarray, shape (n_samples,)
        Prediction of the heart-rate frequencies associated to the power data.

    References
    ----------
    .. [1] de Smet, Dimitri, et al. "Heart rate modelling as a potential
       physical fitness assessment for runners and cyclists." Workshop at ECML
       & PKDD. 2016.

    Examples
    --------
    >>> import numpy as np
    >>> power = np.array([100] * 25 + [240] * 25 + [160] * 25)
    >>> from sksports.model import exp_heart_rate_model
    >>> hr_pred = exp_heart_rate_model(power, 75, 200, 0.30, 3e-5, 24, 30)
    >>> print(hr_pred)  # doctest: +ELLIPSIS
    [...]

    """
    power = check_array(power, ensure_2d=False)
    power_corrected = power + power.cumsum() * hr_drift
    hr_lin = power_corrected * hr_slope + hr_start
    hr_ss = np.minimum(hr_max, hr_lin)

    diff_hr = np.pad(np.diff(power_corrected), pad_width=1, mode='constant')
    rate = np.where((diff_hr <= 0), rate_decay, rate_growth)

    hr_pred = np.ones(power.size) * hr_start
    for curr_idx in range(hr_pred.size):
        hr_prev = hr_pred[curr_idx - 1] if curr_idx > 0 else hr_start
        hr_pred[curr_idx] = (hr_prev +
                             (hr_ss[curr_idx] - hr_prev) / rate[curr_idx])

    return hr_pred


def _model_residual(params, heart_rate, power):
    """Residuals between prediction and ground-truth."""
    return heart_rate - exp_heart_rate_model(power, *params)


def _model_residual_least_square(params, heart_rate, power):
    """Least-square error of the residuals."""
    return np.sum(_model_residual(params, heart_rate, power) ** 2)


class HeartRateRegressor(BaseEstimator, RegressorMixin):
    """Estimate the heart-rate from power data.

    The model used is based on the formulation in [1]_.

    Read more in the :ref:`User Guide <heartrate_inference>`.

    Parameters
    ----------
    hr_start : float, default=75
        Initial heart-rate frequency.

    hr_max : float, default=200
        Heart-rate frequency of the athlete at maximum.

    hr_slope : float, default=0.30
        Slope considered if the model is linear.

    hr_drift : float, default=3e-5
        Attenuation of the heart-rate other time.

    rate_growth : float, default=24
        Growth rate of the exponential when the power increases.

    rate_decay : float, default=30
        Decay rate of the exponential when the power decreases.

    Attributes
    ----------
    hr_start_ : float
        Fitted initial heart-rate frequency.

    hr_max_ : float
        Fitted heart-rate frequency of the athlete at maximum.

    hr_slope_ : float
        Fitted slope considered if the model is linear. FIXME

    hr_drift_ : float
        Fitted attenuation of the heart-rate other time. FIXME

    rate_growth_ : float
        Fitted growth rate of the exponential when the power increases.

    rate_decay_ : float
        Fitted decay rate of the exponential when the power decreases.

    References
    ----------
    .. [1] de Smet, Dimitri, et al. "Heart rate modelling as a potential
       physical fitness assessment for runners and cyclists." Workshop at ECML
       & PKDD. 2016.

    """

    def __init__(self, hr_start=75, hr_max=200, hr_slope=0.30, hr_drift=3e-5,
                 rate_growth=24, rate_decay=30):
        self.hr_start = hr_start
        self.hr_max = hr_max
        self.hr_slope = hr_slope
        self.hr_drift = hr_drift
        self.rate_growth = rate_growth
        self.rate_decay = rate_decay

    def _check_inputs(self, X, y=None):
        """Validate the inputs.

        Check if the inputs are Pandas inputs. Resample with a sampling-rate of
        1 Hertz.

        """
        if y is None:
            is_df = True if hasattr(X, 'loc') else False
            X_ = (X.resample('1s').mean()
                  if is_df and isinstance(X.index, (pd.TimedeltaIndex,
                                                    pd.DatetimeIndex))
                  else X)

            X_ = check_array(X_)
            return X_.ravel(), None, is_df
        else:
            is_df = True if hasattr(X, 'loc') and hasattr(y, 'loc') else False
            X_ = (X.resample('1s').mean()
                  if is_df and isinstance(X.index, (pd.TimedeltaIndex,
                                                    pd.DatetimeIndex))
                  else X)
            y_ = (y.resample('1s').mean()
                  if is_df and isinstance(y.index, (pd.TimedeltaIndex,
                                                    pd.DatetimeIndex))
                  else y)
            X_, y_ = check_X_y(X_, y_)
            return X_.ravel(), y_, is_df

    def fit(self, X, y):
        """Compute the parameters model.

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            The power data.

        y : array-like, shape (n_samples,)
            The heart-rate data.

        Returns
        -------
        self

        """
        power, heart_rate, _ = self._check_inputs(X, y)
        params_init = [self.hr_start, self.hr_max, self.hr_slope,
                       self.hr_drift, self.rate_growth, self.rate_decay]

        res_opt = minimize(_model_residual_least_square, params_init,
                           args=(heart_rate, power),
                           method='Nelder-Mead')['x']

        self.hr_start_, self.hr_max_, self.hr_slope_, self. hr_drift_, \
            self.rate_growth_, self.rate_decay_ = res_opt

        return self

    def predict(self, X):
        """Predict the heart-rate frequency from the power data.

        Parameters
        ----------
        X : array-like, shape (n_samples,)
            The power data.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The heart-rate predictions.

        """
        check_is_fitted(self, ['hr_start_', 'hr_max_', 'hr_slope_',
                               'hr_drift_', 'rate_growth_', 'rate_decay_'])
        power, _, is_df = self._check_inputs(X)

        hr_pred = exp_heart_rate_model(power, self.hr_start_, self.hr_max_,
                                       self.hr_slope_, self. hr_drift_,
                                       self.rate_growth_, self.rate_decay_)

        return pd.Series(hr_pred, index=X.index) if is_df else hr_pred
