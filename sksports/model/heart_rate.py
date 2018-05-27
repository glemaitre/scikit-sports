"""Module to estimate heart rate from data."""

# Authors: Aart Goossens
#          Guillaume Lemaitre
# License: MIT

from __future__ import division

import numpy as np
from scipy.optimize import leastsq, minimize

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted


def _model(power, hr_rest, hr_max, hr_slope, hr_drift, rate_growth,
           rate_decay):
    """Model to predict heart-rate from power data."""
    power_corrected = power + power.cumsum() * hr_drift
    hr_lin = power_corrected * hr_slope + hr_rest
    hr_ss = np.minimum(hr_max, hr_lin)

    diff_hr = np.pad(np.diff(power_corrected), pad_width=1, mode='constant')
    rate = np.where((diff_hr <= 0), rate_decay, rate_growth)

    hr_pred = np.ones(power.shape) * hr_rest
    for curr_idx in enumerate(hr_pred.size):
        hr_prev = hr_pred[curr_idx] if curr_idx > 0 else hr_rest
        hr_pred[curr_idx] = (hr_prev +
                             (hr_ss[curr_idx] - hr_prev) / rate[curr_idx])

    return hr_pred


def _model_residual(params, heart_rate, power):
    """Residuals between prediction and ground-truth."""
    return heart_rate - _model(power, *params)


def _model_residual_least_square(params, heart_rate, power):
    """Least-square error of the residuals."""
    return np.sum(_model_residual(params, heart_rate, power) ** 2)


class HeartRateRegressor(BaseEstimator, RegressorMixin):
    """Estimate the heart-rate from power data.

    The model used is based on the formulation in [1]_.

    Parameters
    ----------
    solver : string, optional
        The solver to estimate the model parameters. Can be either 'leastsq' to
        apply Levenberg-Marquardt or one of 'method' from
        :func:`scipy.optimize.minimize`.

    hr_rest : float, default=75
        Heart-rate frequency of the cyclist at rest.

    hr_max : float, default=200
        Heart-rate frequency of the cyclist at maximum.

    hr_slope : float, default=0.30
        Slope considered if the model is linear. FIXME

    hr_drift : float, default=3e-5
        Attenuation of the heart-rate other time. FIXME

    rate_growth : float, default=24
        Growth rate of the exponential when the power increases.

    rate_decay : float, default=30
        Decay rate of the exponential when the power decreases.

    Attributes
    ----------
    hr_rest_ : float
        Fitted heart-rate frequency of the cyclist at rest.

    hr_max_ : float
        Fitted heart-rate frequency of the cyclist at maximum.

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

    def __init__(self, solver='Nelder-Mead', hr_rest=75, hr_max=200,
                 hr_slope=0.30, hr_drift=3e-5, rate_growth=24, rate_decay=30):
        self.solver = solver
        self.hr_rest = hr_rest
        self.hr_max = hr_max
        self.hr_slope = hr_slope
        self.hr_drift = hr_drift
        self.rate_growth = rate_growth
        self.rate_decay = rate_decay

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
        X, y = check_X_y(X, y, ensure_2d=False)
        params_init = [self.hr_rest, self.hr_max, self.hr_slope, self.hr_drift,
                       self.rate_growth, self.rate_decay]

        if self.solver == 'leastsq':
            res_opt, _ = leastsq(_model_residual, params_init, args=(y, X))
        else:
            res_opt = minimize(_model_residual_least_square, params_init,
                               args=(y, X), method=self.solver)['x']

        self.hr_rest_, self.hr_max_, self.hr_slope_, self. hr_drift_, \
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
        check_is_fitted(self, ['hr_rest_', 'hr_max_', 'hr_slope_', 'hr_drift_',
                               'rate_growth_', 'rate_decay_'])
        X = check_array(X, ensure_2d=False)

        return _model(X, self.hr_rest_, self.hr_max_, self.hr_slope_,
                      self. hr_drift_, self.rate_growth_, self.rate_decay_)
