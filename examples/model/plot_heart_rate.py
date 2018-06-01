"""
===========================================
Infer Heart-Rate from Power data in cyclist
===========================================

This example illustrates how :func:`sksports.model.exp_heart_rate_model` can be
used to infer heart-rate data from power. In addition, we show how to use the
:class:`sksports.model.HeartRateRegressor` to learn the parameters of such
model on some training data to be able to predict the heart-rate on the testing
data.

"""

print(__doc__)

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import matplotlib.pyplot as plt

###############################################################################
# A model to predict heart-rate from power data
###############################################################################
# We will start by simulating some power data

import numpy as np
import pandas as pd

data = pd.DataFrame(
    {'power': np.array([100] * 50 + [240] * 200 + [160] * 200)},
    index=pd.timedelta_range(start='0 day', freq='s', periods=450))

###############################################################################
# The :func:`sksports.model.exp_heart_rate_model` allows to predict the
# heart-rate associated to the power data, assuming that we know the parameters
# of the model.

from sksports.model import exp_heart_rate_model

# The function `exp_heart_rate_model` required numpy array
heart_rate = exp_heart_rate_model(data['power'].values, 75, 200, 0.4, 3e-5,
                                  15, 40)
data['heart-rate'] = heart_rate
data.plot(legend=True)
plt.title('Simulated power of a cyclist and inferred heart-rate.')

###############################################################################
# The parameters of the :func:`sksports.model.exp_heart_rate_model` are usually
# not known and can be learn from data themselves. The
# :class:`sksports.model.HeartRateRegressor` allows to learn those parameters
# from training data (power and associated heart-rate).

from sksports.model import HeartRateRegressor

# create a regressor with the default parameters
reg = HeartRateRegressor()
print('The initial parameters of the regressor are:\n {}'
      .format(reg.get_params()))

# imagine that we have some power and training data
# add some noise to look like more real
power = (data['power'] + np.random.randn(data['power'].shape[0])).to_frame()
heart_rate = data['heart-rate'] + np.random.randn(data['heart-rate'].shape[0])

# get the prediction with the default parameters
heart_rate_initial = exp_heart_rate_model(power.values.ravel(),
                                          **reg.get_params())
heart_rate_initial = pd.Series(heart_rate_initial, index=power.index)
heart_rate_initial = heart_rate_initial.rename('heart-rate initial')

reg.fit(power, heart_rate)

###############################################################################
# Once the regressor is fitted, we can actually check that the parameters have
# changed. We can also predict the heart-rate associated with the original
# power data.

heart_rate_pred = reg.predict(power)
heart_rate_pred = heart_rate_pred.rename('prediction')
print('The fitted parameters of the regressor are: \n {}'
      .format({'hr_start_': reg.hr_start_, 'hr_max_': reg.hr_max_,
               'hr_slope_': reg.hr_slope_, 'hr_drift_': reg.hr_drift_,
               'rate_decay_': reg.rate_decay_,
               'rate_growth_': reg.rate_growth_}))

power.plot(legend=True)
heart_rate.plot(legend=True)
heart_rate_initial.plot(legend=True)
heart_rate_pred.plot(legend=True)

###############################################################################
# Once the parameter fitted, the predictions are actually following the real
# data.

plt.show()
