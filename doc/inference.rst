.. _inference:

.. currentmodule:: sksports

=========
Inference
=========

.. _heartrate_inference:

Heart-rate inference
--------------------

Smet et al. [S2015]_ proposes a model to infer heart-rate from power data in
both cyclists and runners. :func:`exp_heart_rate_model` allows to predict the
heart-rate based on this model::

  >>> import numpy as np
  >>> power = np.array([100] * 25 + [240] * 25 + [160] * 25)
  >>> from sksports.model import exp_heart_rate_model
  >>> hr_pred = exp_heart_rate_model(power, 100, 180, 0.40, 3e-5, 20, 40)
  >>> print(hr_pred)  # doctest: +ELLIPSIS
  [...]

However, this model required parameters which need to be known. The
:class:`sksports.model.HeartRateRegressor` class provides a functionality to
learn the parameters on some training data and later on predict the heart-rate
on some new data::

  >>> from sksports.model import HeartRateRegressor
  >>> heart_rate = hr_pred  # some training heart-rate associated to the power
  >>> reg = HeartRateRegressor()
  >>> reg.get_params()  # doctest: +NORMALIZE_WHITESPACE
  {'hr_drift': 3e-05, 'hr_max': 200, 'hr_slope': 0.3, 'hr_start': 75,
   'rate_decay': 30, 'rate_growth': 24}
  >>> reg.fit(power.reshape(-1, 1), heart_rate)
  HeartRateRegressor(hr_drift=3e-05, hr_max=200, hr_slope=0.3, hr_start=75,
            rate_decay=30, rate_growth=24)
  >>> hr_pred = reg.predict(power.reshape(-1, 1))
  >>> print(hr_pred)  # doctest: +ELLIPSIS
  [...]

.. topic:: Mathematical formulation:

   In [S2015]_, the heart-rate frequency at time :math:`t + 1` is formulated
   as:

   .. math::

      HR(t+1) = \left\{
                \begin{array}{ll}
                  HR(t) + \frac{1}{\tau_r} \left(HR_{ss}(PO(t)) - HR(t) \right), \text{if } HR_{ss}(PO(t)) \geq HR(t)\\
                  HR(t) + \frac{1}{\tau_f} \left(HR_{ss}(PO(t)) - HR(t) \right), \text{if } HR_{ss}(PO(t)) < HR(t)
                \end{array}
              \right.

   where :math:`HR(\cdot)` is the heart-rate, :math:`HR_{ss}(\cdot)` is the
   steady state heart-rate, :math:`PO(\cdot)` is the power output, and
   :math:`\tau_r` and :math:`\tau_f` are the growth and decay rate,
   respectively.

   The steady-state heart-rate :math:`HR_ss(\cdot)` is modeled as a linear
   function bounded by the maximum heart-rate of cyclist :math:`HR_{max}`. It
   is defined as:

   .. math::

      HR_{ss}(PO) = \left\{
                \begin{array}{ll}
                  HR_{start} + slope \times PO, \text{if } HR_{start} + slope \times PO < HR_{max}\\
                  HR_{max}, \text{otherwise}
                \end{array}
              \right.

   where :math:`HR_{start}` is the initial heart-rate and :math:`slope` is the
   slope of the linear function.
  
.. topic:: Examples:

    * :ref:`sphx_glr_auto_examples_model_plot_heart_rate.py`


.. topic:: References

   .. [S2015] de Smet, Dimitri, et al. "Heart rate modelling as a potential
      physical fitness assessment for runners and cyclists." Workshop
      at ECML & PKDD. 2016.

Power inference
---------------

Power meters are expensive tools and it is possible to get an estimate using
simple data which are acquired through phone or GPS-based cycling computer. We
are presenting several models which allow to estimate the power from those
data.

.. _strava:

Physical model using all forces applied to a cyclist
....................................................

This is possible to compute the power by adding all forces applied to cyclist
in motion. The mathematical formulation of such model is:

.. math::
   P_{meca} = \left( \frac{1}{2} \rho \cdot SC_x \cdot V_a^2 + C_r \cdot mg \cdot \cos \alpha + mg \cdot \sin \alpha + m \cdot a \right) \cdot V_d

where :math:`\rho` is the air density in :math:`kg.m^{-3}`, :math:`S` is
frontal surface of the cyclist in :math:`m^2`, :math:`C_x` is the drag
coefficient, :math:`V_a` is the air speed in :math:`m.s^{-1}`, :math:`C_r` is
the rolling coefficient, :math:`m` is the mass of the rider and bicycle in
:math:`kg`, :math:`g` in the gravitational constant which is equal to 9.81
:math:`m.s^{-2}`, :math:`\alpha` is the slope in radian, :math:`a` is the
acceleration in :math:`m.s^{-1}`, and :math:`V_d` is the rider speed in
:math:`m.s^{-1}`.

The function :func:`model.strava_power_model` allows to estimate the power
using this model. Note that we are using the default argument but the you can
set more precisely the argument to fit your condition. To estimate, we need
to::

  >>> from sksports.datasets import load_fit
  >>> from sksports.io import bikeread
  >>> from sksports.model import strava_power_model
  >>> ride = bikeread(load_fit()[0])
  >>> power = strava_power_model(ride, cyclist_weight=72)
  >>> print(power['2014-05-07 12:26:28':
  ...             '2014-05-07 12:26:38'])  # Show 10 sec of estimated power
  2014-05-07 12:26:28    196.567898
  2014-05-07 12:26:29    198.638094
  2014-05-07 12:26:30    191.444894
  2014-05-07 12:26:31     26.365864
  2014-05-07 12:26:32     89.826104
  2014-05-07 12:26:33    150.842325
  2014-05-07 12:26:34    210.083958
  2014-05-07 12:26:35    331.573965
  2014-05-07 12:26:36    425.013711
  2014-05-07 12:26:37    428.806914
  2014-05-07 12:26:38    425.410451
  Freq: S, dtype: float64

By default the term :math:`g \cdot a \cdot V_d` is not computed. Using this
term, the results can be unstable when the change of power is non smooth. To
enable it, turn ``use_acceleration=True``

.. topic:: Examples:

    * :ref:`sphx_glr_auto_examples_model_plot_physic_model.py`


.. _machine_learning:

Machine learning model
......................

This part is in progress. Find more at
`this link <https://github.com/scikit-sports/research/tree/master/power_regression>`_.
