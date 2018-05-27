"""
The :mod:`sksports.model` module includes algorithms to model cycling data.
"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Cedric Lemaitre
# License: MIT

from .heart_rate import HeartRateRegressor

from .power import strava_power_model

__all__ = ['HeartRateRegressor',
           'strava_power_model']
