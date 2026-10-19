"""
Reader for VIIRS I-band SDR HDF5 granules (SVI04, SVI05, GITCO), e.g. from
GINA direct broadcast.

Only the quality-flag logic is ported so far. The file reader itself will
be ported from legacy/pygaarst_hdf5.py (class VIIRSHDF5) once the GINA SDR
archive is accessible for testing.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

# QF1_VIIRSIBANDSDR value marking anomalous I4 pixels in the original analysis
QF1_I4_ANOMALOUS = 193
# I4 brightness temperature reported for saturated pixels in SDRs (K)
I4_SATURATED_BT = 367.0
# Above this I4 BT, flagged pixels are not treated as anomalous (K)
I4_ANOMALOUS_MAX_BT = 360.0


def anomaly_mask(
    i4_bt: ArrayLike, qf1_i4: ArrayLike, qf1_i5: ArrayLike
) -> NDArray[np.bool_]:
    """Return the VIFDAHL "anomalous" mask from SDR quality flags.

    Parameters
    ----------
    i4_bt : array_like
        I4 brightness temperature in K.
    qf1_i4, qf1_i5 : array_like
        ``QF1_VIIRSIBANDSDR`` values for the I4 and I5 granules.

    Returns
    -------
    numpy.ndarray of bool
        True where I4 is flagged anomalous, or saturated with good I5.
    """
    i4_bt = np.asarray(i4_bt, dtype=float)
    qf1_i4 = np.asarray(qf1_i4)
    qf1_i5 = np.asarray(qf1_i5)
    flagged = (qf1_i4 == QF1_I4_ANOMALOUS) & (i4_bt < I4_ANOMALOUS_MAX_BT)
    saturated = (qf1_i5 == 0) & (i4_bt == I4_SATURATED_BT)
    return flagged | saturated
