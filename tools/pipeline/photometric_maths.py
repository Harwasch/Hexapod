"""Display-reference parameters for the pinned PPISP camera-response model, NumPy only."""

import numpy as np
import numpy.typing as npt


def identity_crf() -> npt.NDArray[np.float64]:
    """Unit toe, shoulder and gamma with a midpoint of one half gives y=x.

    PPISP parameterizes positive values as softplus(raw)+minimum, and the midpoint
    as sigmoid(raw). Inverting those maps fixes the response independently of its
    learned initializer. No nonlinear transform is baked into composited SH colours.
    """
    return np.concatenate([np.log(np.expm1(np.array([1 - 0.3, 1 - 0.3, 1 - 0.1]))), np.zeros(1)])
