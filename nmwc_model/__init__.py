import numpy as np
from gt4py.gtscript import Field, IJ, K
import os

BACKEND = os.environ["GT4PY_BACKEND"]
REBUILD = False
DTYPE_FLOAT = np.float64
DTYPE_INT = np.int_
FIELD_FLOAT = Field[DTYPE_FLOAT]
FIELD_FLOAT_IJ = Field[IJ, DTYPE_FLOAT]
FIELD_FLOAT_K = Field[K, DTYPE_FLOAT]
FIELD_INT = Field[DTYPE_INT]
FIELD_INT_IJ = Field[IJ, DTYPE_INT]
FIELD_INT_K = Field[K, DTYPE_INT]
DEFAULT_ORIGIN = (0, 0, 0)