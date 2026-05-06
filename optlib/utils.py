import numpy as np

# =====================================================================
# EXCEPTIONS
# =====================================================================
class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class InputError(Error):
    """
    Exception raised for errors in the input parameters.
    """
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

# =====================================================================
# MATH UTILITIES
# =====================================================================
def linspace(start, stop, step=1.0):
    """
    Creates an array with a specific step size that is guaranteed 
    to exactly hit the 'stop' boundary.
    """
    num = int((stop - start) / step + 1)
    return np.linspace(start, stop, num)

def conv(x1, x2, de, mode='right'):
    """Performs a 1D convolution, scaling by the energy step (de)."""
    n = x1.size
    a = np.convolve(x1, x2)
    if mode == 'right':
        return a[0:n] * de
    elif mode == 'left':
        return a[a.size - n:a.size] * de
    else:
        return a * de

def gauss(x, a1, b1, c1):
    """Evaluates a 1D Gaussian function."""
    return a1 * np.exp(-((x - b1) / c1)**2)

# =====================================================================
# VALIDATION UTILITIES
# =====================================================================
def check_list_type(lst, type_to_check):
    if lst and isinstance(lst, list):
        return all(isinstance(elem, type_to_check) for elem in lst)
    elif lst:
        return isinstance(lst, type_to_check)
    else:
        return True

def is_list_of_int_float(lst):
    if lst and isinstance(lst, list):
        return all(isinstance(elem, (int, float)) for elem in lst)
    elif lst:
        return isinstance(lst, (int, float))
    else:
        return True