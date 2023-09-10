import numpy as np
import AbstractValue


class UnshapedArray(AbstractValue):
    __slots__ = ['dtype', 'weak_type']
    array_abstraction_level = 3

    def __init__(self, dtype, weak_type=False):
        self.dtype = np.dtype(dtype)
        self.weak_type = weak_type
