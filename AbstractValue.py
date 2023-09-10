from typing import (List)


class AbstractValue:
    __slots__: List[str] = []

    def at_least_vspace(self):
        raise NotImplementedError("must override")

    def __repr__(self):
        try:
            kv_pairs = (f'{k}={v}' for k, v in self.__dict__.items())
            return '{}({})'.format(self.__class__.__name__, ','.join(kv_pairs))
        except AttributeError:
            return self.__class__.__name__

    def strip_weak_type(self):
        return self

    def strip_named_shape(self):
        return self

    def join(self, other):
        raise NotImplementedError("must override")

    def update(self, **kwargs):
        raise NotImplementedError("must override")

    def str_short(self, short_dtypes=False):
        return str(self)
