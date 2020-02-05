
from . import iterutils


class SimpleParser:

    def __init__(self, cls):
        self._clstype = cls

    def load(self, x):
        return self.load_many(x) if iterutils.is_sequence(x) \
            else self.load_one(x)

    def load_one(self, x):
        return self._clstype.load(x)

    def load_many(self, x):
        return [self.load_one(item) for item in iterutils.listify(x)]

    @classmethod
    def dump(cls, x):
        return cls.dump_many(x) if iterutils.is_sequence(x) \
            else cls.dump_one(x)

    @classmethod
    def dump_many(cls, x):
        return [cls.dump_one(item) for item in iterutils.listify(x)]

    @staticmethod
    def dump_one(x):
        return x.dump() if x is not None else None


class TypedParser:

    def __init__(self, cls, noneval=None):
        self._clstype = cls
        self._clsname = cls.__name__
        self._parsers = {}
        self._noneval = noneval

    def type(self):
        return self._clsname

    def dict(self):
        return dict(self._parsers)

    def register(self, factory):
        type_ = factory.type()
        if type_ in self._parsers:
            raise RuntimeError(
                f"{self._clsname} parser is already registered: '{type_}'.")
        self._parsers[type_] = factory

    def load(self, x):
        return self.load_many(x) if iterutils.is_sequence(x) \
            else self.load_one(x)

    def load_many(self, x):
        return [self.load_one(item) for item in iterutils.listify(x)]

    def load_many2(self, x, *args, **kwargs):
        nitems = len(x)
        if any([nitems != len(arg) for arg in args + tuple(kwargs.values())]):
            raise RuntimeError("All arguments must have the same length.")
        nargs = len(args)
        args_list_shape = (nitems, nargs)
        args_list = iterutils.make_list(args_list_shape, [], True)
        for i in range(len(x)):
            for j, arg in enumerate(args):
                args_list[i][j] = arg[i]
        kwargs_list_shape = (nitems,)
        kwargs_list = iterutils.make_list(kwargs_list_shape, {}, True)
        for i in range(len(x)):
            for key, value in kwargs.items():
                kwargs_list[i][key] = value[i]
        results = []
        for item, item_args, item_kwargs in zip(x, args_list, kwargs_list):
            results.append(self.load_one(item, *item_args, **item_kwargs))
        return results

    def load_one(self, x, *args, **kwargs):
        if x is None:
            return self._noneval
        type_ = x['type']
        if type_ not in self._parsers:
            raise RuntimeError(
                f"{self._clsname} parser for type '{type_}' is not registered."
                f" Available parsers are: {list(self._parsers.keys())}.")
        return self._parsers[type_].load(x, *args, **kwargs)

    @classmethod
    def dump(cls, x):
        return cls.dump_many(x) if iterutils.is_sequence(x) \
            else cls.dump_one(x)

    @classmethod
    def dump_many(cls, x):
        return [cls.dump_one(item) for item in iterutils.listify(x)]

    @staticmethod
    def dump_one(x):
        return x.dump() if x is not None else None
