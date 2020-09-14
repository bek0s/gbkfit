
import abc
import copy

import emcee.moves

from gbkfit.utils import parseutils


class FitterEmceeMove(parseutils.TypedParserSupport, abc.ABC):

    @classmethod
    def load(cls, info):
        desc = f'{cls.type()} emcee fitter move (class={cls.__qualname__})'
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def dump(self):
        return copy.deepcopy(self._kwargs)

    def __init__(self, cls, kwargs):
        self._kwargs = copy.deepcopy(kwargs)
        self._kwargs.pop('self')
        self._kwargs.pop('__class__')
        self._move = cls(**self._kwargs)

    def move_obj(self):
        return self._move


class FitterEmceeMoveStretch(FitterEmceeMove):

    @staticmethod
    def type():
        return 'stretch'

    def __init__(
            self, a=2.0,
            nsplits=2, randomize_split=True, live_dangerously=False):
        super().__init__(emcee.moves.StretchMove, locals())


class FitterEmceeMoveWalk(FitterEmceeMove):

    @staticmethod
    def type():
        return 'walk'

    def __init__(
            self, s=None,
            nsplits=2, randomize_split=True, live_dangerously=False):
        super().__init__(emcee.moves.WalkMove, locals())


class FitterEmceeMoveKDE(FitterEmceeMove):

    @staticmethod
    def type():
        return 'kde'

    def __init__(
            self, bw_method=None,
            nsplits=2, randomize_split=True, live_dangerously=False):
        super().__init__(emcee.moves.KDEMove, locals())


class FitterEmceeMoveDE(FitterEmceeMove):

    @staticmethod
    def type():
        return 'de'

    def __init__(
            self, sigma=1e-05, gamma0=None,
            nsplits=2, randomize_split=True, live_dangerously=False):
        super().__init__(emcee.moves.DEMove, locals())


class FitterEmceeMoveDESnooker(FitterEmceeMove):

    @staticmethod
    def type():
        return 'desnooker'

    def __init__(
            self, gammas=1.7,
            nsplits=2, randomize_split=True, live_dangerously=False):
        super().__init__(emcee.moves.DESnookerMove, locals())


class FitterEmceeMoveMH(FitterEmceeMove):

    @staticmethod
    def type():
        return 'mh'

    def __init__(self, proposal_function, ndim=None):
        super().__init__(emcee.moves.MHMove, locals())


class FitterEmceeMoveGaussian(FitterEmceeMove):

    @staticmethod
    def type():
        return 'gauss'

    def __init__(self, cov, mode='vector', factor=None):
        super().__init__(emcee.moves.GaussianMove, locals())


parser = parseutils.TypedParser(FitterEmceeMove)
parser.register(FitterEmceeMoveStretch)
parser.register(FitterEmceeMoveWalk)
parser.register(FitterEmceeMoveKDE)
parser.register(FitterEmceeMoveDE)
parser.register(FitterEmceeMoveDESnooker)
parser.register(FitterEmceeMoveMH)
parser.register(FitterEmceeMoveGaussian)
