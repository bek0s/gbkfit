
import abc
import copy

import emcee.moves

from gbkfit.utils import iterutils, parseutils


__all__ = [
    'FitterEmceeMoveDE',
    'FitterEmceeMoveDESnooker',
    'FitterEmceeMoveGaussian',
    'FitterEmceeMoveKDE',
    'FitterEmceeMoveMH',
    'FitterEmceeMoveStretch',
    'FitterEmceeMoveWalk',
    'move_parser',
    'dump_moves_with_weights',
    'load_moves_with_weights']


class FitterEmceeMove(parseutils.TypedParserSupport, abc.ABC):

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'emcee fitter move')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def dump(self):
        return copy.deepcopy(self._kwargs)

    def __init__(self, cls, kwargs):
        self._kwargs = copy.deepcopy(kwargs)
        self._kwargs.pop('self')
        self._kwargs.pop('__class__')
        self._move = cls(**self._kwargs)

    def obj(self):
        return self._move


class FitterEmceeMoveStretch(FitterEmceeMove):

    @staticmethod
    def type():
        return 'stretch'

    def __init__(
            self, a=2.0,
            nsplits=2, randomize_split=True, live_dangerously=False):
        super().__init__(emcee.moves.StretchMove, copy.deepcopy(locals()))


class FitterEmceeMoveWalk(FitterEmceeMove):

    @staticmethod
    def type():
        return 'walk'

    def __init__(
            self, s=None,
            nsplits=2, randomize_split=True, live_dangerously=False):
        super().__init__(emcee.moves.WalkMove, copy.deepcopy(locals()))


class FitterEmceeMoveKDE(FitterEmceeMove):

    @staticmethod
    def type():
        return 'kde'

    def __init__(
            self, bw_method=None,
            nsplits=2, randomize_split=True, live_dangerously=False):
        super().__init__(emcee.moves.KDEMove, copy.deepcopy(locals()))


class FitterEmceeMoveDE(FitterEmceeMove):

    @staticmethod
    def type():
        return 'de'

    def __init__(
            self, sigma=1e-05, gamma0=None,
            nsplits=2, randomize_split=True, live_dangerously=False):
        super().__init__(emcee.moves.DEMove, copy.deepcopy(locals()))


class FitterEmceeMoveDESnooker(FitterEmceeMove):

    @staticmethod
    def type():
        return 'desnooker'

    def __init__(
            self, gammas=1.7,
            nsplits=2, randomize_split=True, live_dangerously=False):
        super().__init__(emcee.moves.DESnookerMove, copy.deepcopy(locals()))


class FitterEmceeMoveMH(FitterEmceeMove):

    @staticmethod
    def type():
        return 'mh'

    def __init__(self, proposal_function, ndim=None):
        super().__init__(emcee.moves.MHMove, copy.deepcopy(locals()))


class FitterEmceeMoveGaussian(FitterEmceeMove):

    @staticmethod
    def type():
        return 'gauss'

    def __init__(self, cov, mode='vector', factor=None):
        super().__init__(emcee.moves.GaussianMove, copy.deepcopy(locals()))


move_parser = parseutils.TypedParser(FitterEmceeMove)
move_parser.register(FitterEmceeMoveStretch)
move_parser.register(FitterEmceeMoveWalk)
move_parser.register(FitterEmceeMoveKDE)
move_parser.register(FitterEmceeMoveDE)
move_parser.register(FitterEmceeMoveDESnooker)
move_parser.register(FitterEmceeMoveMH)
move_parser.register(FitterEmceeMoveGaussian)


def load_moves_with_weights(info):
    moves = iterutils.tuplify(info, False)
    weights = [move.pop('weight', 1) for move in moves]
    moves = move_parser.load(moves)
    return tuple(zip(moves, weights))


def dump_moves_with_weights(moves):
    return [move.dump() | dict(weight=weight) for move, weight in moves]
