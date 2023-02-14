
import abc
import copy
from typing import Literal

import emcee.moves

from gbkfit.utils import iterutils, parseutils


__all__ = [
    'FitterEmceeMove',
    'FitterEmceeMoveDE',
    'FitterEmceeMoveDESnooker',
    'FitterEmceeMoveGaussian',
    'FitterEmceeMoveKDE',
    'FitterEmceeMoveMH',
    'FitterEmceeMoveStretch',
    'FitterEmceeMoveWalk',
    'move_parser',
    'dump_moves_with_weights',
    'load_moves_with_weights'
]


def _locals_to_options(locals_):
    locals_ = copy.deepcopy(locals_)
    locals_.pop('self')
    locals_.pop('__class__')
    return locals_


class FitterEmceeMove(parseutils.TypedParserSupport, abc.ABC):

    @classmethod
    def load(cls, info):
        desc = parseutils.make_typed_desc(cls, 'emcee fitter move')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def dump(self):
        info = copy.deepcopy(self._options)
        info = iterutils.remove_from_mapping_by_value(info, None)
        return info

    def __init__(self, cls, options):
        self._options = copy.deepcopy(options)
        self._move = cls(**self._options)

    def obj(self):
        return self._move


class FitterEmceeMoveStretch(FitterEmceeMove):

    @staticmethod
    def type():
        return 'stretch'

    def __init__(
            self,
            a: int | float = 2.0,
            nsplits: int = 2,
            randomize_split: bool = True,
            live_dangerously: bool = False
    ):
        super().__init__(
            emcee.moves.StretchMove, _locals_to_options(locals()))


class FitterEmceeMoveWalk(FitterEmceeMove):

    @staticmethod
    def type():
        return 'walk'

    def __init__(
            self,
            s: int | None = None,
            nsplits: int = 2,
            randomize_split: bool = True,
            live_dangerously: bool = False
    ):
        super().__init__(
            emcee.moves.WalkMove, _locals_to_options(locals()))


class FitterEmceeMoveKDE(FitterEmceeMove):

    @staticmethod
    def type():
        return 'kde'

    def __init__(
            self,
            bw_method=None,  # todo: add type
            nsplits: int = 2,
            randomize_split: bool = True,
            live_dangerously: bool = False
    ):
        super().__init__(
            emcee.moves.KDEMove, _locals_to_options(locals()))


class FitterEmceeMoveDE(FitterEmceeMove):

    @staticmethod
    def type():
        return 'de'

    def __init__(
            self,
            sigma: int | float = 1e-05,
            gamma0: int | float | None = None,
            nsplits: int = 2,
            randomize_split: bool = True,
            live_dangerously: bool = False
    ):
        super().__init__(
            emcee.moves.DEMove, _locals_to_options(locals()))


class FitterEmceeMoveDESnooker(FitterEmceeMove):

    @staticmethod
    def type():
        return 'desnooker'

    def __init__(
            self,
            gammas: int | float = 1.7,
            nsplits: int = 2,
            randomize_split: bool = True,
            live_dangerously: bool = False
    ):
        super().__init__(
            emcee.moves.DESnookerMove, _locals_to_options(locals()))


class FitterEmceeMoveMH(FitterEmceeMove):

    @staticmethod
    def type():
        return 'mh'

    def __init__(
            self,
            proposal_function,  # todo: add type
            ndim: int | None = None
    ):
        super().__init__(
            emcee.moves.MHMove, _locals_to_options(locals()))


class FitterEmceeMoveGaussian(FitterEmceeMove):

    @staticmethod
    def type():
        return 'gauss'

    def __init__(
            self,
            cov,  # todo: add type
            mode: Literal['vector', 'random', 'sequential'] = 'vector',
            factor: int | float | None = None
    ):
        super().__init__(
            emcee.moves.GaussianMove, _locals_to_options(locals()))


move_parser = parseutils.TypedParser(FitterEmceeMove, [
    FitterEmceeMoveStretch,
    FitterEmceeMoveWalk,
    FitterEmceeMoveKDE,
    FitterEmceeMoveDE,
    FitterEmceeMoveDESnooker,
    FitterEmceeMoveMH,
    FitterEmceeMoveGaussian])


def load_moves_with_weights(info):
    info = iterutils.tuplify(info, False)
    weights = [m.pop('weight', 1) for m in info]
    moves = move_parser.load_many(info)
    return tuple(zip(moves, weights))


def dump_moves_with_weights(moves):
    return [dict(weight=w) | move_parser.dump_one(m) for m, w in moves]
