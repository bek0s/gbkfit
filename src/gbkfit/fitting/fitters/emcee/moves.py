
import abc
import copy
from collections.abc import Callable
from typing import Any, Literal, Mapping, Sequence

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


def _locals_to_options(locals_: dict[str, Any]) -> dict[str, Any]:
    locals_ = copy.deepcopy(locals_)
    locals_.pop('self')
    locals_.pop('__class__')
    return locals_


class FitterEmceeMove(parseutils.TypedSerializable, abc.ABC):

    @classmethod
    def load(cls, info: dict[str, Any], *args, **kwargs) -> 'FitterEmceeMove':
        desc = parseutils.make_typed_desc(cls, 'emcee fitter move')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def dump(self) -> dict[str, Any]:
        info = copy.deepcopy(self._options)
        return dict(iterutils.remove_from_mapping_by_value(info, None))

    def __init__(self, cls: type, options: dict[str, Any]):
        self._options = copy.deepcopy(options)
        self._move = cls(**self._options)

    def obj(self) -> emcee.moves.Move:
        return self._move


class FitterEmceeMoveStretch(FitterEmceeMove):

    @staticmethod
    def type() -> str:
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
    def type() -> str:
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
    def type() -> str:
        return 'kde'

    def __init__(
            self,
            bw_method: str | int | float | Callable | None = None,
            nsplits: int = 2,
            randomize_split: bool = True,
            live_dangerously: bool = False
    ):
        super().__init__(
            emcee.moves.KDEMove, _locals_to_options(locals()))


class FitterEmceeMoveDE(FitterEmceeMove):

    @staticmethod
    def type() -> str:
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
    def type() -> str:
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
    def type() -> str:
        return 'mh'

    def __init__(
            self,
            proposal_function: Callable,
            ndim: int | None = None
    ):
        super().__init__(
            emcee.moves.MHMove, _locals_to_options(locals()))


class FitterEmceeMoveGaussian(FitterEmceeMove):

    @staticmethod
    def type() -> str:
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


def load_moves_with_weights(
        info: Mapping[str, Any] | Sequence[Mapping[str, Any]]
) -> tuple[tuple[FitterEmceeMove, float], ...]:
    info = iterutils.tuplify(info, False)
    weights = [m.pop('weight', 1.0) for m in info]
    moves = move_parser.load_many(info)
    return tuple(zip(moves, weights))


def dump_moves_with_weights(
        moves: tuple[tuple[FitterEmceeMove, int | float], ...]
) -> list[dict[str, Any]]:
    return [dict(weight=w) | move_parser.dump_one(m) for m, w in moves]
