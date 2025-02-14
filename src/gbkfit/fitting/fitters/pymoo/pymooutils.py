# pymoo version: 0.5.0

import abc
import copy

from pymoo.operators.crossover.dex import DEX
from pymoo.operators.crossover.erx import EdgeRecombinationCrossover
from pymoo.operators.crossover.expx import ExponentialCrossover
from pymoo.operators.crossover.hux import HalfUniformCrossover
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.crossover.pcx import PCX
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.operators.mutation.bitflip import BinaryBitflipMutation
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.mutation.nom import NoMutation
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.operators.sampling.lhs import LatinHypercubeSampling
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.sampling.rnd import PermutationRandomSampling
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.util.termination.default import MultiObjectiveDefaultTermination
from pymoo.util.termination.default import SingleObjectiveDefaultTermination
from pymoo.util.termination.f_tol import MultiObjectiveSpaceToleranceTermination
from pymoo.util.termination.f_tol_single import SingleObjectiveSpaceToleranceTermination  # noqa: E501
from pymoo.util.termination.max_eval import MaximumFunctionCallTermination
from pymoo.util.termination.max_gen import MaximumGenerationTermination
from pymoo.util.termination.min_igd import IGDTermination
from pymoo.util.termination.max_time import TimeBasedTermination
from pymoo.util.termination.x_tol import DesignSpaceToleranceTermination

from gbkfit.utils import parseutils


__all__ = [
    'CrossoverSimulatedBinary',
    'CrossoverDEX',
    'CrossoverPCX',
    'CrossoverUniform',
    'CrossoverHalfUniform',
    'CrossoverExponential',
    'CrossoverKPoint',
    'CrossoverEdgeRecombination',
    'MutationNoMutation',
    'MutationPolynomial',
    'MutationBitflip',
    'MutationInversion',
    'SamplingRandomFloat',
    'SamplingLatinHypercube',
    'SamplingRandomBinary',
    'SamplingRandomPermutation',
    'SelectionRandom',
    'SelectionTournament',
    'TerminationMaximumFunctionCall',
    'TerminationMaximumGeneration',
    'TerminationIGD',
    'TerminationTimeBased',
    'TerminationDesignSpaceTolerance',
    'TerminationMultiObjectiveSpaceTolerance',
    'TerminationSingleObjectiveSpaceTolerance',
    'TerminationMultiObjectiveDefault',
    'TerminationSingleObjectiveDefault',
    'crossover_parser',
    'mutation_parser',
    'selection_parser',
    'termination_parser'
]


def _locals_to_options(locals_):
    locals_ = copy.deepcopy(locals_)
    locals_.pop('self')
    locals_.pop('__class__')
    return locals_


class _Base(parseutils.TypedSerializable, abc.ABC):

    @classmethod
    def load(cls, info, *args, **kwargs):
        type_name = kwargs.get('type_name')
        desc = parseutils.make_typed_desc(cls, f'pymoo {type_name}')
        opts = parseutils.parse_options_for_callable(info, desc, cls.__init__)
        return cls(**opts)

    def dump(self, *args, **kwargs):
        options = parseutils.prepare_for_dump(self._kwargs)
        return dict(type=self.type()) | options

    def __init__(self, type_, kwargs_):
        self._type = type_
        self._kwargs = kwargs_
        self._pymoo_object = type_(**kwargs_)

    def pymoo_object(self):
        return self._pymoo_object


class Crossover(_Base, abc.ABC):

    @classmethod
    def load(cls, info, *args, **kwargs):
        return super().load(info, type_name='crossover')


class CrossoverSimulatedBinary(Crossover):

    @staticmethod
    def type():
        return 'simulated_binary'

    def __init__(self, eta, n_offsprings=2, prob_per_variable=0.5, prob=0.9):
        super().__init__(
            SimulatedBinaryCrossover,
            _locals_to_options(locals()))


class CrossoverDEX(Crossover):

    @staticmethod
    def type():
        return 'dex'

    def __init__(
            self, F=None, cr=0.7, variant='bin', dither=None, jitter=False, # noqa
            n_diffs=1, n_iter=1, at_least_once=True, prob=0.9):
        options = _locals_to_options(locals())
        options.update(F=options.pop('f'), CR=options.pop('cr'))
        super().__init__(
            DEX,
            options)


class CrossoverPCX(Crossover):

    @staticmethod
    def type():
        return 'pcx'

    def __init__(self, eta=0.1, zeta=0.1, prob=0.9):
        super().__init__(
            PCX,
            _locals_to_options(locals()))


class CrossoverUniform(Crossover):

    @staticmethod
    def type():
        return 'uniform'

    def __init__(self, prob=0.9):
        super().__init__(
            UniformCrossover,
            _locals_to_options(locals()))


class CrossoverHalfUniform(Crossover):

    @staticmethod
    def type():
        return 'half_uniform'

    def __init__(self, prob_hux=0.5, prob=0.9):
        super().__init__(
            HalfUniformCrossover,
            _locals_to_options(locals()))


class CrossoverExponential(Crossover):

    @staticmethod
    def type():
        return 'exponential'

    def __init__(self, prob_exp=0.75, prob=0.9):
        super().__init__(
            ExponentialCrossover,
            _locals_to_options(locals()))


class CrossoverKPoint(Crossover):

    @staticmethod
    def type():
        return 'k_point'

    def __init__(self, n_points, prob=0.9):
        super().__init__(
            PointCrossover,
            _locals_to_options(locals()))


class CrossoverOrder(Crossover):

    @staticmethod
    def type():
        return 'order'

    def __init__(self, shift=False, prob=0.9):
        super().__init__(
            OrderCrossover,
            _locals_to_options(locals()))


class CrossoverEdgeRecombination(Crossover):

    @staticmethod
    def type():
        return 'edge_recombination'

    def __init__(self, prob=0.9):
        super().__init__(
            EdgeRecombinationCrossover,
            _locals_to_options(locals()))


class Mutation(_Base, abc.ABC):

    @classmethod
    def load(cls, info, *args, **kwargs):
        return super().load(info, type_name='mutation')


class MutationNoMutation(Mutation):

    @staticmethod
    def type():
        return 'none'

    def __init__(self):
        super().__init__(
            NoMutation,
            _locals_to_options(locals()))


class MutationPolynomial(Mutation):

    @staticmethod
    def type():
        return 'polynomial'

    def __init__(self, eta, prob=None):
        super().__init__(
            PolynomialMutation,
            _locals_to_options(locals()))


class MutationBitflip(Mutation):

    @staticmethod
    def type():
        return 'bitflip'

    def __init__(self, prob=None):
        super().__init__(
            BinaryBitflipMutation,
            _locals_to_options(locals()))


class MutationInversion(Mutation):

    @staticmethod
    def type():
        return 'inversion'

    def __init__(self, prob=1.0):
        super().__init__(
            InversionMutation,
            _locals_to_options(locals()))


class Sampling(_Base, abc.ABC):

    @classmethod
    def load(cls, info, *args, **kwargs):
        return super().load(info, type_name='sampling')


class SamplingRandomFloat(Sampling):

    @staticmethod
    def type():
        return 'random_float'

    def __init__(self):
        super().__init__(
            FloatRandomSampling,
            _locals_to_options(locals()))


class SamplingLatinHypercube(Sampling):

    @staticmethod
    def type():
        return 'latin_hypercube'

    def __init__(self):
        super().__init__(
            LatinHypercubeSampling,
            _locals_to_options(locals()))


class SamplingRandomBinary(Sampling):

    @staticmethod
    def type():
        return 'random_binary'

    def __init__(self):
        super().__init__(
            BinaryRandomSampling,
            _locals_to_options(locals()))


class SamplingRandomPermutation(Sampling):

    @staticmethod
    def type():
        return 'random_permutation'

    def __init__(self):
        super().__init__(
            PermutationRandomSampling,
            _locals_to_options(locals()))


class Selection(_Base, abc.ABC):

    @classmethod
    def load(cls, info, *args, **kwargs):
        return super().load(info, type_name='selection')


class SelectionRandom(Selection):

    @staticmethod
    def type():
        return 'random'

    def __init__(self):
        super().__init__(
            RandomSelection,
            _locals_to_options(locals()))


class SelectionTournament(Selection):

    @staticmethod
    def type():
        return 'tournament'

    @classmethod
    def load(cls, *args, **kwargs):
        raise NotImplementedError()

    def dump(self):
        raise NotImplementedError()

    def __init__(self, func_comp, pressure=2):
        super().__init__(
            TournamentSelection,
            _locals_to_options(locals()))


class Termination(_Base, abc.ABC):

    @classmethod
    def load(cls, info, *args, **kwargs):
        return super().load(info, type_name='termination')


class TerminationMaximumFunctionCall(Termination):

    @staticmethod
    def type():
        return 'n_eval'

    def __init__(self, n_max_evals):
        super().__init__(
            MaximumFunctionCallTermination,
            _locals_to_options(locals()))


class TerminationMaximumGeneration(Termination):

    @staticmethod
    def type():
        return 'n_iter'

    def __init__(self, n_max_gen):
        super().__init__(
            MaximumGenerationTermination,
            _locals_to_options(locals()))


class TerminationIGD(Termination):

    @staticmethod
    def type():
        return 'igd'

    def __init__(self, min_igd, pf):
        super().__init__(
            IGDTermination,
            _locals_to_options(locals()))


class TerminationTimeBased(Termination):

    @staticmethod
    def type():
        return 'time'

    def __init__(self, max_time):
        super().__init__(
            TimeBasedTermination,
            _locals_to_options(locals()))


class TerminationDesignSpaceTolerance(Termination):

    @staticmethod
    def type():
        return 'xtol'

    def __init__(
            self, n_last=20, tol=1e-6, nth_gen=1,
            n_max_gen=None, n_max_evals=None,
            truncate_metrics=True, truncate_data=True):
        super().__init__(
            DesignSpaceToleranceTermination,
            _locals_to_options(locals()))


class TerminationMultiObjectiveSpaceTolerance(Termination):

    @staticmethod
    def type():
        return 'ftol'

    def __init__(
            self, tol=0.0025, n_last=30, nth_gen=5,
            n_max_gen=None, n_max_evals=None,
            truncate_metrics=True, truncate_data=True):
        super().__init__(
            MultiObjectiveSpaceToleranceTermination,
            _locals_to_options(locals()))


class TerminationSingleObjectiveSpaceTolerance(Termination):

    @staticmethod
    def type():
        return 'ftol_s'

    def __init__(
            self, tol=1e-6, n_last=20, nth_gen=1,
            n_max_gen=None, n_max_evals=None,
            truncate_metrics=True, truncate_data=True):
        super().__init__(
            SingleObjectiveSpaceToleranceTermination,
            _locals_to_options(locals()))


class TerminationMultiObjectiveDefault(Termination):

    @staticmethod
    def type():
        return 'default_multi'

    def __init__(
            self, x_tol=1e-8, cv_tol=1e-6, f_tol=0.0025, nth_gen=5,
            n_last=30, n_max_gen=1000, n_max_evals=100000,
            truncate_metrics=True, truncate_data=True):
        super().__init__(
            MultiObjectiveDefaultTermination,
            _locals_to_options(locals()))


class TerminationSingleObjectiveDefault(Termination):

    @staticmethod
    def type():
        return 'default_single'

    def __init__(
            self, x_tol=1e-8, cv_tol=1e-6, f_tol=1e-6, nth_gen=5,
            n_last=20, n_max_gen=1000, n_max_evals=100000,
            truncate_metrics=True, truncate_data=True):
        super().__init__(
            SingleObjectiveDefaultTermination,
            _locals_to_options(locals()))


crossover_parser = parseutils.TypedParser(Crossover, [
    CrossoverSimulatedBinary,
    CrossoverDEX,
    CrossoverPCX,
    CrossoverUniform,
    CrossoverHalfUniform,
    CrossoverExponential,
    CrossoverKPoint,
    CrossoverEdgeRecombination])

mutation_parser = parseutils.TypedParser(Mutation, [
    MutationNoMutation,
    MutationPolynomial,
    MutationBitflip,
    MutationInversion])

selection_parser = parseutils.TypedParser(Selection, [
    SelectionRandom,
    SelectionTournament])

termination_parser = parseutils.TypedParser(Termination, [
    TerminationMaximumFunctionCall,
    TerminationMaximumGeneration,
    TerminationIGD,
    TerminationTimeBased,
    TerminationDesignSpaceTolerance,
    TerminationMultiObjectiveSpaceTolerance,
    TerminationSingleObjectiveSpaceTolerance,
    TerminationMultiObjectiveDefault,
    TerminationSingleObjectiveDefault])
