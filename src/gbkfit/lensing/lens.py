"""
Gravitational lenses (see Lens): flat bendings of the light at a
redshift, with no light of their own, made of pieces (see
gbkfit.lensing.pieces).
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
from annotated_types import Gt

from gbkfit.config import (
    BasicParser, Serializable, TypedParser, check_name, check_options)
from gbkfit.cosmology import Cosmology
from gbkfit.params import ParamScalarDesc
from gbkfit.utils import iterutils
from .galaxy_mass import LensGalaxyMass
from .pieces import (
    Deflects, GalaxyBending, LensDeflectionMap, LensEPL, LensNFW, LensPiece,
    LensSIE, LensShear, LensSheet, Normalisation, OfGalaxy, SkyPoints)
from .planes import PLANE_LINKING_VELOCITY, velocity_separation

if TYPE_CHECKING:
    from gbkfit.driver import Driver


__all__ = [
    'Lens',
    'lens_parser',
    'lens_piece_parser',
    'quoted_factor'
]


lens_piece_parser = TypedParser(LensPiece, [
    LensDeflectionMap, LensEPL, LensGalaxyMass, LensNFW, LensSIE, LensShear,
    LensSheet], family='lens_piece')


class Lens(Serializable):
    """
    A gravitational lens: a flat bending of the light that passes its
    plane, at a redshift, with no light of its own (a galaxy's light is
    that of a galaxy at the redshift; see Galaxy). It is made of pieces
    (see LensPiece), whose deflections add up, and bends the light of the
    galaxies behind it, of every observation of them (see Tracer).

    The strengths of its quoted pieces (see Normalisation; e.g. the
    Einstein radius of an SIE) are those of a source at the
    source_redshift of each piece, or else at its own, as lens models
    quote them; the deflection of the light of a source at another
    redshift is scaled by D_ls / D_s from them (see quoted_factors). Its
    other pieces are of D_ls / D_s = 1. A
    piece that is the mass of a galaxy (see OfGalaxy and galaxies) bends
    light by that mass laid out on the sky, which its deflection takes
    (see deflection).

    Its parameters are those of its pieces, each prefixed by its name,
    one piece alone too ('main_theta_e', 'shear_gamma'; a piece is named
    after its type by default: 'sie_theta_e'); its name prefixes them all
    among those of the galaxies and the observations (e.g.
    'lens_main_theta_e').

    Parameters
    ----------
    name : str
        Its name.
    redshift : float
        Its redshift: that of the plane of its pieces (above 0).
    pieces : LensPiece or Sequence of LensPiece
        Its pieces; of different names when there are several (each is
        named after its type by default, see LensPiece).
    source_redshift : float, optional
        The redshift of the source whose light the strengths of its
        quoted pieces are those of (see Normalisation), of those without
        one of their own (see LensPiece); required if one has none, and
        only then.

    Raises
    ------
    ValueError
        If the name is not valid, there are no pieces, several pieces of
        one name, a piece that is not quoted has a source_redshift, the
        source_redshift of the lens is missing where a quoted piece has
        none or given where none takes it, or a source_redshift is not
        behind the plane of the lens (more than PLANE_LINKING_VELOCITY
        behind its redshift).
    """

    config_titles = dict(
        name='Name', redshift='Redshift', pieces='Pieces',
        source_redshift='Source redshift')

    def dump(
            self, prefix: str = '', dump_path: bool = True,
            overwrite: bool = False
    ) -> dict[str, Any]:
        count = len(self._pieces)
        info = dict(
            name=self._name, redshift=self._redshift,
            pieces=lens_piece_parser.dump(
                list(self._pieces),
                prefix=[f'{prefix}{self._name}_{p}' for p in self._prefixes],
                dump_path=[dump_path] * count,
                overwrite=[overwrite] * count))
        if self._source_redshift is not None:
            info['source_redshift'] = self._source_redshift
        return info

    def __init__(
            self,
            name: str,
            redshift: Annotated[float, Gt(0)],
            pieces: LensPiece | Sequence[LensPiece],
            source_redshift: Annotated[float, Gt(0)] | None = None
    ):
        check_options(Lens.__init__, locals())
        check_name(name)
        pieces = iterutils.tuplify(pieces)
        if not pieces:
            raise ValueError(f"lens {name!r} needs at least one piece")
        if neither := [
                piece.type() for piece in pieces
                if not isinstance(piece, (Deflects, OfGalaxy))]:
            raise TypeError(
                f"the pieces {neither} of lens {name!r} neither deflect by "
                f"their parameters (Deflects) nor are the mass of a galaxy "
                f"(OfGalaxy)")
        names = [piece.name() for piece in pieces]
        if len(set(names)) != len(names):
            raise ValueError(
                f"the pieces of lens {name!r} repeat names: {names} (a "
                f"piece is named after its type unless given a name)")
        quoted = [
            piece for piece in pieces
            if piece.normalisation() is Normalisation.QUOTED]
        if unquoted := [
                piece.name() for piece in pieces
                if piece not in quoted and piece.source_redshift() is not None]:
            raise ValueError(
                f"the pieces {unquoted} of lens {name!r} are not quoted for "
                f"a source; their source_redshift says nothing")
        # (the quoted pieces of no source_redshift of their own: of that
        # of the lens)
        if (taking := [
                piece.name() for piece in quoted
                if piece.source_redshift() is None]) \
                and source_redshift is None:
            raise ValueError(
                f"the pieces {taking} of lens {name!r} have strengths quoted "
                f"for a source: give the redshift of that source "
                f"(source_redshift, of the lens or of each piece)")
        if not taking and source_redshift is not None:
            raise ValueError(
                f"no piece of lens {name!r} takes its source_redshift (none "
                f"is quoted for a source, or each has its own); it says "
                f"nothing")
        for what, value in [
                (f"lens {name!r}", source_redshift),
                *((f"piece {piece.name()!r} of lens {name!r}",
                   piece.source_redshift()) for piece in quoted)]:
            if value is not None and velocity_separation(
                    redshift, value) <= PLANE_LINKING_VELOCITY:
                raise ValueError(
                    f"the source_redshift of the {what} ({value}) must be "
                    f"behind its plane: more than "
                    f"{PLANE_LINKING_VELOCITY:g} km/s behind its redshift "
                    f"({redshift})")
        self._name = name
        self._redshift = redshift
        self._pieces = pieces
        self._source_redshift = source_redshift
        # (each prefixed by its name, one alone too: adding a piece leaves
        # the names of the others as they were)
        self._prefixes = [f'{piece_name}_' for piece_name in names]
        self._pdescs, self._mappings = iterutils.merge_with_prefixes(
            [piece.pdescs() for piece in pieces], self._prefixes)

    def name(self) -> str:
        """Return its name."""
        return self._name

    def redshift(self) -> float:
        """Return its redshift."""
        return self._redshift

    def source_redshift(self) -> float | None:
        """
        Return the redshift of the source of its quoted pieces without one
        of their own, if any (see source_redshift_of).
        """
        return self._source_redshift

    def source_redshift_of(self, piece: LensPiece) -> float | None:
        """
        Return the redshift of the source the strength of one of its
        pieces is quoted for: its own, or else that of the lens; None if
        it is not quoted (see Normalisation).
        """
        if piece.normalisation() is not Normalisation.QUOTED:
            return None
        own = piece.source_redshift()
        return self._source_redshift if own is None else own

    def pieces(self) -> tuple[LensPiece, ...]:
        """Return its pieces."""
        return self._pieces

    def galaxies(self) -> tuple[str, ...]:
        """
        Return the names of the galaxies whose masses its pieces are (see
        OfGalaxy), in the order of its pieces.
        """
        return tuple(
            piece.galaxy() for piece in self._pieces
            if isinstance(piece, OfGalaxy))

    def pdescs(self) -> dict[str, ParamScalarDesc]:
        """
        Return its parameters, by name: those of its pieces, with the name
        of each piece and '_' before them when it has several (e.g.
        'shear_gamma').
        """
        return dict(self._pdescs)

    def quoted_factors(
            self, redshift: float, cosmology: Cosmology
    ) -> tuple[float, ...]:
        """
        Return the factor that turns the deflection of each of its pieces
        into that of D_ls / D_s = 1, at the distance of the given redshift
        (that of its plane; see plane_layout): of a quoted piece (see
        Normalisation), D_s / D_ls of its source (see source_redshift_of);
        1 of the others.

        Parameters
        ----------
        redshift : float
            The redshift of its distance.
        cosmology : Cosmology
            The cosmology of the distances.

        Returns
        -------
        tuple of float
            The factors, in the order of its pieces.

        Raises
        ------
        ValueError
            If the source of a quoted piece is not behind the redshift.
        """
        def factor(piece: LensPiece) -> float:
            source = self.source_redshift_of(piece)
            if source is None:
                return 1.0
            if not source > redshift:
                raise ValueError(
                    f"the source of the quoted piece {piece.name()!r} of lens "
                    f"{self._name!r} (at the redshift {source}) is not "
                    f"behind its plane (at {redshift})")
            return quoted_factor(redshift, source, cosmology)
        return tuple(factor(piece) for piece in self._pieces)

    def deflection(
            self,
            points: SkyPoints,
            params: Mapping[str, float],
            quoted_factors: Sequence[float],
            driver: 'Driver',
            galaxies: Mapping[str, GalaxyBending] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return its deflection at points of its plane, at D_ls / D_s = 1
        (alpha-hat): the sum of those of its pieces, each times the factor
        that turns it into it (see quoted_factors).

        Parameters
        ----------
        points : SkyPoints
            The points.
        params : Mapping of str to float
            The values of its parameters, by their names (see pdescs).
        quoted_factors : Sequence of float
            The factor of each of its pieces (see quoted_factors).
        driver : Driver
            The driver whose native kernels compute the deflections of its
            pieces that have any (see Deflects).
        galaxies : Mapping of str to GalaxyBending, optional
            The bending of the light by the mass of each galaxy whose mass
            its pieces are (see galaxies and OfGalaxy), by name (others may
            be there too); none by default, for a lens of no such pieces.

        Returns
        -------
        tuple of np.ndarray
            The deflection (arcsec) along x and y, of the shape of the
            points.

        Raises
        ------
        ModelEvaluationError
            If a piece has no deflection at a point (e.g. off a map).
        ValueError
            If the bending by the mass of the galaxy of a piece is
            missing.
        """
        galaxies = {} if galaxies is None else galaxies
        alpha_x = np.zeros(np.shape(points.x))
        alpha_y = np.zeros(np.shape(points.y))
        for piece, names, scale in zip(
                self._pieces, self._mappings, quoted_factors):
            if isinstance(piece, OfGalaxy):
                if (bending := galaxies.get(piece.galaxy())) is None:
                    raise ValueError(
                        f"the piece {piece.name()!r} of lens "
                        f"{self._name!r} bends light by the mass of the "
                        f"galaxy {piece.galaxy()!r}, which is not given")
                piece_x, piece_y = bending.deflection(points)
            else:
                own = {name: params[full] for name, full in names.items()}
                piece_x, piece_y = piece.deflection(points, own, driver)
            alpha_x += scale * piece_x
            alpha_y += scale * piece_y
        return alpha_x, alpha_y


def quoted_factor(
        redshift: float, source_redshift: float, cosmology: Cosmology
) -> float:
    """
    Return the factor that turns a deflection quoted for a source at a
    redshift into that of D_ls / D_s = 1, of a plane at another, in front
    of it: D_s / D_ls (see Lens.quoted_factors).
    """
    return cosmology.angular_diameter_distance(source_redshift) \
        / cosmology.angular_diameter_distance_between(
            redshift, source_redshift)


lens_parser = BasicParser(Lens)
