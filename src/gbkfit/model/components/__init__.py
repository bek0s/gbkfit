from ._functions import GalaxyFunctions
from ._stage import GalaxyMass, GalaxyStage, GalaxyState
from ._units import DataUnit, LightUnit
from .base import *
from .extinction import *
from .geometries import *
from .lines import *
from .nodes import *
from .particles import *
from .parts import *
from .points import *
from .roles import *
from .scenes import *

from . import cones, disks, spheroids, traits
from ._nodal import TraitTable
from .cones import Cone
from .disks import Clouds, Disk, LineOpacity, ThickDisk, ThinDisk
from .spheroids import (
    JeansKinematics, LightMGE, LightSersic, SchwarzschildKinematics,
    Spheroid, SpheroidKinematics, SpheroidTraitsKinematics)

from .cones._detail import CONE_TRAITS as _CONE_TRAITS
from .disks._detail import (
    THICK_DISK_TRAITS as _THICK_DISK_TRAITS,
    THIN_DISK_TRAITS as _THIN_DISK_TRAITS)
from .spheroids._kinematics import SPHEROID_TRAITS as _SPHEROID_TRAITS

# The traits that each type of component takes, by its type (see
# TraitTable; of a spheroid, those of its kinematics of traits)
TRAIT_TABLES: dict[str, TraitTable] = dict(
    thin_disk=_THIN_DISK_TRAITS, thick_disk=_THICK_DISK_TRAITS,
    cone=_CONE_TRAITS, spheroid=_SPHEROID_TRAITS)
