"""
The extra outputs of an evaluation of the model: what it computes on the
way to the model data (e.g. the light and the velocity of each component
on its grid, the rotation curves of the gravity, the cubes of the
instrument), each with a title and a description for those who show
them.
"""

from collections.abc import Iterator
from typing import Any, NamedTuple


__all__ = [
    'Extra',
    'Extras'
]


class Extra(NamedTuple):
    """
    An extra output.

    Attributes
    ----------
    name : str
        What it is (e.g. 'vdata'; unique among those of the same ids).
    ids : dict of str to str
        What it is of: e.g. 'observation', 'galaxy', 'group' (of galaxies
        along the line of sight) and 'component', by their names (or their
        positions: e.g. 'observation0', 'galaxy0', 'component0').
    value : Any
        Its value (e.g. data on a grid, an array, a table).
    title : str
        A short title (e.g. for the caption of a figure of it).
    description : str
        What it is, in a few sentences.
    """
    name: str
    ids: dict[str, str]
    value: Any
    title: str
    description: str


class Extras:
    """
    The extra outputs of an evaluation: values by name and the ids of what
    they are of, with a title and a description (see Extra).

    Each part of the model adds its own outputs to a view of the extras
    scoped to the ids of what it evaluates (see scoped), which adds them
    to each output; e.g. the plan of an observation group gives each
    observation extras.scoped(observation=...), and the observation gives
    each of its galaxies a view scoped to it too.

    Examples
    --------
    >>> extras = Extras()
    >>> group.plan().model_h(values, extras)
    >>> extras.get('vdata', component='disk')
    """

    def __init__(self) -> None:
        self._extras: dict[tuple[str, tuple[tuple[str, str], ...]], Extra] \
            = {}
        self._scope: dict[str, str] = {}

    def scoped(self, **ids: str) -> 'Extras':
        """
        Return a view of the extras that adds the given ids (and those of
        this view) to the outputs added to it.
        """
        view = Extras.__new__(Extras)
        view._extras = self._extras
        view._scope = self._scope | ids
        return view

    def add(
            self,
            name: str,
            value: Any,
            title: str,
            description: str,
            **ids: str
    ) -> None:
        """
        Add an output, of the ids of the view and the given ones (it
        replaces one of the same name and ids).
        """
        ids = self._scope | ids
        key = (name, tuple(sorted(ids.items())))
        self._extras[key] = Extra(name, ids, value, title, description)

    def find(self, name: str | None = None, **ids: str) -> list[Extra]:
        """
        Return the outputs of a name (any, if None) with the given ids (they
        may have others), in the order they were added.
        """
        return [
            extra for extra in self._extras.values()
            if (name is None or extra.name == name)
            and all(extra.ids.get(k) == v for k, v in ids.items())]

    def get(self, name: str, **ids: str) -> Any:
        """
        Return the value of the one output of a name with the given ids
        (it may have others).

        Raises
        ------
        LookupError
            If there is none, or several.
        """
        found = self.find(name, **ids)
        if len(found) != 1:
            raise LookupError(
                f"there are {len(found)} extra outputs {name!r} with the ids "
                f"{ids}; one was asked for (there are: "
                f"{sorted({(e.name, *e.ids.values()) for e in self})})")
        return found[0].value

    def __iter__(self) -> Iterator[Extra]:
        """Iterate over all the outputs (of any scope)."""
        return iter(list(self._extras.values()))

    def __len__(self) -> int:
        return len(self._extras)
