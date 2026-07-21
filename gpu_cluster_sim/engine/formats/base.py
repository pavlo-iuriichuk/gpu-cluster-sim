"""Abstract export/import interface for `ClusterTopology`, plus a small
registry so new formats can be added without touching existing callers —
implement `TopologyFormat`, then call `register_format(MyFormat())` at the
bottom of the module (see `graphml.py`/`graphar.py`) and import that module
from `formats/__init__.py`.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from ..topology import ClusterTopology

_FORMATS: Dict[str, "TopologyFormat"] = {}


class TopologyFormat(ABC):
    """A serializer/deserializer for one on-disk representation of a
    `ClusterTopology`. `path` is a file path for single-file formats (e.g.
    GraphML) or a directory for chunked/multi-file formats (e.g. GraphAr).
    """

    name: str

    @abstractmethod
    def export(self, topology: "ClusterTopology", path: str) -> None:
        """Write `topology` to `path`."""

    @abstractmethod
    def import_(self, path: str) -> "ClusterTopology":
        """Read a `ClusterTopology` back from `path`."""


def register_format(fmt: TopologyFormat) -> None:
    _FORMATS[fmt.name] = fmt


def get_format(name: str) -> TopologyFormat:
    try:
        return _FORMATS[name]
    except KeyError:
        raise ValueError(f"Unknown topology format {name!r}. Available: {sorted(_FORMATS)}") from None


def available_formats() -> Dict[str, TopologyFormat]:
    return dict(_FORMATS)
