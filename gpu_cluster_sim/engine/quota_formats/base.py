"""Abstract export/import interface for `QuotaPolicy`, plus a small registry
— the same shape as `engine.formats.base`, one level up. Add a new config
format by implementing `QuotaConfigFormat`, calling
`register_format(MyFormat())` at the bottom of the module (see
`yaml_format.py`), and importing that module from
`quota_formats/__init__.py`.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from ..quota_policy import QuotaPolicy

_FORMATS: Dict[str, "QuotaConfigFormat"] = {}


class QuotaConfigFormat(ABC):
    """A serializer/deserializer for one on-disk representation of a
    `QuotaPolicy`."""

    name: str

    @abstractmethod
    def export(self, policy: "QuotaPolicy", path: str) -> None:
        """Write `policy` to `path`."""

    @abstractmethod
    def import_(self, path: str) -> "QuotaPolicy":
        """Read a `QuotaPolicy` back from `path`."""


def register_format(fmt: QuotaConfigFormat) -> None:
    _FORMATS[fmt.name] = fmt


def get_format(name: str) -> QuotaConfigFormat:
    try:
        return _FORMATS[name]
    except KeyError:
        raise ValueError(f"Unknown quota config format {name!r}. Available: {sorted(_FORMATS)}") from None


def available_formats() -> Dict[str, QuotaConfigFormat]:
    return dict(_FORMATS)
