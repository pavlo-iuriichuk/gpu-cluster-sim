"""Flatten/reconstruct entity and link-metrics NamedTuples to and from
primitives (str, int, float) that every serialization format can store —
GraphML attributes and Parquet columns both reject nested/enum types.
Enums become their `.value`; tuples become a `|`-joined string. Both are
reversed on decode using the NamedTuple's own type hints, so a new model
field doesn't require any format-specific code.
"""

from enum import Enum
from typing import Any, Dict, Tuple, Type, get_origin, get_type_hints


def encode_namedtuple(obj: Tuple) -> Dict[str, Any]:
    encoded: Dict[str, Any] = {}
    for field, value in obj._asdict().items():
        if isinstance(value, Enum):
            encoded[field] = value.value
        elif isinstance(value, tuple):
            encoded[field] = "|".join(value)
        else:
            encoded[field] = value
    return encoded


def decode_namedtuple(cls: Type, data: Dict[str, Any]):
    hints = get_type_hints(cls)
    kwargs = {field: _decode_value(hints[field], data[field]) for field in cls._fields}
    return cls(**kwargs)


def _decode_value(hint: Type, raw: Any) -> Any:
    if get_origin(hint) is tuple:
        return tuple(raw.split("|")) if raw else ()
    if isinstance(hint, type) and issubclass(hint, Enum):
        return hint(raw)
    return raw
