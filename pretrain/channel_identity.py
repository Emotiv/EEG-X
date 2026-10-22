"""Versioned EEG channel identities shared by pretraining and finetuning."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from dataclasses import dataclass
from enum import Enum
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


PRETRAIN_ROOT = Path(__file__).resolve().parent
STEEGFORMER_VOCABULARY_PATH = PRETRAIN_ROOT / "senloc_file" / "sen_chan_idx.pkl"
EEGX2_VOCABULARY_PATH = PRETRAIN_ROOT / "senloc_file" / "senloc.json"

EMOTIV_EPOC_CHANNEL_NAMES = (
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
)

LEGACY_CHANNEL_EMBEDDING_CAPACITY = 145


class ChannelIndexing(str, Enum):
    """Supported policies for turning dataset channels into model identities."""

    DATASET_CONFIG = "dataset_config"
    EMOTIV_ORDER = "emotiv_order"
    STEEGFORMER = "steegformer"
    EEGX_CATALOG = "steegformer"
    EEGX2 = "eegx2"


DEFAULT_CHANNEL_INDEXING = ChannelIndexing.DATASET_CONFIG.value
CHANNEL_INDEXING_CHOICES = (
    ChannelIndexing.DATASET_CONFIG.value,
    ChannelIndexing.EMOTIV_ORDER.value,
    ChannelIndexing.STEEGFORMER.value,
    "eegx_catalog",
    ChannelIndexing.EEGX2.value,
)


class ChannelVocabularyError(ValueError):
    """Raised when a channel-vocabulary artifact violates its schema."""


def normalize_channel_indexing(
    channel_indexing: str | ChannelIndexing,
) -> ChannelIndexing:
    """Normalize a CLI or API value into the typed indexing policy."""
    if isinstance(channel_indexing, ChannelIndexing):
        return channel_indexing
    normalized = str(channel_indexing).strip().lower()
    if normalized == "eegx_catalog":
        normalized = ChannelIndexing.STEEGFORMER.value
    try:
        return ChannelIndexing(normalized)
    except ValueError as error:
        raise ValueError(
            f"Unknown channel indexing policy {channel_indexing!r}; "
            f"expected one of {CHANNEL_INDEXING_CHOICES}."
        ) from error


def channel_embedding_capacity(
    channel_indexing: str | ChannelIndexing,
) -> int:
    """Return the learned channel-table capacity required by a policy."""
    policy = normalize_channel_indexing(channel_indexing)
    if policy is ChannelIndexing.EEGX2:
        return load_channel_vocabulary().embedding_capacity
    return LEGACY_CHANNEL_EMBEDDING_CAPACITY


def normalize_channel_name(channel_name: object) -> str:
    """Return the case-insensitive, whitespace-insensitive identity key."""
    return "".join(str(channel_name).strip().split()).upper()


def sha256_file(path: str | Path) -> str:
    """Return a deterministic digest of a vocabulary or source artifact."""
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@lru_cache(maxsize=8)
def _load_steegformer_channel_mapping_cached(
    resolved_path: str,
) -> dict[str, int]:
    path = Path(resolved_path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a mapping in {path}")
    raw_mapping = payload.get("channels_mapping")
    if not isinstance(raw_mapping, Mapping):
        raise ValueError(f"channels_mapping missing from {path}")
    return {
        normalize_channel_name(channel_name): int(channel_index)
        for channel_name, channel_index in raw_mapping.items()
    }


def load_steegformer_channel_mapping(
    path: str | Path = STEEGFORMER_VOCABULARY_PATH,
) -> dict[str, int]:
    """Load the frozen historical STEEGFormer name-to-row catalog."""
    return _load_steegformer_channel_mapping_cached(str(Path(path).expanduser().resolve()))


def _position(raw_position: object, *, context: str) -> tuple[float, float, float]:
    if not isinstance(raw_position, Sequence) or isinstance(raw_position, (str, bytes)):
        raise ChannelVocabularyError(f"{context} position must contain three numbers")
    if len(raw_position) != 3:
        raise ChannelVocabularyError(f"{context} position must contain three numbers")
    try:
        position = tuple(float(value) for value in raw_position)
    except (TypeError, ValueError) as error:
        raise ChannelVocabularyError(f"{context} position must contain three numbers") from error
    if not all(math.isfinite(value) for value in position):
        raise ChannelVocabularyError(f"{context} position values must be finite")
    return position  # type: ignore[return-value]


def _index(raw_index: object, *, context: str, capacity: int) -> int:
    if isinstance(raw_index, bool) or not isinstance(raw_index, int):
        raise ChannelVocabularyError(f"{context} index must be an integer")
    index = int(raw_index)
    if not 0 <= index < capacity:
        raise ChannelVocabularyError(
            f"{context} index {index} is outside embedding capacity {capacity}"
        )
    return index


@dataclass(frozen=True)
class ChannelAlias:
    name: str
    reserved_index: int
    position: tuple[float, float, float]


@dataclass(frozen=True)
class ChannelRecord:
    name: str
    index: int
    position: tuple[float, float, float]
    aliases: tuple[ChannelAlias, ...]


@dataclass(frozen=True)
class ChannelVocabulary:
    schema_version: int
    embedding_capacity: int
    channels: tuple[ChannelRecord, ...]
    reserved_indices: tuple[int, ...]
    fingerprint: str
    source_path: Path

    @cached_property
    def _accepted_names(self) -> dict[str, ChannelRecord]:
        accepted: dict[str, ChannelRecord] = {}
        for record in self.channels:
            accepted[normalize_channel_name(record.name)] = record
            for alias in record.aliases:
                accepted[normalize_channel_name(alias.name)] = record
        return accepted

    @property
    def active_identity_count(self) -> int:
        return len(self.channels)

    @property
    def accepted_name_count(self) -> int:
        return len(self._accepted_names)

    def resolve_names(self, channel_names: Iterable[object]) -> np.ndarray:
        indices: list[int] = []
        missing: list[str] = []
        for raw_name in channel_names:
            record = self._accepted_names.get(normalize_channel_name(raw_name))
            if record is None:
                missing.append(str(raw_name))
            else:
                indices.append(record.index)
        if missing:
            raise ChannelVocabularyError(f"Channels missing from {self.source_path}: {missing}")
        return np.asarray(indices, dtype=np.int64)

    def canonical_names(self, channel_names: Iterable[object]) -> tuple[str, ...]:
        names: list[str] = []
        missing: list[str] = []
        for raw_name in channel_names:
            record = self._accepted_names.get(normalize_channel_name(raw_name))
            if record is None:
                missing.append(str(raw_name))
            else:
                names.append(record.name)
        if missing:
            raise ChannelVocabularyError(f"Channels missing from {self.source_path}: {missing}")
        return tuple(names)

    @cached_property
    def _active_positions(self) -> np.ndarray:
        return np.asarray(
            [record.position for record in self.channels],
            dtype=np.float64,
        )

    def _nearest_position_rows(
        self,
        positions: object,
        *,
        max_distance: float,
    ) -> np.ndarray:
        query = np.asarray(positions, dtype=np.float64)
        if query.ndim != 2 or query.shape[1] != 3:
            raise ChannelVocabularyError(
                f"Expected channel positions shaped [C, 3], got {query.shape}"
            )
        if not np.isfinite(query).all():
            raise ChannelVocabularyError("Channel positions must be finite")
        distances = np.linalg.norm(
            query[:, None, :] - self._active_positions[None, :, :],
            axis=-1,
        )
        nearest = distances.argmin(axis=1)
        nearest_distances = distances[np.arange(query.shape[0]), nearest]
        too_far = np.flatnonzero(nearest_distances > float(max_distance))
        if too_far.size:
            examples = [(int(row), float(nearest_distances[row])) for row in too_far[:5].tolist()]
            raise ChannelVocabularyError(
                "Positions do not exactly match EEGX2 identities within "
                f"max_distance={max_distance}: {examples}"
            )
        return nearest

    def resolve_positions(
        self,
        positions: object,
        *,
        max_distance: float = 1.0e-6,
    ) -> np.ndarray:
        """Resolve exact physical positions to active learned embedding IDs."""
        nearest = self._nearest_position_rows(
            positions,
            max_distance=max_distance,
        )
        return np.asarray(
            [self.channels[int(row)].index for row in nearest],
            dtype=np.int64,
        )

    def canonical_names_from_positions(
        self,
        positions: object,
        *,
        max_distance: float = 1.0e-6,
    ) -> tuple[str, ...]:
        """Resolve exact physical positions to canonical identity names."""
        nearest = self._nearest_position_rows(
            positions,
            max_distance=max_distance,
        )
        return tuple(self.channels[int(row)].name for row in nearest)


def _require_mapping(raw_value: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(raw_value, Mapping):
        raise ChannelVocabularyError(f"{context} must be an object")
    return raw_value


def _require_name(raw_name: object, *, context: str) -> str:
    if not isinstance(raw_name, str) or not normalize_channel_name(raw_name):
        raise ChannelVocabularyError(f"{context} name must be a non-empty string")
    return raw_name.strip()


@lru_cache(maxsize=8)
def _load_channel_vocabulary_cached(resolved_path: str) -> ChannelVocabulary:
    path = Path(resolved_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = _require_mapping(json.load(handle), context="Vocabulary")

    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ChannelVocabularyError(
            f"Unsupported channel-vocabulary schema_version {schema_version!r}"
        )
    capacity = payload.get("embedding_capacity")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ChannelVocabularyError("embedding_capacity must be a positive integer")

    raw_reserved = payload.get("reserved_indices", [])
    if not isinstance(raw_reserved, list):
        raise ChannelVocabularyError("reserved_indices must be an array")
    reserved = tuple(_index(value, context="Reserved", capacity=capacity) for value in raw_reserved)
    if len(set(reserved)) != len(reserved):
        raise ChannelVocabularyError("reserved_indices must be unique")

    raw_channels = payload.get("channels")
    if not isinstance(raw_channels, list) or not raw_channels:
        raise ChannelVocabularyError("channels must be a non-empty array")

    channels: list[ChannelRecord] = []
    active_indices: set[int] = set()
    accepted_names: set[str] = set()
    active_positions: set[tuple[float, float, float]] = set()
    alias_reserved_indices: set[int] = set()
    for channel_number, raw_channel in enumerate(raw_channels):
        context = f"channels[{channel_number}]"
        channel = _require_mapping(raw_channel, context=context)
        name = _require_name(channel.get("name"), context=context)
        normalized_name = normalize_channel_name(name)
        if normalized_name in accepted_names:
            raise ChannelVocabularyError(f"Duplicate accepted channel name {name!r}")
        accepted_names.add(normalized_name)

        index = _index(channel.get("index"), context=context, capacity=capacity)
        if index in active_indices:
            raise ChannelVocabularyError(f"Duplicate active channel index {index}")
        if index in reserved:
            raise ChannelVocabularyError(f"Active channel index {index} is reserved")
        active_indices.add(index)

        position = _position(channel.get("position"), context=context)
        if position in active_positions:
            raise ChannelVocabularyError(
                f"Duplicate active channel position for {name!r}; declare an alias"
            )
        active_positions.add(position)

        raw_aliases = channel.get("aliases", [])
        if not isinstance(raw_aliases, list):
            raise ChannelVocabularyError(f"{context} aliases must be an array")
        aliases: list[ChannelAlias] = []
        for alias_number, raw_alias in enumerate(raw_aliases):
            alias_context = f"{context}.aliases[{alias_number}]"
            alias = _require_mapping(raw_alias, context=alias_context)
            alias_name = _require_name(alias.get("name"), context=alias_context)
            normalized_alias = normalize_channel_name(alias_name)
            if normalized_alias in accepted_names:
                raise ChannelVocabularyError(f"Duplicate accepted channel name {alias_name!r}")
            accepted_names.add(normalized_alias)
            reserved_index = _index(
                alias.get("reserved_index"),
                context=alias_context,
                capacity=capacity,
            )
            if reserved_index not in reserved:
                raise ChannelVocabularyError(
                    f"Alias {alias_name!r} index {reserved_index} is not reserved"
                )
            if reserved_index in alias_reserved_indices:
                raise ChannelVocabularyError(
                    f"Reserved alias index {reserved_index} is used more than once"
                )
            alias_reserved_indices.add(reserved_index)
            alias_position = _position(alias.get("position"), context=alias_context)
            if alias_position != position:
                raise ChannelVocabularyError(
                    f"Alias {alias_name!r} position must equal canonical {name!r}"
                )
            aliases.append(
                ChannelAlias(
                    name=alias_name,
                    reserved_index=reserved_index,
                    position=alias_position,
                )
            )

        channels.append(
            ChannelRecord(
                name=name,
                index=index,
                position=position,
                aliases=tuple(aliases),
            )
        )

    if alias_reserved_indices != set(reserved):
        raise ChannelVocabularyError(
            "Every reserved index must belong to exactly one declared alias"
        )

    return ChannelVocabulary(
        schema_version=1,
        embedding_capacity=capacity,
        channels=tuple(channels),
        reserved_indices=reserved,
        fingerprint=sha256_file(path),
        source_path=path,
    )


def load_channel_vocabulary(
    path: str | Path = EEGX2_VOCABULARY_PATH,
) -> ChannelVocabulary:
    """Load and validate a versioned EEGX2 vocabulary artifact."""
    return _load_channel_vocabulary_cached(str(Path(path).expanduser().resolve()))


def channel_vocabulary_contract(
    vocabulary: ChannelVocabulary | None = None,
) -> dict[str, object]:
    """Return the exact JSON-safe identity contract for an EEGX2 vocabulary."""
    resolved = load_channel_vocabulary() if vocabulary is None else vocabulary
    return {
        "schema_version": resolved.schema_version,
        "sha256": resolved.fingerprint,
        "embedding_capacity": resolved.embedding_capacity,
        "active_identity_count": resolved.active_identity_count,
        "accepted_name_count": resolved.accepted_name_count,
        "reserved_indices": list(resolved.reserved_indices),
    }


def channel_identity_contract(
    channel_indexing: str | ChannelIndexing,
) -> dict[str, object]:
    """Build the checkpoint-level contract for one explicit indexing policy."""
    policy = normalize_channel_indexing(channel_indexing)
    contract: dict[str, object] = {
        "schema_version": 1,
        "indexing": policy.value,
        "embedding_capacity": channel_embedding_capacity(policy),
    }
    if policy is ChannelIndexing.EEGX2:
        contract["vocabulary"] = channel_vocabulary_contract()
    return contract
