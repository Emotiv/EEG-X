"""Self-contained HDF5 dataset creation and lazy reading."""

from .reader import (
    DatasetReference as DatasetReference,
    Windows as Windows,
    read_metadata as read_metadata,
)
from .schema import (
    INDEX_DTYPE as INDEX_DTYPE,
    SPLITS as SPLITS,
    TASK_WINDOWS as TASK_WINDOWS,
    subject_split as subject_split,
)
from .writer import PreparedWriter as PreparedWriter
