"""Share original detection geometry with the native recognition worker."""
from pathlib import Path
import sys

_shared = str(Path(__file__).resolve().parents[2] / 'services' / 'structure_v3')
if _shared not in sys.path:
    sys.path.insert(0, _shared)

from source_containment import (VERSION, build_containment, block_metadata,
                                conflict_is_internal, source_contains_pair,
                                strictly_contains)
