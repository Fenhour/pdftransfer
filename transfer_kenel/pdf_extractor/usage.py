"""Load the shared standard-library logger used by the local V3 service."""
from pathlib import Path
import sys
_shared = str(Path(__file__).resolve().parents[2] / 'services' / 'structure_v3')
if _shared not in sys.path:
    sys.path.insert(0, _shared)
from pdftransfer_usage import span, tracked, trace_headers
