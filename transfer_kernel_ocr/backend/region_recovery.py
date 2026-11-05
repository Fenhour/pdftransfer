"""Persist geometry failures while preserving the current editable state."""
from pathlib import Path

from .layout_pending import diagnostic_root, save_pending
from .regional_layout import RegionLayoutError


def layout_with_region_recovery(project, output_dir):
    from .adaptive_export import _validate
    from .adaptive_layout import layout_document

    root = Path(output_dir)
    token = diagnostic_root.set(root)
    try:
        project['layout_result'] = layout_document(project)
        _validate(project, root=root)
        return project['layout_result']
    except RegionLayoutError as error:
        raise save_pending(project, root, error) from error
    finally:
        diagnostic_root.reset(token)
