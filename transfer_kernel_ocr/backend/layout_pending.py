"""Persist unfinished reconstruction without rasterising editable objects."""
from copy import deepcopy
import json
from pathlib import Path
import hashlib
from contextvars import ContextVar

diagnostic_root=ContextVar('layout_diagnostic_root',default=None)


def save_layout_issue(project, root, error):
    if root is None:
        return None
    token=hashlib.sha256((error.page['id']+':'+error.frame['id']).encode()).hexdigest()[:16]
    path=Path(root)/'layout-diagnostics'/('issue-'+token+'.json')
    path.parent.mkdir(parents=True,exist_ok=True)
    state=dict(schema='feature4-layout-diagnostic/1',status='export-current-state-and-continue',
        source_page_id=error.page['id'],physical_page_number=error.page.get('physical_page_index',0)+1,
        reason=str(error),frame=deepcopy(error.frame),measurements=deepcopy(getattr(error,'measurements',{})),
        positions=deepcopy(getattr(error,'positions',{})),solver=deepcopy(getattr(error,'solver_diagnostics',{})),
        blocks=[deepcopy(b) for b in project['blocks'] if b['id'] in error.frame['block_ids']])
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf8')
    temporary.replace(path)
    return path


class LayoutPendingError(ValueError):
    def __init__(self, message, checkpoint):
        super().__init__(message)
        self.checkpoint = str(checkpoint)


def save_pending(project, root, error):
    root = Path(root)
    path = root/'layout-pending.json'
    root.mkdir(parents=True,exist_ok=True)
    state = dict(schema='feature4-layout-pending/1',status='waiting-for-layout-resolution',
        project=deepcopy(project),reason=str(error),
        frame=deepcopy(getattr(error,'frame',None)),
        measurements=deepcopy(getattr(error,'measurements',{})),
        positions=deepcopy(getattr(error,'positions',{})),
        solver=deepcopy(getattr(error,'solver_diagnostics',{})))
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf8')
    temporary.replace(path)
    return LayoutPendingError('布局待解决，已保留文字/表格与测量现场，未替换区域原图：'+str(error),path)
