import json

import pymupdf as fitz
import pytest

from test_app import local_app


def rows(directory):
    return [json.loads(line) for path in directory.glob('*.jsonl*')
            for line in path.read_text(encoding='utf-8').splitlines()]


def pdf_bytes():
    with fitz.open() as pdf:
        page = pdf.new_page(width=300, height=200)
        page.insert_text((20, 40), 'Synthetic usage logging evidence', fontsize=12)
        return pdf.tobytes()


@pytest.mark.parametrize('format', ['html', 'md'])
def test_reading_feature_modules_share_request_trace(local_app, monkeypatch, tmp_path, format):
    directory = tmp_path / 'logs'
    monkeypatch.setenv('PDFTRANSFER_LOG_DIR', str(directory))
    client, _, _ = local_app
    response = client.post('/api/convert/reading', data={'format': format, 'ocr': 'false'},
                           files=[('files', ('private-name.pdf', pdf_bytes(), 'application/pdf'))])
    assert response.status_code == 200, response.text
    records = rows(directory)
    assert {r['traceId'] for r in records} == {response.headers['X-Pdftransfer-Trace']}
    assert {'reading.convert', 'reading.extract', 'reading.export', 'reading.package', 'http.request'} <= {r['module'] for r in records}
    assert next(r for r in records if r['module'] == 'reading.convert')['feature'] == 'reading.' + format
    assert not any(value in json.dumps(records) for value in ['private-name', 'Synthetic usage'])
    before = len(records)
    client.get('/api/health')
    assert len(rows(directory)) == before


def test_editor_and_rebuild_features_and_failure_status(local_app, monkeypatch, tmp_path):
    directory = tmp_path / 'logs'
    monkeypatch.setenv('PDFTRANSFER_LOG_DIR', str(directory))
    monkeypatch.setenv('PDFTRANSFER_RECOGNITION', 'v3')
    client, _, _ = local_app
    files = [('files', ('native.pdf', pdf_bytes(), 'application/pdf'))]
    uploaded = client.post('/api/documents', files=files, data={'ocr': 'false'})
    assert uploaded.status_code == 200, uploaded.text
    scene = uploaded.json()
    saved = client.put('/api/documents/' + scene['id'], json=scene)
    assert saved.status_code == 200
    exported = client.post('/api/documents/' + scene['id'] + '/export/html', json=scene)
    assert exported.status_code == 200
    rebuilt = client.post('/api/convert/rebuild', files=files, data={'format': 'html'})
    assert rebuilt.status_code == 200, rebuilt.text
    failed = client.post('/api/convert/reading', files=files, data={'format': 'invalid'})
    assert failed.status_code == 422
    records = rows(directory)
    assert {'editor.import', 'editor.save', 'editor.export', 'rebuild.html'} <= {r['feature'] for r in records}
    error = next(r for r in records if r['spanId'] and r['module'] == 'http.request' and
                 r['traceId'] == failed.headers['X-Pdftransfer-Trace'])
    assert error['status'] == 'error'
    assert error['metadata']['httpStatus'] == 422
    assert next(r for r in records if r['module'] == 'document.v3_parse')['processCpuSeconds'] >= 0

