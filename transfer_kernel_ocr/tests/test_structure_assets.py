"""Table reference images survive portable scenes and obey document ownership."""
import json

import pytest

from backend.exporter import portable_document
from test_app import local_app, image_document


@pytest.mark.parametrize('embedded', [False, True])
def test_import_localizes_table_reference_with_scene_assets(local_app, image_document, embedded):
    client, _, output = local_app
    scene = image_document
    original = scene['pages'][0]['sourceSrc']
    scene['pages'][0]['tables'] = [{'id': 'reference-table', 'referenceSrc': original, 'cells': []}]
    payload = portable_document(scene, output / scene['id']) if embedded else scene
    response = client.post('/api/documents', files=[
        ('files', ('reference.json', json.dumps(payload).encode(), 'application/json')),
    ])
    assert response.status_code == 200, response.text
    imported = response.json()
    reference = imported['pages'][0]['tables'][0]['referenceSrc']
    assert reference.startswith(f'/files/{imported["id"]}/images/')
    assert imported['id'] != scene['id']
    assert client.get(reference).status_code == 200
    assert scene['pages'][0]['tables'][0]['referenceSrc'] == original


@pytest.mark.parametrize('reference', ['https://example.com/table.png', '/files/other-document/table.png', 123])
def test_save_rejects_unowned_or_invalid_table_reference(local_app, image_document, reference):
    client, _, _ = local_app
    scene = image_document
    scene['pages'][0]['tables'] = [{'id': 'reference-table', 'referenceSrc': reference}]
    response = client.put(f'/api/documents/{scene["id"]}', json=scene)
    assert response.status_code == 422, response.text
