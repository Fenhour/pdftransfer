from types import SimpleNamespace

from backend.font_size_audit import font_size_audit


def test_audit_reports_reduced_line_with_source_and_final_page_coordinates(monkeypatch):
    font = SimpleNamespace(ascender=.8, text_length=lambda text, fontsize: len(text)*fontsize*.5)
    monkeypatch.setattr('backend.font_size_audit.get_library',
                        lambda: SimpleNamespace(export_font_for_key=lambda key: font))
    block = dict(id='b', style=dict(font_size_pt=10), content={},
                 source=dict(page_id='p', ocr_lines=[dict(line_id='l', bbox_px=[20,40,120,60])]))
    measurement = dict(font_key='f', font_size=8, uniform_scale=.9,
                       paragraph_adjustments=[dict(paragraph_id='para', font_scale=8/9)],
                       lines=[dict(text='word', line_id='l', paragraph_id='para', x=2,
                                   baseline=8.4, height=8, advance_width=20)])
    project = dict(blocks=[block], source_pages=[dict(id='p', physical_page_index=4, pixel_to_point=[.5,.25])],
                   layout_result=dict(placements=[dict(block_id='b', measurement=measurement,
                                                       bbox=[100,200,200,220], page_index=1)]))
    result = font_size_audit(project)
    row, = result['lines']
    assert result['shrunk_line_count'] == 1
    assert row['source_page_number'] == 5
    assert row['page_number'] == 2
    assert row['original_bbox_pt'] == [10,10,60,15]
    assert row['output_bbox_pt'] == [102,202,122,210]
    assert row['initial_font_size_pt'] == 10
    assert row['final_font_size_pt'] == 8
    assert row['reasons'] == ['paragraph-font-scaling','block-or-region-uniform-scaling']
    block['style']['font_size_pt'] = 8
    assert font_size_audit(project)['shrunk_line_count'] == 0
