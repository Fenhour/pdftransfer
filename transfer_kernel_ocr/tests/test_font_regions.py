"""Word alignment must use real ink and preserve OCR text, including thin marks."""
import numpy as np
import pytest

from backend import font_library


@pytest.mark.parametrize('height', [8, 22, 25, 26, 30])
def test_word_regions_keep_thin_ink_at_its_original_coordinates(height):
    mask = np.zeros((height, 64), dtype=np.uint8)
    mask[:, 0:8] = 255
    mask[:, 28] = 255
    mask[:, 50:64] = 255
    before = mask.copy()
    text = 'alpha I omega'
    regions = font_library._word_regions(mask, text, 'unused-on-direct-word-alignment')
    assert ''.join(text[start:end] for start, end, _, _ in regions) == text
    covered = np.zeros_like(mask, dtype=bool)
    for start, end, left, right in regions:
        assert 0 <= start < end <= len(text)
        assert 0 <= left < right <= mask.shape[1]
        assert font_library._tight(mask[:, left:right]) is not None
        covered[:, left:right] = True
    assert np.all(covered[mask > 0]), 'Grouping must retain all original ink columns'
    np.testing.assert_array_equal(mask, before)


def test_word_regions_fall_back_to_whole_line_if_a_group_has_no_original_ink(monkeypatch):
    mask = np.zeros((25, 64), dtype=np.uint8)
    mask[:, 0:8] = 255
    mask[:, 50:64] = 255
    changed = mask.copy()
    changed[:, 28] = 255
    monkeypatch.setattr(font_library.cv2, 'morphologyEx', lambda *args, **kwargs: changed)
    text = 'alpha I omega'
    assert font_library._word_regions(mask, text, 'unused') == [(0, len(text), 0, 64)]
