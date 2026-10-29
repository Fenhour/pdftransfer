"""Shared V3 entry: local Python/model files by default, optional legacy HTTP."""
from __future__ import annotations

from .usage import tracked, span, trace_headers

import io
import base64
import json
import os
import socket
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, ProxyHandler, build_opener


def transport():
    # An explicitly supplied old URL remains an opt-in compatibility route.
    value = os.environ.get('PDFTRANSFER_V3_TRANSPORT') or ('http' if os.environ.get('PDFTRANSFER_STRUCTURE_URL') else 'native')
    if value not in {'native', 'http'}:
        raise ValueError('PDFTRANSFER_V3_TRANSPORT 仅支持 native 或 http')
    return value


def endpoint():
    value = os.environ.get('PDFTRANSFER_STRUCTURE_URL', 'http://127.0.0.1:8790').rstrip('/')
    parts = urlparse(value)
    if (parts.scheme != 'http' or parts.hostname not in {'127.0.0.1', 'localhost', '::1'}
            or parts.username or parts.password or parts.path or parts.query or parts.fragment):
        raise ValueError('V3 服务必须使用本机 HTTP 地址，例如 http://127.0.0.1:8790')
    return value


def service_status(timeout=2):
    if transport() == 'native':
        from .native_client import client
        return client.status()
    try:
        request = Request(endpoint() + '/health')
        with build_opener(ProxyHandler({})).open(request, timeout=timeout) as response:
            data = json.load(response)
        if data.get('service') != 'pdftransfer-structure-v3' or data.get('status') != 'ok':
            raise ValueError('本机端口不是 pdftransfer V3 服务')
        return dict(available=True, **data)
    except (OSError, ValueError, URLError) as exc:
        return dict(available=False, error=str(exc))


@tracked('v3.recognize_page')
def recognize_page(rgb):
    """Return the raw JSON result, in the supplied RGB image's pixel coordinates."""
    from PIL import Image
    buffer = io.BytesIO()
    image = rgb if isinstance(rgb, Image.Image) else Image.fromarray(rgb)
    with span('v3.png_encode'):
        image.convert('RGB').save(buffer, format='PNG')
    if transport() == 'native':
        from .native_client import client
        with span('v3.native_roundtrip'):
            headers = trace_headers()
            trace = dict(traceId=headers.get('X-Pdftransfer-Trace'), parentId=headers.get('X-Pdftransfer-Parent'),
                         feature=headers.get('X-Pdftransfer-Feature'))
            body = client.request(dict(op='recognize', pngBase64=base64.b64encode(buffer.getvalue()).decode('ascii'), trace=trace))
        result = body.get('result')
        return _validated(result, body, image.size)
    return _http_recognize(buffer.getvalue(), image.size)


@tracked('v3.recognize_rebuild')
def recognize_rebuild(rgb):
    """Feature-four typed crops; never changes the shared reading/editor path."""
    from PIL import Image
    if transport() != 'native':
        raise ValueError('功能四的分类裁片重建需要本机原生 V3；旧 HTTP 服务不支持此操作')
    image = rgb if isinstance(rgb, Image.Image) else Image.fromarray(rgb)
    buffer = io.BytesIO()
    with span('v3.png_encode'):
        image.convert('RGB').save(buffer, format='PNG')
    from .native_client import client
    headers = trace_headers()
    trace = dict(traceId=headers.get('X-Pdftransfer-Trace'), parentId=headers.get('X-Pdftransfer-Parent'),
                 feature=headers.get('X-Pdftransfer-Feature'))
    with span('v3.native_roundtrip'):
        body = client.request(dict(op='recognize_rebuild',
                    pngBase64=base64.b64encode(buffer.getvalue()).decode('ascii'),trace=trace))
    result = _validated(body.get('result'),body,image.size)
    if not result.get('rebuild_policy') or not isinstance(result.get('region_det_res'),dict):
        raise ValueError('本机 V3 未返回功能四要求的类别与大区域证据')
    return result


@tracked('v3.http_roundtrip')
def _http_recognize(data, size):
    request = Request(endpoint() + '/recognize', data=data,
                      headers={'Content-Type': 'image/png', **trace_headers()}, method='POST')
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=1800) as response:
            body = json.load(response)
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get('error', str(exc))
        except (ValueError, OSError):
            detail = str(exc)
        raise ValueError(f'本地 V3 识别失败：{detail}；没有改用旧识别结果。') from exc
    except (URLError, OSError, socket.timeout) as exc:
        raise ValueError('本地 V3 服务不可用或超时，请运行 services/structure_v3/start.ps1；'
                         '原生提取不受影响，扫描内容未偷偷回退到旧算法。') from exc
    return _validated(body.get('result'), body, size)


def _validated(result, body, size):
    if not isinstance(result, dict) or not isinstance(result.get('parsing_res_list'), list):
        raise ValueError('本地 V3 返回了不兼容的页面结果')
    if (result.get('width'), result.get('height')) != size:
        raise ValueError('本地 V3 页面尺寸与输入不一致，拒绝猜测坐标')
    result['_service'] = {key: body.get(key) for key in ('engine', 'seconds', 'cacheHit')}
    result['_service']['transport'] = transport()
    if 'modelProfile' in body:
        result['_service']['modelProfile'] = body['modelProfile']
    return result
