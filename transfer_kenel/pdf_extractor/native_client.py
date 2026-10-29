"""A resident local Python child owns Paddle; no HTTP listener or Docker needed."""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
SERVICE = ROOT / 'services' / 'structure_v3'
if str(SERVICE) not in sys.path:
    # This client also supports direct imports, before the logging bridge has
    # loaded the service's pure configuration module.
    sys.path.insert(0, str(SERVICE))


def python_executable():
    value = os.environ.get('PDFTRANSFER_V3_PYTHON')
    return Path(value).expanduser().resolve() if value else SERVICE / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def _read_lines(stream, replies):
    try:
        for line in stream:
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    replies.put(value)
            except ValueError:
                continue
    finally:
        replies.put(None)


class NativeClient:
    def __init__(self):
        self.lock = threading.RLock()
        self.process = None
        self.replies = None
        self.loaded = False
        self.check = (None, 0., None)

    def status(self):
        executable = python_executable()
        if not executable.is_file():
            return dict(available=False, transport='native', loaded=False,
                        error='本机 V3 环境未安装，请运行 services/structure_v3/setup-native.ps1')
        # Metadata probing does not import Paddle or load any model. Cache it so
        # frontend health polling cannot repeatedly launch Python processes.
        key = (str(executable), str(os.environ.get('PDFTRANSFER_V3_MODEL_DIR', '')),
               str(os.environ.get('PDFTRANSFER_V3_CONFIG', '')))
        previous, deadline, cached = self.check
        if previous != key or time.monotonic() >= deadline:
            code = ('import importlib.util,json; '
                    'print(json.dumps({m:bool(importlib.util.find_spec(m)) '
                    'for m in ("paddle","paddleocr","paddlex")}))')
            try:
                result = subprocess.run([str(executable), '-X', 'utf8', '-c', code],
                                        capture_output=True, text=True, encoding='utf-8', timeout=10,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                packages = json.loads(result.stdout)
                available = result.returncode == 0 and all(packages.values())
                cached = dict(available=available, transport='native', packages=packages)
                if not available:
                    cached['error'] = '本机 V3 依赖不完整，请运行 services/structure_v3/setup-native.ps1'
            except (OSError, ValueError, subprocess.TimeoutExpired):
                cached = dict(available=False, transport='native', error='本机 V3 Python 环境无法运行')
            self.check = (key, time.monotonic() + 30, cached)
        result = dict(cached, loaded=bool(self.process and self.process.poll() is None and self.loaded))
        try:
            # Same pure configuration loader as the child; no Paddle import.
            from native_worker import load_config
            from runtime_threads import cpu_threads
            config = load_config()
            ready = Path(config['model_dir']).is_dir()
            result.update(modelDirReady=ready, cpuThreads=cpu_threads(config), device=config['device'])
            if not ready:
                result.update(available=False, error='本机 V3 模型目录不存在，请配置 native_config.json 的 model_dir')
        except (OSError, ValueError):
            result.update(available=False, modelDirReady=False, error='本机 V3 配置无效，请检查 native_config.json')
        return result

    def _start(self):
        if self.process is not None and self.process.poll() is None:
            return
        executable = python_executable()
        if not executable.is_file():
            raise ValueError('本机 V3 环境未安装，请运行 services/structure_v3/setup-native.ps1；无需 Docker')
        environment = os.environ.copy()
        environment.update(PYTHONUTF8='1', PDFTRANSFER_LOG_ROLE='v3-native',
                           PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True')
        from native_worker import load_config
        from runtime_threads import configure_cpu_environment
        configure_cpu_environment(load_config(), environment)
        self.replies = queue.Queue()
        self.process = subprocess.Popen([str(executable), '-X', 'utf8', '-u', str(SERVICE / 'native_worker.py')],
                                        cwd=str(SERVICE), env=environment,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                        text=True, encoding='utf-8', bufsize=1,
                                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        self.loaded = False
        threading.Thread(target=_read_lines, args=(self.process.stdout, self.replies), daemon=True).start()

    def request(self, message, timeout=1800):
        with self.lock:
            self._start()
            identifier = uuid.uuid4().hex
            try:
                self.process.stdin.write(json.dumps(dict(message, id=identifier), ensure_ascii=False) + '\n')
                self.process.stdin.flush()
                reply = self.replies.get(timeout=timeout)
                if reply is None:
                    raise ValueError('本机 V3 进程退出；请检查原生依赖与模型目录，不会自动改用 Docker 或旧 OCR')
                if reply.get('id') != identifier:
                    raise ValueError('本机 V3 返回了不匹配的请求编号')
                if not reply.get('ok'):
                    raise ValueError('本机 V3 识别失败：' + str(reply.get('message', reply.get('errorType', 'unknown')))[:1000])
                if message.get('op') in {'recognize', 'recognize_rebuild'} and not reply.get('cacheHit'):
                    self.loaded = True
                return reply
            except queue.Empty as exc:
                self.close()
                raise ValueError('本机 V3 推理超时；已终止本次模型进程') from exc
            except (BrokenPipeError, OSError) as exc:
                self.close()
                raise ValueError('无法与本机 V3 模型进程通信；不会自动改用其他识别后端') from exc
            except BaseException:
                # Keep protocol synchronization after errors/cancellation. The
                # next request may create a clean child; never reuse stale data.
                self.close()
                raise

    def close(self):
        with self.lock:
            process, self.process = self.process, None
            self.loaded = False
            if process is None:
                return
            if process.poll() is None:
                try:
                    process.stdin.write(json.dumps(dict(id='shutdown', op='shutdown')) + '\n')
                    process.stdin.flush()
                    process.wait(timeout=3)
                except (OSError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
            for stream in (process.stdin, process.stdout):
                if stream:
                    try:
                        stream.close()
                    except (OSError, ValueError):
                        pass


client = NativeClient()
atexit.register(client.close)
