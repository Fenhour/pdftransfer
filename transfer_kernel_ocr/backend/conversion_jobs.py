"""Single-worker FIFO reconstruction queue, with durable page-ready evidence."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import logging
from pathlib import Path
import threading
import time


class ConversionJobs:
    def __init__(self, root, runner):
        self.root = Path(root)
        self.runner = runner
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='rebuild-queue')
        self.items = {}
        for path in self.root.glob('*/queue-job.json'):
            try:
                item = json.loads(path.read_text(encoding='utf-8'))
                if item['status'] in {'queued','running'}:
                    item.update(status='interrupted', error='服务已重启；已完成页面保留，请重新提交未完成文档。')
                    self._write(item)
                self.items[item['id']] = item
            except (OSError,ValueError,KeyError):
                logging.exception('Cannot recover conversion job %s',path)

    def _write(self, item):
        path = self.root/item['id']/'queue-job.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(item,ensure_ascii=False),encoding='utf-8')
        temp.replace(path)

    def submit(self, identifier, names, options, paths):
        with self.lock:
            if sum(i['status'] in {'queued','running'} for i in self.items.values()) >= 30:
                raise ValueError('等待中的任务已达30个，请稍后提交')
            item = dict(id=identifier, names=list(names), options=deepcopy(options),
                        status='queued', created=time.time(), pages=[], error=None)
            self._write(item)
            self.items[identifier] = item
            self.executor.submit(self._run,identifier,list(paths))
            return deepcopy(item)

    def _run(self, identifier, paths):
        from .progress import store, bind_progress
        with self.lock:
            item = self.items[identifier]
            item.update(status='running',started=time.time())
            self._write(item)
        token = store.create()
        store.begin(token)
        with self.lock:
            item['progress_id'] = token
        def ready(page):
            with self.lock:
                item['pages'].append(deepcopy(page))
                self._write(item)
        try:
            with bind_progress(token):
                destination, report = self.runner(paths,self.root/identifier,identifier,
                                                  on_page=ready,**item['options'])
            destination = Path(destination).resolve()
            relative = destination.relative_to((self.root/identifier).resolve()).as_posix()
            with self.lock:
                item.update(status='completed',result=relative,report=report,finished=time.time())
        except Exception as exc:
            from .layout_pending import LayoutPendingError
            logging.exception('Queued reconstruction failed: %s',identifier)
            with self.lock:
                if isinstance(exc,LayoutPendingError):
                    item.update(status='waiting-for-layout-resolution',error=str(exc)[:1500],
                                checkpoint=Path(exc.checkpoint).name)
                else:
                    item.update(status='failed',error=str(exc)[:1500],finished=time.time())
        finally:
            store.finish(token,item['status']=='completed')
            with self.lock:
                self._write(item)

    def list(self):
        from .progress import store
        with self.lock:
            result = deepcopy(sorted(self.items.values(),key=lambda i:i['created']))
        position = 0
        for item in result:
            if item['status']=='queued':
                position += 1
                item['queue_position'] = position
            item['progress'] = store.get(item.get('progress_id'))
        return result

    def close(self):
        self.executor.shutdown(wait=True)
