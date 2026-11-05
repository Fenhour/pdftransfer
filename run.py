"""Launch the local conversion service from any working directory."""
import argparse
import json
from pathlib import Path
import sys
import threading
import time
from urllib.request import urlopen
import webbrowser

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'transfer_kernel_ocr'))

def open_browser(url):
    for _ in range(60):
        try:
            with urlopen(url+'/api/health',timeout=1) as response:
                if json.load(response).get('status')=='ok':
                    webbrowser.open(url)
                    return
        except (OSError,ValueError):
            pass
        time.sleep(.5)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--no-browser',action='store_true')
    arguments=parser.parse_args()
    if not 1024<=arguments.port<=65535: parser.error('Port must be between 1024 and 65535')
    if not arguments.no_browser:
        threading.Thread(target=open_browser,args=(f'http://127.0.0.1:{arguments.port}',),daemon=True).start()
    import uvicorn
    uvicorn.run('backend.app:app',host='127.0.0.1',port=arguments.port)

if __name__=='__main__':
    main()
