"""The local V3 runtime shared by PDF reconstruction and table parsing."""
from pathlib import Path
import sys

def shared_structure():
    directory = str(Path(__file__).resolve().parents[2]/'transfer_kenel')
    if directory not in sys.path:
        sys.path.insert(0,directory)
    from pdf_extractor import structure
    return structure

def status():
    info = shared_structure().service_status()
    return dict(available=info['available'], backend='v3', local=True,
                loaded=info.get('loaded',False), service=info)
