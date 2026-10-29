"""Import legacy saved jobs as new scenes, without altering their source files."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="old parser root or backup directory")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    from backend import app
    state_path = ROOT / ".migration-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    report = []
    for path in sorted((source / "output").glob("*/summary/project.json")):
        raw = path.read_bytes()
        key = str(path.resolve()) + ":" + hashlib.sha256(raw).hexdigest()
        previous = state.get(key)
        if previous and (app.OUTPUT / previous / "summary/document.json").is_file():
            report.append({"source": path.parent.parent.name, "id": previous, "status": "already-imported"})
            continue
        document_id = uuid.uuid4().hex
        try:
            data = json.loads(raw.decode("utf-8-sig"))
            document = app.import_scene(data, document_id, app.OUTPUT / document_id, data.get("name") or path.parent.parent.name)
            document = app.persist(document)
            state[key] = document_id
            app.atomic_write(state_path, json.dumps(state, ensure_ascii=False, indent=2))
            report.append({"source": path.parent.parent.name, "id": document_id, "status": "imported",
                           "pages": len(document["pages"]), "elements": sum(len(p["elements"]) for p in document["pages"])})
        except Exception as exc:
            report.append({"source": path.parent.parent.name, "status": "error", "error": str(exc)})
        print(json.dumps(report[-1], ensure_ascii=False), flush=True)
    destination = app.OUTPUT / "migration-report.json"
    app.atomic_write(destination, json.dumps({"source": str(source), "jobs": report}, ensure_ascii=False, indent=2))
    if any(item["status"] == "error" for item in report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
