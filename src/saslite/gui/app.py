"""SASLite Web GUI — Flask backend."""

from __future__ import annotations

import io
import math
import re
import secrets
import sys
import threading
import traceback
import webbrowser
from argparse import ArgumentParser
from datetime import date, datetime
from html import escape
from pathlib import Path

import pandas as pd
from flask import Flask, Response, jsonify, request

from saslite import SasInterpreter

app = Flask(__name__, static_folder="static")
app.config.setdefault("SAS_FACTORY", SasInterpreter)
app.config["SAS_SESSION"] = app.config["SAS_FACTORY"]()
app.config.setdefault("SAS_LOCK", threading.RLock())
app.config.setdefault("SAS_API_TOKEN", secrets.token_urlsafe(32))
app.config.setdefault("MAX_CONTENT_LENGTH", 100 * 1024 * 1024)

_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}
_DATASET_NAME_RE = re.compile(r"[^A-Z0-9_]+")


def get_sas() -> SasInterpreter:
    """Return the persistent interpreter for the current GUI app instance."""
    sas = app.config.get("SAS_SESSION")
    if sas is None:
        sas = app.config["SAS_FACTORY"]()
        app.config["SAS_SESSION"] = sas
    return sas


def reset_sas() -> SasInterpreter:
    """Reset the GUI interpreter; useful for tests and explicit state isolation."""
    sas = app.config["SAS_FACTORY"]()
    app.config["SAS_SESSION"] = sas
    return sas


def _request_host_name() -> str:
    host = request.host.lower()
    if host.startswith("[") and "]" in host:
        return host[1:host.index("]")]
    return host.rsplit(":", 1)[0]


def _authorized_api_request() -> bool:
    expected = app.config["SAS_API_TOKEN"]
    provided = request.headers.get("X-SASLite-Token", "")
    return secrets.compare_digest(provided, expected)


@app.before_request
def _protect_local_api():
    """Reject rebinding hosts and require a token for state-changing API calls."""
    if _request_host_name() not in _ALLOWED_HOSTS:
        return jsonify({"success": False, "error": "Forbidden host"}), 403

    if (
        request.path.startswith("/api/")
        and request.method not in {"GET", "HEAD", "OPTIONS"}
        and not _authorized_api_request()
    ):
        return jsonify({"success": False, "error": "Unauthorized"}), 403


@app.after_request
def _security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    if request.path == "/":
        response.headers["Cache-Control"] = "no-store"
    return response


def _json_body() -> dict[str, object] | None:
    body = request.get_json(silent=True)
    return body if isinstance(body, dict) else None


def _sas_lock():
    return app.config["SAS_LOCK"]


def _safe_dataset_name(filename: str) -> str:
    name = Path(filename).stem.upper()
    name = _DATASET_NAME_RE.sub("_", name).strip("_")
    if not name:
        name = "DATA"
    if not (name[0].isalpha() or name[0] == "_"):
        name = f"_{name}"
    return name[:32]


def _dataset_summary(libref: str, backend: object, name: str) -> dict[str, object] | None:
    try:
        ds = backend.read(name)
    except Exception:
        return None
    if not ds:
        return None
    return {
        "libref": libref,
        "name": name,
        "full_name": f"{libref}.{name}",
        "rows": ds.nrow,
        "columns": ds.ncol,
        "col_names": list(ds.data.columns),
    }


def _json_safe_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat(sep=" ")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _json_safe_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    rows = df.to_dict(orient="records")
    return [
        {key: _json_safe_value(value) for key, value in row.items()}
        for row in rows
    ]


@app.route("/")
def index():
    html_path = Path(app.static_folder or "static") / "index.html"
    html = html_path.read_text(encoding="utf-8")
    html = html.replace("__SASLITE_API_TOKEN__", escape(app.config["SAS_API_TOKEN"], quote=True))
    return Response(html, mimetype="text/html")


@app.route("/api/execute", methods=["POST"])
def api_execute():
    """Execute SAS code."""
    body = _json_body()
    if body is None:
        return jsonify({"success": False, "error": "Invalid JSON body"}), 400
    code = body.get("code", "")

    if not code.strip():
        return jsonify({"success": False, "error": "Empty code"})

    with _sas_lock():
        sas = get_sas()
        captured = io.StringIO()
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_reporter_stream = sas.reporter._stream

        try:
            sys.stdout = captured
            sys.stderr = captured
            sas.reporter._stream = captured

            result = sas.execute(code)
            output_text = captured.getvalue()

            steps = []
            for step in result.steps:
                steps.append({
                    "success": step.success,
                    "error": step.error,
                    "notes": step.notes,
                    "warnings": step.warnings,
                })

            return jsonify({
                "success": result.success,
                "error": result.error,
                "output": output_text,
                "steps": steps,
            })
        except Exception as e:
            traceback.print_exc()
            return jsonify({
                "success": False,
                "error": str(e),
            })
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            sas.reporter._stream = old_reporter_stream


@app.route("/api/datasets", methods=["GET"])
def api_datasets():
    """List all datasets."""
    with _sas_lock():
        sas = get_sas()
        datasets = []
        for libref, backend in sas.session.storage._backends.items():
            for name in backend.list_datasets():
                summary = _dataset_summary(libref, backend, name)
                if summary:
                    datasets.append(summary)
    return jsonify({"datasets": datasets})


@app.route("/api/libraries", methods=["GET"])
def api_libraries():
    """List all libraries with their datasets grouped by library."""
    with _sas_lock():
        sas = get_sas()
        libraries = []
        for libref, backend in sas.session.storage._backends.items():
            ds_list = []
            for name in backend.list_datasets():
                summary = _dataset_summary(libref, backend, name)
                if summary:
                    ds_list.append({
                        "name": summary["name"],
                        "rows": summary["rows"],
                        "columns": summary["columns"],
                        "col_names": summary["col_names"],
                    })

            # Determine library description
            engine = getattr(backend, "engine", "MEMORY")
            path = str(getattr(backend, "path", "") or "")
            if libref == "WORK":
                desc = "Work Library"
                icon = "work"
            elif path:
                desc = path
                icon = "disk"
            else:
                desc = "Memory Library"
                icon = "memory"

            libraries.append({
                "libref": libref,
                "description": desc,
                "engine": engine,
                "path": path,
                "icon": icon,
                "datasets": sorted(ds_list, key=lambda d: d["name"]),
                "count": len(ds_list),
            })

    # WORK first, then alphabetical
    libraries.sort(key=lambda lib: (0 if lib["libref"] == "WORK" else 1, lib["libref"]))
    return jsonify({"libraries": libraries})


@app.route("/api/datasets/<libref>/<name>", methods=["GET"])
def api_get_dataset(libref: str, name: str):
    """Get dataset content as JSON."""
    with _sas_lock():
        sas = get_sas()
        try:
            ds = sas.session.get_dataset(libref.upper(), name.upper())
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        except KeyError:
            return jsonify({"error": f"Dataset {libref}.{name} not found"}), 404

        df = ds.data.copy()

    columns = []
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        if "int" in dtype_str:
            col_type = "int"
        elif "float" in dtype_str:
            col_type = "float"
        elif "datetime" in dtype_str:
            col_type = "datetime"
        elif "bool" in dtype_str:
            col_type = "bool"
        else:
            col_type = "str"
        columns.append({"name": col, "type": col_type})

    return jsonify({
        "libref": libref.upper(),
        "name": name.upper(),
        "columns": columns,
        "rows": _json_safe_rows(df),
        "total_rows": len(df),
    })


@app.route("/api/datasets/<libref>/<name>", methods=["DELETE"])
def api_delete_dataset(libref: str, name: str):
    """Delete a dataset."""
    try:
        with _sas_lock():
            sas = get_sas()
            backend = sas.session.storage.get_backend(libref.upper())
            if backend and backend.exists(name.upper()):
                backend.delete(name.upper())
                return jsonify({"success": True})
        return jsonify({"error": "Dataset not found"}), 404
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/open-file", methods=["POST"])
def api_open_file():
    """Open a .sas file and return its content."""
    body = _json_body()
    if body is None:
        return jsonify({"success": False, "error": "Invalid JSON body"}), 400
    filepath = body.get("path", "")
    if not filepath:
        return jsonify({"success": False, "error": "No file path provided"})
    try:
        p = Path(filepath)
        if not p.exists():
            return jsonify({"success": False, "error": f"File not found: {filepath}"})
        if not p.suffix.lower() == ".sas":
            return jsonify({"success": False, "error": "Only .sas files are supported"})
        content = p.read_text(encoding="utf-8", errors="replace")
        return jsonify({"success": True, "content": content, "filename": p.name})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/import-data", methods=["POST"])
def api_import_data():
    """Import a data file (sas7bdat, xpt, csv, excel) into WORK library."""
    body = _json_body()
    if body is None:
        return jsonify({"success": False, "error": "Invalid JSON body"}), 400
    filepath = body.get("path", "")
    if not filepath:
        return jsonify({"success": False, "error": "No file path provided"})
    try:
        p = Path(filepath)
        if not p.exists():
            return jsonify({"success": False, "error": f"File not found: {filepath}"})

        suffix = p.suffix.lower()
        dataset_name = _safe_dataset_name(p.name)

        if suffix == ".sas7bdat":
            from saslite.storage.sas_backend import _read_sas7bdat
            df = _read_sas7bdat(p)
        elif suffix == ".xpt":
            import pyreadstat
            df, _ = pyreadstat.read_xport(str(p))
        elif suffix == ".csv":
            df = pd.read_csv(p)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(p)
        else:
            return jsonify({"success": False, "error": f"Unsupported format: {suffix}"})

        from saslite.runtime.dataset import Dataset
        ds = Dataset.from_dataframe(df, name=dataset_name, libref="WORK")
        with _sas_lock():
            sas = get_sas()
            sas.session.put_dataset("WORK", dataset_name, ds)

        return jsonify({
            "success": True,
            "dataset": f"WORK.{dataset_name}",
            "rows": ds.nrow,
            "columns": ds.ncol,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/upload-data", methods=["POST"])
def api_upload_data():
    """Upload a data file via browser and import into WORK library."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"})
    f = request.files["file"]
    if not f.filename:
        return jsonify({"success": False, "error": "Empty filename"})

    import tempfile
    suffix = Path(f.filename).suffix.lower()
    dataset_name = _safe_dataset_name(f.filename)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        f.save(tmp)
        tmp_path = tmp.name

    try:
        if suffix == ".sas7bdat":
            from saslite.storage.sas_backend import _read_sas7bdat
            df = _read_sas7bdat(Path(tmp_path))
        elif suffix == ".xpt":
            import pyreadstat
            df, _ = pyreadstat.read_xport(tmp_path)
        elif suffix == ".csv":
            df = pd.read_csv(tmp_path)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(tmp_path)
        else:
            return jsonify({"success": False, "error": f"Unsupported format: {suffix}"})

        from saslite.runtime.dataset import Dataset
        ds = Dataset.from_dataframe(df, name=dataset_name, libref="WORK")
        with _sas_lock():
            sas = get_sas()
            sas.session.put_dataset("WORK", dataset_name, ds)

        return jsonify({
            "success": True,
            "dataset": f"WORK.{dataset_name}",
            "rows": ds.nrow,
            "columns": ds.ncol,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    finally:
        tmp_file = Path(tmp_path)
        if tmp_file.exists():
            tmp_file.unlink()


@app.route("/api/file-dialog", methods=["POST"])
def api_file_dialog():
    """Open a native file dialog and return the selected path.

    Requires pywebview integration; falls back to None if not available.
    """
    body = _json_body()
    if body is None:
        return jsonify({"success": False, "error": "Invalid JSON body"}), 400
    mode = body.get("mode", "open")  # open or import
    try:
        import webview
        window = webview.windows[0] if webview.windows else None
        if not window:
            return jsonify({"success": False, "error": "No window available"})

        if mode == "open":
            result = window.create_file_dialog(
                webview.OPEN_DIALOG,
                directory="",
                allow_multiple=False,
                file_types=("SAS Files (*.sas)", "All Files (*.*)"),
            )
        else:
            result = window.create_file_dialog(
                webview.OPEN_DIALOG,
                directory="",
                allow_multiple=False,
                file_types=(
                    "Data Files (*.sas7bdat;*.xpt;*.csv;*.xlsx;*.xls)",
                    "SAS7BDAT (*.sas7bdat)",
                    "XPORT (*.xpt)",
                    "CSV (*.csv)",
                    "Excel (*.xlsx;*.xls)",
                    "All Files (*.*)",
                ),
            )

        if result and len(result) > 0:
            return jsonify({"success": True, "path": result[0]})
        return jsonify({"success": False, "error": "No file selected"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


def main(argv: list[str] | None = None) -> int:
    """Launch the browser-based GUI."""
    parser = ArgumentParser(
        prog="saslite-gui",
        description="Start the SASLite browser GUI.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind.")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the server without opening a browser tab.",
    )
    args = parser.parse_args(argv)

    url = f"http://{args.host}:{args.port}"
    print("SASLite Web GUI starting...")
    print(f"Open {url} in your browser")
    if not args.no_browser:
        webbrowser.open(url)
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
