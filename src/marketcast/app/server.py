"""Local dashboard server.   python -m marketcast app [--port 8765]     (or double-click RUN_DASHBOARD.bat)

Standard library only (http.server): no web framework to install or keep patched. It binds to 127.0.0.1, so the
page is reachable from this PC alone; that is deliberate (00 rule 11: output stays internal). Do not change
the bind address to expose it on a network without the legal review the plan asks for.

ROUTES
    GET  /                      the page (static/index.html)
    GET  /api/meta              date range, scenario sources, option lists
    GET  /api/companies?date=   universe members on that date (for the holdings picker)
    POST /api/report            {source, date, holdings: {key: weight}, horizon, n_paths, value_inr} -> risk view
    GET  /api/runs              training runs with their loss curves
    GET  /api/scores            mean scores per model from reports/scores
Errors come back as JSON {"error": "..."} with a 4xx/5xx status, and are shown in the page as a message.
"""
from __future__ import annotations

import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from marketcast.util import _json_default, get_logger

log = get_logger("app")
STATIC = Path(__file__).resolve().parent / "static"
MAX_BODY = 1_000_000


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        server_version = "MarketCastDashboard/0.1"

        def log_message(self, fmt, *args):                      # route access logs to our logger (DEBUG: file only)
            log.debug("%s %s", self.address_string(), fmt % args)

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj, default=_json_default).encode("utf-8"), "application/json; charset=utf-8")

        def _guard(self, fn) -> None:
            try:
                self._json(fn())
            except PermissionError as e:                        # vault / policy refusals
                self._json({"error": str(e)}, 403)
            except (ValueError, KeyError, FileNotFoundError) as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 400)
            except Exception as e:                              # never leak a stack trace to the page; log it
                log.exception("request failed")
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/meta":
                return self._guard(engine.meta)
            if u.path == "/api/companies":
                return self._guard(lambda: engine.companies(q.get("date", "")))
            if u.path == "/api/runs":
                return self._guard(lambda: {"runs": engine.runs()})
            if u.path == "/api/scores":
                return self._guard(engine.scores)
            self._json({"error": "not found"}, 404)

        def do_POST(self):
            if urlparse(self.path).path != "/api/report":
                return self._json({"error": "not found"}, 404)
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > MAX_BODY:
                return self._json({"error": "bad request size"}, 400)
            try:
                p = json.loads(self.rfile.read(n))
                holdings = {str(k): float(v) for k, v in (p.get("holdings") or {}).items() if float(v) > 0}
            except (ValueError, TypeError) as e:
                return self._json({"error": f"bad JSON: {e}"}, 400)
            self._guard(lambda: engine.report(str(p.get("source", "B1")), str(p.get("date", "")), holdings, int(p.get("horizon", 20)),
                                              int(p.get("n_paths", 256)), p.get("value_inr")))

    return Handler


def main(port: int = 8765, open_browser: bool = True) -> int:
    from marketcast.app.engine import Engine

    engine = Engine()
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(engine))
    url = f"http://127.0.0.1:{port}/"
    log.info("MarketCast dashboard on %s  (this PC only; Ctrl+C to stop)", url)
    if open_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        log.info("dashboard stopped")
    finally:
        srv.server_close()
    return 0
