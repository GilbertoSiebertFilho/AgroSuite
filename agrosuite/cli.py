"""AgroSuite entry point.

Starts the local server and opens the browser. By default it listens only on
``127.0.0.1``: no other machine on the network can reach the app, and no data
leaves the computer.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import threading
import webbrowser


def find_free_port(preferred: int, host: str = "127.0.0.1") -> int:
    """Return ``preferred`` if it is free, or the first free port above it.

    A previous session that did not shut down cleanly leaves the port taken;
    rather than failing with "address in use", the app simply starts on the
    next one.
    """
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(
        f"No free port between {preferred} and {preferred + 49}. "
        "Close other AgroSuite instances and try again."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agrosuite",
        description="Open AgroSuite in the browser.",
    )
    parser.add_argument("--port", type=int, default=8765, help="Preferred port (default: 8765).")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Listen address. Keep 127.0.0.1 for local use.")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open the browser automatically.")
    parser.add_argument("--reload", action="store_true",
                        help="Reload on code changes (development).")
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        print(
            "Dependencies are not installed. Run:\n"
            "    pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    # Imported here rather than at the top: this module must still be able
    # to print the "dependencies are not installed" message above, and the
    # app package pulls in pandas on the way.
    from .app import autosave

    port = find_free_port(args.port, args.host)
    url = f"http://{args.host}:{port}"

    # This process is the app, so it is the one that saves the session by
    # itself and picks the last project up again. The environment carries
    # that, rather than the app object, because with --reload the app is
    # imported in a child process — and because importing the app in a test
    # or a script must not start a thread writing to somebody's Documents
    # folder. Someone who has set it already, to "off", keeps that.
    os.environ.setdefault(autosave.ENV_FLAG, "on")

    print(f"\n  AgroSuite is running at {url}")
    print("  Close this window to stop the app.\n")

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    uvicorn.run(
        "agrosuite.app.server:app",
        host=args.host,
        port=port,
        reload=args.reload,
        log_level="warning",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
