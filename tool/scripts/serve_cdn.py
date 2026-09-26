#!/usr/bin/env python3
"""Loopback static mirror with the model CDN's GET/CORS/cache headers."""
import argparse
import functools
import http.server
import pathlib


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        super().end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=pathlib.Path, required=True)
    parser.add_argument("--port", type=int, default=8877)
    args = parser.parse_args()
    if not args.directory.is_dir():
        parser.error("directory must exist")
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", args.port), functools.partial(Handler, directory=str(args.directory.resolve()))
    )
    print(f"Serving {args.directory.resolve()} at http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
