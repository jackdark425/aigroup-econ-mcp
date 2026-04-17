"""Command-line entry point: ``aigroup-econ-mcp``."""

from __future__ import annotations

import argparse
import logging
import sys

from . import __author__, __email__, __version__
from .server import main as run_server


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aigroup-econ-mcp",
        description="AIGroup Econometrics MCP server (stdio transport).",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="store_true",
        help="print version and exit",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="enable debug logging",
    )
    return parser


def cli() -> None:
    args = _parser().parse_args()

    if args.version:
        print(f"aigroup-econ-mcp {__version__}")
        print(f"{__author__} <{__email__}>")
        return

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    try:
        run_server()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    cli()
