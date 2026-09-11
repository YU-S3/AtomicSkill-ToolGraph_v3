"""Frozen-protocol entry point for common baseline manifest generation."""

from __future__ import annotations

from .common.manifest_builder import main


if __name__ == "__main__":
    raise SystemExit(main())
