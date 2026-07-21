"""
cli.py — command-line entry point for iMem (the `imem` command).

Subcommands wrap the library functions: `add` (indexing), `query` (search), and
`collection ls/rm` (management).

Usage:
    python -m src.cli add ~/Photos --collection personal
    python -m src.cli query "red cat on sofa"
    python -m src.cli query ~/reference.jpg --collection personal -k 10
    python -m src.cli collection ls
    python -m src.cli collection rm old_collection
"""

from __future__ import annotations

import argparse
import json
import sys

from qdrant_client import QdrantClient

from .catalog import collection_count, collection_exists, drop_collection, list_collections
from .config import CONFIG
from .encoder import ImageTextEncoder
from .indexer import DEFAULT_EXTENSIONS, _parse_extensions, index_folders
from .query import query_images
from .vector_store import ImageVectorStore


def _connect(collection: str, embedding_dim: int, recreate: bool = False) -> ImageVectorStore:
    """
    Connects to Qdrant in server mode (host/port from config). A unified
    connection factory (embedded default + --qdrant-url + env override) is a
    separate upcoming chunk.
    """
    return ImageVectorStore(
        collection,
        embedding_dim,
        host=CONFIG.qdrant.host,
        port=CONFIG.qdrant.port,
        recreate=recreate,
    )


def _connect_client() -> QdrantClient:
    """
    Bare Qdrant connection for catalog operations, which must not create a
    collection as a side effect (so they can't go through _connect). Same
    server-mode target as _connect; consolidated in the connection-factory chunk.
    """
    return QdrantClient(host=CONFIG.qdrant.host, port=CONFIG.qdrant.port)


def _cmd_add(args: argparse.Namespace) -> int:
    extensions = _parse_extensions(args.extensions)

    encoder = ImageTextEncoder()
    store = _connect(args.collection, encoder.embedding_dim, recreate=args.recreate)
    report = index_folders(args.folders, store, encoder, extensions=extensions)

    print(
        f"Found {report.found}, skipped {report.skipped_existing} (already indexed), "
        f"indexed {report.indexed}, failed {len(report.failed)}."
    )
    for path in report.failed:
        print(f"  failed: {path}")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    mode = "text" if args.text else "image" if args.image else "auto"

    encoder = ImageTextEncoder()
    store = _connect(args.collection, encoder.embedding_dim)
    results = query_images(args.query, store, encoder, top_k=args.top_k, mode=mode)

    if args.json:
        print(json.dumps([{"path": p, "score": s} for p, s in results], indent=2))
    else:
        for path, _ in results:
            print(path)
    return 0


def _cmd_collection_ls(args: argparse.Namespace) -> int:
    rows = list_collections(_connect_client())

    if args.json:
        print(json.dumps([{"name": name, "images": count} for name, count in rows], indent=2))
        return 0

    if not rows:
        print("No collections.")
        return 0

    width = max(len(name) for name, _ in rows)
    print(f"{'NAME'.ljust(width)}  IMAGES")
    for name, count in rows:
        print(f"{name.ljust(width)}  {count}")
    return 0


def _cmd_collection_rm(args: argparse.Namespace) -> int:
    client = _connect_client()

    if not collection_exists(client, args.name):
        print(f"error: collection '{args.name}' not found", file=sys.stderr)
        return 1

    if not args.yes:
        count = collection_count(client, args.name)
        reply = input(f"Delete collection '{args.name}' ({count} images)? [y/N] ")
        if reply.strip().lower() not in {"y", "yes"}:
            print("Aborted.")
            return 1

    drop_collection(client, args.name)
    print(f"Deleted collection '{args.name}'.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="imem", description="Semantic image memory: index and search images.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_add = subparsers.add_parser("add", help="Index images from folders into a collection.")
    p_add.add_argument("folders", nargs="+", help="Folder(s) to scan recursively for images.")
    p_add.add_argument(
        "--collection",
        default=CONFIG.qdrant.collection,
        help=f"Collection to index into (default: {CONFIG.qdrant.collection}).",
    )
    p_add.add_argument(
        "--extensions",
        default=None,
        help="Comma-separated file extensions to scan for, e.g. .jpg,.png "
        f"(default: {','.join(sorted(DEFAULT_EXTENSIONS))}).",
    )
    p_add.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection instead of adding to it.",
    )
    p_add.set_defaults(func=_cmd_add)

    p_query = subparsers.add_parser("query", help="Search an indexed collection by text or image.")
    p_query.add_argument("query", help="Text query, or path to a reference image.")
    mode_group = p_query.add_mutually_exclusive_group()
    mode_group.add_argument("--text", action="store_true", help="Force treating the query as text.")
    mode_group.add_argument("--image", action="store_true", help="Force treating the query as an image path.")
    p_query.add_argument(
        "--collection",
        default=CONFIG.qdrant.collection,
        help=f"Collection to search (default: {CONFIG.qdrant.collection}).",
    )
    p_query.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=CONFIG.vector_store.top_k,
        help=f"Number of results to return (default: {CONFIG.vector_store.top_k}).",
    )
    p_query.add_argument("--json", action="store_true", help="Output paths and scores as JSON.")
    p_query.set_defaults(func=_cmd_query)

    p_collection = subparsers.add_parser("collection", help="Manage collections.")
    collection_sub = p_collection.add_subparsers(dest="collection_command", required=True)

    p_ls = collection_sub.add_parser("ls", help="List collections and their image counts.")
    p_ls.add_argument("--json", action="store_true", help="Output as JSON.")
    p_ls.set_defaults(func=_cmd_collection_ls)

    p_rm = collection_sub.add_parser("rm", help="Delete a collection.")
    p_rm.add_argument("name", help="Collection to delete.")
    p_rm.add_argument("-f", "--yes", action="store_true", help="Skip the confirmation prompt.")
    p_rm.set_defaults(func=_cmd_collection_rm)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
