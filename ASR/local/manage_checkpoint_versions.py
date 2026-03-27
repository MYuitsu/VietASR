#!/usr/bin/env python3

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


DEFAULT_REGISTRY = Path("ASR/checkpoint_versions.json")


def load_registry(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {"versions": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def cmd_register(args: argparse.Namespace) -> None:
    registry = load_registry(args.registry)
    versions = registry.setdefault("versions", {})

    versions[args.name] = {
        "exp_dir": str(Path(args.exp_dir).resolve()),
        "notes": args.notes or "",
        "updated_at": utc_now(),
    }

    save_registry(args.registry, registry)
    print(f"Registered {args.name} -> {Path(args.exp_dir).resolve()}")


def cmd_remove(args: argparse.Namespace) -> None:
    registry = load_registry(args.registry)
    versions = registry.setdefault("versions", {})
    if args.name not in versions:
        raise KeyError(f"Unknown version: {args.name}")
    del versions[args.name]
    save_registry(args.registry, registry)
    print(f"Removed {args.name}")


def cmd_list(args: argparse.Namespace) -> None:
    registry = load_registry(args.registry)
    versions = registry.get("versions", {})
    if not versions:
        print("No checkpoint versions registered.")
        return

    print("name\texp_dir\tupdated_at\tnotes")
    for name in sorted(versions):
        item = versions[name]
        print(
            f"{name}\t{item.get('exp_dir', '')}\t"
            f"{item.get('updated_at', '')}\t{item.get('notes', '')}"
        )


def cmd_show(args: argparse.Namespace) -> None:
    registry = load_registry(args.registry)
    versions = registry.get("versions", {})
    if args.name not in versions:
        raise KeyError(f"Unknown version: {args.name}")
    print(json.dumps(versions[args.name], ensure_ascii=False, indent=2))


def cmd_resolve(args: argparse.Namespace) -> None:
    registry = load_registry(args.registry)
    versions = registry.get("versions", {})
    if args.name not in versions:
        raise KeyError(f"Unknown version: {args.name}")
    item = versions[args.name]
    value = item.get(args.field)
    if value is None:
        raise KeyError(f"Field {args.field!r} not found in version {args.name!r}")
    print(value)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage named checkpoint/experiment versions for ASR evaluation."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
        help="Path to checkpoint version registry JSON.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    register = subparsers.add_parser("register")
    register.add_argument("--name", required=True)
    register.add_argument("--exp-dir", required=True)
    register.add_argument("--notes", default="")
    register.set_defaults(func=cmd_register)

    remove = subparsers.add_parser("remove")
    remove.add_argument("--name", required=True)
    remove.set_defaults(func=cmd_remove)

    list_cmd = subparsers.add_parser("list")
    list_cmd.set_defaults(func=cmd_list)

    show = subparsers.add_parser("show")
    show.add_argument("--name", required=True)
    show.set_defaults(func=cmd_show)

    resolve = subparsers.add_parser("resolve")
    resolve.add_argument("--name", required=True)
    resolve.add_argument("--field", default="exp_dir")
    resolve.set_defaults(func=cmd_resolve)

    return parser


def main() -> None:
    parser = get_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
