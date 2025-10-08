# src/moderators/cli.py
import argparse
import json
from dataclasses import asdict, is_dataclass

from moderators.auto_model import AutoModerator


def _to_jsonable(obj):
    """Convert objects to JSON-serializable format."""
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (list, dict, str, int, float)) or obj is None:
        return obj
    return str(obj)


def main():
    """Run the moderators CLI."""
    parser = argparse.ArgumentParser(prog="moderators", description="Moderators CLI")
    parser.add_argument("model", nargs="?", help="Local model folder or HF model id")
    parser.add_argument("input", nargs="?", help="Input text or file path")
    parser.add_argument("--local-files-only", action="store_true", dest="local_files_only", help="Use only local files")
    args = parser.parse_args()

    if not args.model:
        parser.print_help()
        return 0

    mod = AutoModerator.from_pretrained(args.model, local_files_only=args.local_files_only)
    if args.input:
        out = mod(args.input)
        print(json.dumps([_to_jsonable(x) for x in out], ensure_ascii=False, indent=2))
    else:
        print("Model loaded. Provide the 'input' argument to run inference.")


if __name__ == "__main__":
    main()
