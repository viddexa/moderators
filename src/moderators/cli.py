import argparse
import json
from pathlib import Path


def _settings_path() -> Path:
    base = Path.home() / ".moderators"
    base.mkdir(parents=True, exist_ok=True)
    return base / "settings.json"


def _read_settings() -> dict:
    p = _settings_path()
    if not p.exists():
        return {"sync": True}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"sync": True}


def _write_settings(d: dict) -> None:
    p = _settings_path()
    try:
        p.write_text(json.dumps(d))
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(prog="moderators", description="Moderators CLI")
    subparsers = parser.add_subparsers(dest="command")

    settings_parser = subparsers.add_parser("settings", help="Manage settings")
    # Support both `--sync false` and `sync=false` styles
    settings_parser.add_argument(
        "--sync", choices=["true", "false"], help="Enable/disable analytics sync"
    )
    # New: control auto-install behavior
    settings_parser.add_argument(
        "--autoinstall", choices=["true", "false"], help="Enable/disable auto-install of optional deps"
    )
    settings_parser.add_argument(
        "kv", nargs="*", help="Alternative key=value options (e.g., sync=false autoinstall=true)", default=[]
    )
    settings_parser.add_argument("--show", action="store_true", help="Show settings")

    args = parser.parse_args()

    if args.command == "settings":
        if args.show:
            print(json.dumps(_read_settings(), indent=2))
            return
        sync_val = args.sync
        autoinstall_val = args.autoinstall
        # Parse key=value if provided
        for entry in args.kv:
            if entry.startswith("sync="):
                sync_val = entry.split("=", 1)[1]
            if entry.startswith("autoinstall="):
                autoinstall_val = entry.split("=", 1)[1]
        if sync_val is not None or autoinstall_val is not None:
            current = _read_settings()
            if sync_val is not None:
                current["sync"] = str(sync_val).lower() == "true"
            if autoinstall_val is not None:
                current["autoinstall"] = str(autoinstall_val).lower() == "true"
            _write_settings(current)
            print("Saved settings.")
            return
        parser.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
