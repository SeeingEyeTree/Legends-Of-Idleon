"""Command-line entry point: ``python -m legends_of_idleon <command>``."""

from __future__ import annotations

import argparse
import time

COUNTDOWN_SECONDS = 5


def _countdown(seconds=COUNTDOWN_SECONDS):
    print(f"Switch to the game window. Starting in {seconds} seconds...")
    time.sleep(seconds)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="legends_of_idleon", description="Quality-of-life automation for Legends of Idleon."
    )
    parser.add_argument(
        "--no-countdown", action="store_true", help="start immediately instead of waiting 5 seconds"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    cook = sub.add_parser("cook", help="fill the kitchens and level up meals")
    cook.add_argument("ladles", help="ladles to use per meal (digits only)")

    sub.add_parser("anvil", help="collect from every anvil")

    vials = sub.add_parser("vials", help="run the vial routine (start with inventory open)")
    vials.add_argument("loops", type=int, help="number of times to loop")

    worship = sub.add_parser("read-worship", help="read a number from the Worship screen")
    worship.add_argument("what", choices=["charge-rate", "cur-charge"])

    sub.add_parser("tree-bot", help="experimental timing bot (unfinished)")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.no_countdown:
        _countdown()

    # Imports are deferred: they need Windows-only libraries.
    if args.command == "cook":
        from .tasks.cooking import cook

        cook(args.ladles)
    elif args.command == "anvil":
        from .tasks.anvil import collect_all

        collect_all()
    elif args.command == "vials":
        from .tasks.vials import run_vials

        run_vials(args.loops)
    elif args.command == "read-worship":
        from .digits import read_worship

        print(read_worship(args.what))
    elif args.command == "tree-bot":
        from .tasks.tree_bot import run

        run()


if __name__ == "__main__":
    main()
