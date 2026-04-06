"""CLI interface for the NeuroPool Context Engine."""

import argparse
import sys

from .engine import ContextEngine
from .token_counter import count_tokens_approximate


def cmd_count(args):
    text = sys.stdin.read() if args.file == "-" else open(args.file).read()
    print(count_tokens_approximate(text))


def cmd_info(args):
    engine = ContextEngine(max_tokens=args.max_tokens, model=args.model)
    print(engine.summarize())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="npe",
        description="NeuroPool Context Engine - manage token budgets effectively",
    )
    parser.add_argument("--model", default="gpt-4", help="Model name (default: gpt-4)")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Token budget (default: 8192)")

    sub = parser.add_subparsers(dest="command", required=True)

    # count subcommand
    count_p = sub.add_parser("count", help="Count tokens in text from stdin or a file")
    count_p.add_argument("file", nargs="?", default="-",
                         help="File to read (default: stdin)")
    count_p.set_defaults(func=cmd_count)

    # info subcommand
    info_p = sub.add_parser("info", help="Show engine configuration")
    info_p.set_defaults(func=cmd_info)

    # TODO: Add 'compress' subcommand to summarize a conversation and reduce token usage.
    # TODO: Add 'chat' subcommand for interactive context-managed conversation.

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
