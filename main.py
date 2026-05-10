import argparse

from data.generator import generate_data
from data.validator import validate_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smart Oak Insurance data tools")
    parser.add_argument("--generate-data", action="store_true", help="Generate synthetic quote and claim datasets")
    parser.add_argument("--validate-data", action="store_true", help="Validate generated datasets")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.generate_data:
        quotes_df, claims_df = generate_data()
        print(f"Generated {len(quotes_df)} quotes and {len(claims_df)} claims")
        return

    if args.validate_data:
        validate_data()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
