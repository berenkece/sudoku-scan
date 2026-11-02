import argparse
from pathlib import Path

from pipeline.io_utils import ensure_output_dir, save_json, save_text
from pipeline.validate import validate_solution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan and solve a Sudoku image")
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument("--output-dir", type=Path, default=Path("out"), help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Save intermediate images")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image}")

    ensure_output_dir(args.output_dir)
    result = {"image": str(args.image), "grid": None, "valid": False}
    result["valid"] = validate_solution(result["grid"])
    save_json(args.output_dir / "result.json", result)
    save_text(args.output_dir / "result.txt", "Pipeline not yet implemented\n")
    print(f"Created outputs in {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
