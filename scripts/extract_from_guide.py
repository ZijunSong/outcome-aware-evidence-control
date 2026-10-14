import argparse
import re
from pathlib import Path


def extract(markdown: Path, destination: Path):
    pattern = r"<!-- file: ([^\n]+) -->\n```[^\n]*\n(.*?)\n```"
    matches = re.findall(pattern, markdown.read_text(encoding="utf-8"), re.DOTALL)
    if not matches:
        raise ValueError("No marked source files found")
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    seen = set()
    for name, content in matches:
        relative = Path(name.strip())
        if relative.is_absolute() or ".." in relative.parts or relative in seen:
            raise ValueError("Duplicate or unsafe file path")
        seen.add(relative)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content + "\n", encoding="utf-8")
    print(f"Extracted {len(seen)} files into {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    extract(args.md, args.out)
