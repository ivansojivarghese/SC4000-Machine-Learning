import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Write a response text to tmp/test_a.txt or tmp/test_b.txt")
    parser.add_argument("--which", choices=["a", "b"], required=True, help="Target file: a -> test_a.txt, b -> test_b.txt")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--text", type=str, help="Response text to write")
    group.add_argument("--in-file", type=str, help="Path to a file whose entire contents will be written")
    args = parser.parse_args()

    # Resolve output path
    repo_root = os.path.dirname(os.path.dirname(__file__))
    tmp_dir = os.path.join(repo_root, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    out_name = "test_a.txt" if args.which.lower() == "a" else "test_b.txt"
    out_path = os.path.join(tmp_dir, out_name)

    # Determine content
    if args.text is not None:
        content = args.text
        # Interpret common escape sequences in --text (e.g., "\n", "\t").
        # This does NOT run for --in-file or stdin, which already contain real newlines.
        # Also handle a user-typed "/n" case by converting it to a newline conservatively.
        content = content.replace("\\r", "\r").replace("\\n", "\n").replace("\\t", "\t")
        # Optional: map literal "/n" to newline if present (helps when shell escaped backslashes)
        if "/n" in content and "http" not in content:  # avoid mangling URLs
            content = content.replace("/n", "\n")
    elif args.in_file:
        with open(args.in_file, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        # Read from stdin (supports multi-line)
        content = sys.stdin.read()
        if not content.strip():
            parser.error("No input provided. Use --text, --in-file, or pipe text via stdin.")

    # Write file
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[write_response] Wrote {len(content)} chars to {out_path}")


if __name__ == "__main__":
    main()
