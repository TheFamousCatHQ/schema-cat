import sys
import subprocess


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in {"major", "minor", "patch"}:
        print("Usage: bump-version [major|minor|patch]")
        sys.exit(1)
    part = sys.argv[1]
    try:
        subprocess.run(["poetry", "version", part], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to bump version: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 