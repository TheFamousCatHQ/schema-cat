import subprocess
import sys


def main():
    print("Running tests with pytest...")
    result = subprocess.run([sys.executable, '-m', 'pytest'])
    if result.returncode != 0:
        print("Tests failed. Aborting build.")
        sys.exit(result.returncode)
    print("Tests passed. Running poetry build...")
    build_result = subprocess.run(['poetry', 'build'])
    sys.exit(build_result.returncode)


if __name__ == "__main__":
    main()
