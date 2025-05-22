#!/bin/sh

set -e
poetry run check-and-build
poetry publish
poetry run bump-version
