#!/bin/bash

MODULE=$1

set -ex

source $(dirname "$0")/env_setup.sh

install_all
setup_test_env

python -m pytest -n 2 -v --capture=tee-sys --log-cli-level=INFO --junitxml=results.xml tests/unittests/$MODULE/
