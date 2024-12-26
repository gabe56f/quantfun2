#!/usr/bin/env bash

# Install the dependencies
git submodule update --init --recursive
# just in case, because older git versions are unreliable...
git submodule update --recursive

cd torch-cublas-hgemm
# memory problems xd
MAX_JOBS=1 pip install -U -v .

cd ..