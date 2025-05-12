#!/usr/bin/env bash
set -euo pipefail

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

conda run -n cuda118 bash -c '
  rm -rf build                      # wipe old cache
  cmake -S . -B build \
         -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j
'
