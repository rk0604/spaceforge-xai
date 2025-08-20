#!/bin/bash

BUILD_DIR="build"

if [ "$1" == "clean" ]; then
    rm -rf "$BUILD_DIR"
    exit 0
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || exit
cmake ..
cmake --build .
./sim
