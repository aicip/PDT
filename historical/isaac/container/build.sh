#!/bin/bash
set -e

if [ $# -ne 1 ]; then
    echo "usage: $0 <container.def>"
    echo -e "\tBuilds <container.def> to <container.sif>"
    exit 1
fi

filename=$1
f="${filename%.*}"
mkdir -p ${SCRATCH}/containers
singularity build -F ${SCRATCH}/containers/$f.sif $1
