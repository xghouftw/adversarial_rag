#!/bin/bash
cd "$(dirname "$0")"
rclone copy "gdrive:colab/$(basename "$PWD")/outputs" outputs -P "$@"
