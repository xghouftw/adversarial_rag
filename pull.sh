#!/bin/bash
cd "$(dirname "$0")"
rclone copy "gdrive:colab/$(basename "$PWD")-shared/outputs" outputs -P "$@"
