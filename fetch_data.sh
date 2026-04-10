#!/usr/bin/env bash
set -e
echo "ToolBench Data Download"

if ! command -v gdown &> /dev/null; then
  echo "Installing gdown..."
  pip install gdown --break-system-packages -q
fi
 
echo "Downloading from Google Drive (official ToolBench release)..."
gdown "1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk" -O data.zip
 
echo "Unzipping..."
unzip -q data.zip
rm data.zip
 
echo "Done."