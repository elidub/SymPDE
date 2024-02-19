#!/bin/bash

echo "Touching data on scratch-shared..."

# Set the directories to touch files in
scratch_dir="/scratch-shared/eliasd"

# Find all files in the specified directories and update their modification timestamp
find "$scratch_dir" -type f -exec touch {} +

echo "Files touched successfully."

