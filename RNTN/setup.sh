#!/bin/sh

# Create the neccessary folders
if [ -d "models" ]; then
    echo "models already exists"
else
    echo "Create models directory..."
    mkdir models
fi

if [ -d "trees" ]; then
    echo "trees directory already exists"
else
    echo "Creating trees directory..."
    mkdir trees
fi

if [ -d "output" ]; then
    echo "output directory already exists"
else
    echo "Creating output directory..."
    mkdir output
fi
