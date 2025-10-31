#!/bin/bash
# Installation script for OMR System

echo "================================"
echo "OMR Evaluation System Setup"
echo "================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "================================"
echo "Setup complete!"
echo "================================"
echo ""
echo "To start the application:"
echo "1. source venv/bin/activate"
echo "2. python app.py"
echo ""
echo "Then open: http://localhost:5000"
