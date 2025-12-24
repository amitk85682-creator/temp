#!/bin/bash

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

echo "✅ Setup complete! Don't forget to:"
echo "1. Add your tokens to .env file"
echo "2. Run: python bot.py"
