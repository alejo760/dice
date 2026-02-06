#!/bin/bash

# Quick Start Script for Pneumonia Consolidation Segmentation
# This script sets up the environment and runs initial tests

echo "🫁 Pneumonia Consolidation Segmentation - Quick Start"
echo "======================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "dice_calculator_app.py" ]; then
    echo "❌ Error: Please run this script from the dice/ directory"
    exit 1
fi

echo "✓ Directory check passed"
echo ""

# Step 1: Install dependencies
echo "📦 Step 1: Installing dependencies..."
echo "Command: pip install -r requirements.txt"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "❌ Error installing dependencies"
    exit 1
fi
echo ""

# Step 2: Test Streamlit installation
echo "🧪 Step 2: Testing Streamlit installation..."
streamlit --version
if [ $? -eq 0 ]; then
    echo "✓ Streamlit is working"
else
    echo "❌ Streamlit test failed"
    exit 1
fi
echo ""

# Step 3: Create sample test images (optional)
echo "📸 Step 3: Would you like to preprocess sample images now?"
echo "This will enhance chest X-rays for better consolidation visibility."
echo ""
read -p "Enter 'y' to preprocess images, or 'n' to skip: " preprocess

if [ "$preprocess" = "y" ]; then
    echo ""
    echo "Enter the path to your input directory (e.g., ../data/Pacientes/):"
    read input_dir
    
    if [ -d "$input_dir" ]; then
        echo "Processing images from: $input_dir"
        echo "Output will be saved to: ./enhanced_images/"
        python preprocessing_consolidation.py \
            --input "$input_dir" \
            --output ./enhanced_images/ \
            --batch \
            --extension .jpg
        
        echo "✓ Preprocessing complete"
    else
        echo "❌ Directory not found: $input_dir"
    fi
fi
echo ""

# Step 4: Launch Streamlit app
echo "🚀 Step 4: Launch Dice Calculator App"
echo ""
echo "The Streamlit app will open in your browser."
echo "You can then upload images and masks to calculate Dice scores."
echo ""
read -p "Press Enter to launch the app, or Ctrl+C to exit..."

streamlit run dice_calculator_app.py

echo ""
echo "✓ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Annotate your images using CVAT or Label Studio"
echo "2. Use the Dice Calculator to validate annotations"
echo "3. See TODO.md for complete project roadmap"
