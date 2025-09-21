#!/bin/bash

echo "============================================"
echo "Chapter 7: AI-driven ISP Tuning Automation Demo"
echo "============================================"
echo

echo "Checking virtual environment..."
if [ ! -f "venv/bin/python" ]; then
    echo "❌ Virtual environment not found!"
    echo
    echo "Please run the following command to create virtual environment:"
    echo "python setup_environment.py"
    echo
    exit 1
fi

echo "✅ Virtual environment found"
echo

echo "Select demo to run:"
echo "1. Quick demo (recommended for beginners)"
echo "2. Full demo"
echo "3. Interactive tool"
echo "4. ISP pipeline demo"
echo "5. Performance analysis"
echo "6. Create virtual environment"
echo "7. Exit"
echo

read -p "Please enter your choice (1-7): " choice

case $choice in
    1)
        echo
        echo "🚀 Starting quick demo..."
        venv/bin/python quick_start_demo.py
        ;;
    2)
        echo
        echo "🎯 Starting full demo..."
        venv/bin/python adaptive_isp_demo.py --input test_data/876.png --output results
        ;;
    3)
        echo
        echo "🖥️ Starting interactive tool..."
        venv/bin/python isp_comparison_tool.py
        ;;
    4)
        echo
        echo "⚙️ Starting ISP pipeline demo..."
        venv/bin/python isp_pipeline_demo.py
        ;;
    5)
        echo
        echo "📊 Starting performance analysis..."
        venv/bin/python performance_analyzer.py --input_dir images --output_dir analysis
        ;;
    6)
        echo
        echo "🔧 Creating virtual environment..."
        python setup_environment.py
        ;;
    7)
        echo
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo
        echo "❌ Invalid choice, please run the script again"
        exit 1
        ;;
esac

echo
read -p "Press Enter to continue..."
