@echo off
chcp 65001 >nul
echo ============================================
echo Chapter 7: AI-driven ISP Tuning Automation Demo
echo ============================================
echo.

echo Checking virtual environment...
if not exist "venv\Scripts\python.exe" (
    echo ❌ Virtual environment not found!
    echo.
    echo Please run the following command to create virtual environment:
    echo python setup_environment.py
    echo.
    pause
    exit /b 1
)

echo ✅ Virtual environment found
echo.

echo Select demo to run:
echo 1. Quick demo (recommended for beginners)
echo 2. Full demo
echo 3. Interactive tool
echo 4. ISP pipeline demo
echo 5. Performance analysis
echo 6. Create virtual environment
echo 7. Exit
echo.

set /p choice="Please enter your choice (1-7): "

if "%choice%"=="1" (
    echo.
    echo 🚀 Starting quick demo...
    venv\Scripts\python.exe quick_start_demo.py
) else if "%choice%"=="2" (
    echo.
    echo 🎯 Starting full demo...
        venv\Scripts\python.exe adaptive_isp_demo.py --input test_data/876.png --output results
) else if "%choice%"=="3" (
    echo.
    echo 🖥️ Starting interactive tool...
    venv\Scripts\python.exe isp_comparison_tool.py
) else if "%choice%"=="4" (
    echo.
    echo ⚙️ Starting ISP pipeline demo...
    venv\Scripts\python.exe isp_pipeline_demo.py
) else if "%choice%"=="5" (
    echo.
    echo 📊 Starting performance analysis...
    venv\Scripts\python.exe performance_analyzer.py --input_dir images --output_dir analysis
) else if "%choice%"=="6" (
    echo.
    echo 🔧 Creating virtual environment...
    python setup_environment.py
) else if "%choice%"=="7" (
    echo.
    echo 👋 Goodbye!
    exit /b 0
) else (
    echo.
    echo ❌ Invalid choice, please run the script again
)

echo.
pause
