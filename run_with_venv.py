#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Virtual Environment Runner Script

This script helps users run Chapter 7 demonstration code in a virtual environment
without manually activating the virtual environment.

Usage:
python run_with_venv.py [demo_type] [args...]

demo_type options:
- quick: Quick demonstration
- full: Complete demonstration
- interactive: Interactive tool
- pipeline: ISP pipeline demonstration
- analysis: Performance analysis
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


class VenvRunner:
    """Virtual Environment Runner"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / "venv"
        self.is_windows = os.name == 'nt'
        
    def get_venv_python(self):
        """Get Python path in virtual environment"""
        if self.is_windows:
            return self.venv_path / "Scripts" / "python.exe"
        else:
            return self.venv_path / "bin" / "python"
    
    def check_venv_exists(self):
        """Check virtual environment是否存在"""
        python_path = self.get_venv_python()
        return python_path.exists()
    
    def run_quick_demo(self, args):
        """Run quick demonstration"""
        print("🚀 Starting AI-driven ISP tuningQuickDemonstration...")
        script_path = self.project_root / "quick_start_demo.py"
        return self.run_script(script_path, args)
    
    def run_full_demo(self, args):
        """Run complete demonstration"""
        print("🎯 Starting AI-driven ISP tuningCompleteDemonstration...")
        script_path = self.project_root / "adaptive_isp_demo.py"
        
        # SettingsDefaultParameter
        if not args:
            args = ["--input", "test_data/876.png", "--output", "results"]
        
        return self.run_script(script_path, args)
    
    def run_interactive_tool(self, args):
        """Runinteractive tool"""
        print("🖥️  启动交互式ISPComparisonTool...")
        script_path = self.project_root / "isp_comparison_tool.py"
        return self.run_script(script_path, args)
    
    def run_pipeline_demo(self, args):
        """RunISP pipeline demonstration"""
        print("⚙️  启动ISP pipeline demonstration...")
        script_path = self.project_root / "isp_pipeline_demo.py"
        return self.run_script(script_path, args)
    
    def run_analysis(self, args):
        """Runperformance analysis"""
        print("📊 启动performance analysisTool...")
        script_path = self.project_root / "performance_analyzer.py"
        
        # SettingsDefaultParameter
        if not args:
            args = ["--input_dir", "images", "--output_dir", "analysis"]
        
        return self.run_script(script_path, args)
    
    def run_script(self, script_path, args):
        """Run指定Script"""
        if not script_path.exists():
            print(f"❌ Script file not found: {script_path}")
            return False
        
        if not self.check_venv_exists():
            print("❌ virtual environment未找到，please run setup_environment.py to create virtual environment")
            return False
        
        python_path = self.get_venv_python()
        
        try:
            # Build command
            cmd = [str(python_path), str(script_path)] + args
            
            print(f"Executing command: {' '.join(cmd)}")
            print("-" * 60)
            
            # RunScript
            result = subprocess.run(cmd, check=True)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Run failed, exit code: {e.returncode}")
            return False
        except KeyboardInterrupt:
            print("\n⚠️  User interrupted execution")
            return False
        except Exception as e:
            print(f"❌ Run error: {e}")
            return False
    
    def create_test_image(self):
        """CreateTestImage"""
        print("📸 CreateTestImage...")
        
        try:
            import cv2
            import numpy as np
            
            # CreateTestImage
            height, width = 400, 400
            image = np.ones((height, width, 3), dtype=np.uint8) * 128
            
            # Add some geometric shapes
            cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)  # Red rectangle
            cv2.circle(image, (300, 100), 50, (0, 255, 0), -1)  # Green circle
            cv2.ellipse(image, (200, 300), (80, 40), 45, 0, 360, (0, 0, 255), -1)  # Blue ellipse
            
            # Save image
            test_image_path = self.project_root / "test_image.jpg"
            cv2.imwrite(str(test_image_path), image)
            print(f"✅ Test image created: {test_image_path}")
            
            return True
            
        except ImportError:
            print("⚠️  OpenCV not installed, skipping test image creation")
            return False
        except Exception as e:
            print(f"❌ CreateTestImageFailed: {e}")
            return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Run Chapter 7 AI-driven ISP tuning demonstration in virtual environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python run_with_venv.py quick                    # QuickDemonstration
  python run_with_venv.py full --input image.jpg  # CompleteDemonstration
  python run_with_venv.py interactive             # interactive tool
  python run_with_venv.py pipeline                # ISP pipeline demonstration
  python run_with_venv.py analysis                # performance analysis
        """
    )
    
    parser.add_argument('demo_type', 
                       choices=['quick', 'full', 'interactive', 'pipeline', 'analysis'],
                       help='DemonstrationType')
    
    parser.add_argument('args', nargs='*', 
                       help='Parameters passed to demonstration script')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Chapter 7: AI-driven ISP Tuning Automation - Virtual Environment Runner")
    print("=" * 60)
    
    runner = VenvRunner()
    
    # Check virtual environment
    if not runner.check_venv_exists():
        print("❌ Virtual environment not found!")
        print("\nPlease run the following command to create virtual environment:")
        print("python setup_environment.py")
        sys.exit(1)
    
    print(f"✅ Virtual environment found: {runner.venv_path}")
    
    # Run corresponding script based on demonstration type
    success = False
    
    if args.demo_type == 'quick':
        success = runner.run_quick_demo(args.args)
    elif args.demo_type == 'full':
        success = runner.run_full_demo(args.args)
    elif args.demo_type == 'interactive':
        success = runner.run_interactive_tool(args.args)
    elif args.demo_type == 'pipeline':
        success = runner.run_pipeline_demo(args.args)
    elif args.demo_type == 'analysis':
        success = runner.run_analysis(args.args)
    
    if success:
        print("\n🎉 DemonstrationRunSuccessCompleted！")
    else:
        print("\n❌ DemonstrationRunFailed")
        sys.exit(1)


if __name__ == "__main__":
    main()
