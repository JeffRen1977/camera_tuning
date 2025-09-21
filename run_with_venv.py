#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
虚拟EnvironmentRunScript

这个Script帮助用户在虚拟Environment中Run第7章的Demonstration代码，
无需手动激活虚拟Environment。

UseMethod：
python run_with_venv.py [demo_type] [args...]

demo_type Option：
- quick: QuickDemonstration
- full: CompleteDemonstration
- interactive: 交互式Tool
- pipeline: ISP流水LineDemonstration
- analysis: PerformanceAnalyze
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


class VenvRunner:
    """虚拟EnvironmentRun器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / "venv"
        self.is_windows = os.name == 'nt'
        
    def get_venv_python(self):
        """Get虚拟Environment中的PythonPath"""
        if self.is_windows:
            return self.venv_path / "Scripts" / "python.exe"
        else:
            return self.venv_path / "bin" / "python"
    
    def check_venv_exists(self):
        """Check虚拟Environment是否存在"""
        python_path = self.get_venv_python()
        return python_path.exists()
    
    def run_quick_demo(self, args):
        """RunQuickDemonstration"""
        print("🚀 启动AI驱动ISP调优QuickDemonstration...")
        script_path = self.project_root / "quick_start_demo.py"
        return self.run_script(script_path, args)
    
    def run_full_demo(self, args):
        """RunCompleteDemonstration"""
        print("🎯 启动AI驱动ISP调优CompleteDemonstration...")
        script_path = self.project_root / "adaptive_isp_demo.py"
        
        # SettingsDefaultParameter
        if not args:
            args = ["--input", "test_data/876.png", "--output", "results"]
        
        return self.run_script(script_path, args)
    
    def run_interactive_tool(self, args):
        """Run交互式Tool"""
        print("🖥️  启动交互式ISPComparisonTool...")
        script_path = self.project_root / "isp_comparison_tool.py"
        return self.run_script(script_path, args)
    
    def run_pipeline_demo(self, args):
        """RunISP流水LineDemonstration"""
        print("⚙️  启动ISP流水LineDemonstration...")
        script_path = self.project_root / "isp_pipeline_demo.py"
        return self.run_script(script_path, args)
    
    def run_analysis(self, args):
        """RunPerformanceAnalyze"""
        print("📊 启动PerformanceAnalyzeTool...")
        script_path = self.project_root / "performance_analyzer.py"
        
        # SettingsDefaultParameter
        if not args:
            args = ["--input_dir", "images", "--output_dir", "analysis"]
        
        return self.run_script(script_path, args)
    
    def run_script(self, script_path, args):
        """Run指定Script"""
        if not script_path.exists():
            print(f"❌ ScriptFile未找到: {script_path}")
            return False
        
        if not self.check_venv_exists():
            print("❌ 虚拟Environment未找到，请先Run setup_environment.py Create虚拟Environment")
            return False
        
        python_path = self.get_venv_python()
        
        try:
            # Build命令
            cmd = [str(python_path), str(script_path)] + args
            
            print(f"执Row命令: {' '.join(cmd)}")
            print("-" * 60)
            
            # RunScript
            result = subprocess.run(cmd, check=True)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ RunFailed，退出码: {e.returncode}")
            return False
        except KeyboardInterrupt:
            print("\n⚠️  用户中断执Row")
            return False
        except Exception as e:
            print(f"❌ Run出错: {e}")
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
            
            # Add一些几何形状
            cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)  # 红色矩形
            cv2.circle(image, (300, 100), 50, (0, 255, 0), -1)  # 绿色圆形
            cv2.ellipse(image, (200, 300), (80, 40), 45, 0, 360, (0, 0, 255), -1)  # 蓝色椭圆
            
            # SaveImage
            test_image_path = self.project_root / "test_image.jpg"
            cv2.imwrite(str(test_image_path), image)
            print(f"✅ TestImage已Create: {test_image_path}")
            
            return True
            
        except ImportError:
            print("⚠️  OpenCV未Install，SkipTestImageCreate")
            return False
        except Exception as e:
            print(f"❌ CreateTestImageFailed: {e}")
            return False


def main():
    """主Function"""
    parser = argparse.ArgumentParser(
        description="在虚拟Environment中Run第7章AI驱动ISP调优Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python run_with_venv.py quick                    # QuickDemonstration
  python run_with_venv.py full --input image.jpg  # CompleteDemonstration
  python run_with_venv.py interactive             # 交互式Tool
  python run_with_venv.py pipeline                # ISP流水LineDemonstration
  python run_with_venv.py analysis                # PerformanceAnalyze
        """
    )
    
    parser.add_argument('demo_type', 
                       choices=['quick', 'full', 'interactive', 'pipeline', 'analysis'],
                       help='DemonstrationType')
    
    parser.add_argument('args', nargs='*', 
                       help='传递给DemonstrationScript的Parameter')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("第7章：AI驱动的ISP调优Automation - 虚拟EnvironmentRun器")
    print("=" * 60)
    
    runner = VenvRunner()
    
    # Check虚拟Environment
    if not runner.check_venv_exists():
        print("❌ 虚拟Environment未找到！")
        print("\n请先Run以下命令Create虚拟Environment:")
        print("python setup_environment.py")
        sys.exit(1)
    
    print(f"✅ 虚拟Environment已找到: {runner.venv_path}")
    
    # 根据DemonstrationTypeRun相应Script
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
