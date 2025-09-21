#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Virtual Environment Setup Script

This script helps users automatically set up the virtual environment required 
to run the Chapter 7 AI-driven ISP tuning demonstration.
Includes:
1. Create virtual environment
2. Install required dependencies
3. Verify installation
4. Provide usage guide

Usage:
python setup_environment.py
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


class EnvironmentSetup:
    """虚拟EnvironmentSettingsClass"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.venv_path = self.project_root / "venv"
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.is_windows = platform.system() == "Windows"
        
    def check_python_version(self):
        """CheckPythonVersion"""
        print("CheckPythonVersion...")
        if sys.version_info < (3, 8):
            print("❌ Error: NeedPython 3.8或更高Version")
            print(f"当前Version: {sys.version}")
            return False
        
        print(f"✅ PythonVersionCheck通过: {sys.version}")
        return True
    
    def create_virtual_environment(self):
        """Create虚拟Environment"""
        print("\nCreate虚拟Environment...")
        
        if self.venv_path.exists():
            print(f"⚠️  虚拟Environment已存在: {self.venv_path}")
            response = input("是否RestartCreate? (y/N): ").lower()
            if response == 'y':
                print("Delete现有虚拟Environment...")
                import shutil
                shutil.rmtree(self.venv_path)
            else:
                print("Use现有虚拟Environment")
                return True
        
        try:
            # Create虚拟Environment
            cmd = [sys.executable, "-m", "venv", str(self.venv_path)]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ 虚拟EnvironmentCreateSuccess: {self.venv_path}")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Create虚拟EnvironmentFailed: {e}")
            print(f"ErrorOutput: {e.stderr}")
            return False
    
    def get_pip_path(self):
        """Get虚拟Environment中的pipPath"""
        if self.is_windows:
            return self.venv_path / "Scripts" / "pip.exe"
        else:
            return self.venv_path / "bin" / "pip"
    
    def get_python_path(self):
        """Get虚拟Environment中的pythonPath"""
        if self.is_windows:
            return self.venv_path / "Scripts" / "python.exe"
        else:
            return self.venv_path / "bin" / "python"
    
    def install_dependencies(self):
        """InstallDependency包"""
        print("\nInstallDependency包...")
        
        pip_path = self.get_pip_path()
        if not pip_path.exists():
            print(f"❌ pip未找到: {pip_path}")
            return False
        
        # 定义Dependency包List
        dependencies = [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
            "opencv-python>=4.5.0",
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "pandas>=1.3.0",
            "numpy>=1.21.0",
            "Pillow>=8.3.0",
            "tqdm>=4.62.0",
            "pyyaml>=5.4.0",
            "easydict>=1.9.0",
        ]
        
        print("InstallCoreDependency包...")
        for dep in dependencies:
            print(f"  Install {dep}...")
            try:
                cmd = [str(pip_path), "install", dep]
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"  ✅ {dep} InstallSuccess")
            except subprocess.CalledProcessError as e:
                print(f"  ❌ {dep} InstallFailed: {e}")
                print(f"  ErrorOutput: {e.stderr}")
                return False
        
        # InstallOptionalDependency
        optional_dependencies = [
            "tensorboard>=2.7.0",
            "wandb>=0.12.0",
            "jupyter>=1.0.0",
        ]
        
        print("\nInstallOptionalDependency包...")
        for dep in optional_dependencies:
            print(f"  Install {dep}...")
            try:
                cmd = [str(pip_path), "install", dep]
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"  ✅ {dep} InstallSuccess")
            except subprocess.CalledProcessError as e:
                print(f"  ⚠️  {dep} InstallFailed，Skip")
        
        print("✅ Dependency包InstallCompleted")
        return True
    
    def create_requirements_file(self):
        """Createrequirements.txtFile"""
        print("\nCreaterequirements.txtFile...")
        
        requirements_content = """# 第7章：AI驱动的ISP调优Automation - Dependency包List
# CoreDependency
torch>=1.9.0
torchvision>=0.10.0
opencv-python>=4.5.0
matplotlib>=3.3.0
seaborn>=0.11.0
pandas>=1.3.0
numpy>=1.21.0
Pillow>=8.3.0
tqdm>=4.62.0
pyyaml>=5.4.0
easydict>=1.9.0

# OptionalDependency
tensorboard>=2.7.0
wandb>=0.12.0
jupyter>=1.0.0

# Development和Debug
ipython>=7.25.0
jupyterlab>=3.0.0
"""
        
        requirements_path = self.project_root / "requirements.txt"
        with open(requirements_path, 'w', encoding='utf-8') as f:
            f.write(requirements_content)
        
        print(f"✅ requirements.txt 已Create: {requirements_path}")
        return True
    
    def create_activation_scripts(self):
        """Create激活Script"""
        print("\nCreate激活Script...")
        
        # Windows批HandleScript
        if self.is_windows:
            activate_bat = self.project_root / "activate_env.bat"
            with open(activate_bat, 'w') as f:
                f.write(f"""@echo off
echo 激活第7章ISP调优DemonstrationEnvironment...
call {self.venv_path}\\Scripts\\activate.bat
echo Environment已激活！
echo 现在可以RunDemonstrationScript了。
echo.
echo QuickStart：
echo   python quick_start_demo.py
echo.
echo DetailedDemonstration：
echo   python chapter7_adaptive_isp_demo.py --input test_image.jpg
echo.
cmd /k
""")
            print(f"✅ Windows激活Script已Create: {activate_bat}")
        
        # Unix shellScript
        activate_sh = self.project_root / "activate_env.sh"
        with open(activate_sh, 'w') as f:
            f.write(f"""#!/bin/bash
echo "激活第7章ISP调优DemonstrationEnvironment..."
source {self.venv_path}/bin/activate
echo "Environment已激活！"
echo "现在可以RunDemonstrationScript了。"
echo ""
echo "QuickStart："
echo "  python quick_start_demo.py"
echo ""
echo "DetailedDemonstration："
echo "  python chapter7_adaptive_isp_demo.py --input test_image.jpg"
echo ""
exec "$SHELL"
""")
        
        # Settings执RowPermission
        if not self.is_windows:
            os.chmod(activate_sh, 0o755)
            print(f"✅ Unix激活Script已Create: {activate_sh}")
    
    def create_run_scripts(self):
        """CreateRunScript"""
        print("\nCreateRunScript...")
        
        python_path = self.get_python_path()
        
        # QuickDemonstrationScript
        quick_demo_script = self.project_root / "run_quick_demo.py"
        with open(quick_demo_script, 'w') as f:
            f.write(f"""#!/usr/bin/env python3
import subprocess
import sys
import os

def run_quick_demo():
    \"\"\"RunQuickDemonstration\"\"\"
    print("启动第7章AI驱动ISP调优QuickDemonstration...")
    
    # Check虚拟Environment
    venv_python = "{python_path}"
    if not os.path.exists(venv_python):
        print("❌ 虚拟Environment未找到，请先Run setup_environment.py")
        return False
    
    try:
        # RunQuickDemonstration
        cmd = [venv_python, "quick_start_demo.py"]
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ RunFailed: {{e}}")
        return False
    except FileNotFoundError:
        print("❌ quick_start_demo.py File未找到")
        return False

if __name__ == "__main__":
    success = run_quick_demo()
    if not success:
        sys.exit(1)
""")
        
        if not self.is_windows:
            os.chmod(quick_demo_script, 0o755)
        
        print(f"✅ QuickDemonstrationScript已Create: {quick_demo_script}")
    
    def verify_installation(self):
        """VerifyInstall"""
        print("\nVerifyInstall...")
        
        python_path = self.get_python_path()
        if not python_path.exists():
            print(f"❌ Python未找到: {python_path}")
            return False
        
        # Test导入关Key包
        test_imports = [
            "torch",
            "cv2",
            "matplotlib",
            "numpy",
            "PIL"
        ]
        
        for package in test_imports:
            try:
                cmd = [str(python_path), "-c", f"import {package}; print('✅ {package} 导入Success')"]
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(result.stdout.strip())
            except subprocess.CalledProcessError:
                print(f"❌ {package} 导入Failed")
                return False
        
        print("✅ All包导入Test通过")
        return True
    
    def create_usage_guide(self):
        """CreateUse指南"""
        print("\nCreateUse指南...")
        
        python_path = self.get_python_path()
        
        usage_guide = f"""# 第7章：AI驱动的ISP调优Automation - Use指南

## EnvironmentSettings

虚拟Environment已Create并ConfigurationCompleted：
- 虚拟EnvironmentPath: {self.venv_path}
- PythonPath: {python_path}

## 激活Environment

### Windows:
```bash
# Method1: Use激活Script
activate_env.bat

# Method2: 手动激活
{self.venv_path}\\Scripts\\activate.bat
```

### macOS/Linux:
```bash
# Method1: Use激活Script
source activate_env.sh

# Method2: 手动激活
source {self.venv_path}/bin/activate
```

## RunDemonstration

### 1. QuickVolume验（推荐新手）
```bash
python quick_start_demo.py
```

### 2. CompleteDemonstration
```bash
# 单ImageHandle
python chapter7_adaptive_isp_demo.py --input test_image.jpg --output results

# BatchHandle
python chapter7_adaptive_isp_demo.py --input image_folder/ --batch --output batch_results
```

### 3. 交互式Tool
```bash
python isp_comparison_tool.py
```

### 4. ISP流水LineDemonstration
```bash
python isp_pipeline_demo.py
```

### 5. PerformanceAnalyze
```bash
python performance_analyzer.py --input_dir images/ --output_dir analysis/
```

## Use虚拟EnvironmentRun

如果不想激活Environment，可以直接Use虚拟Environment的Python：

```bash
# Windows
{self.venv_path}\\Scripts\\python.exe quick_start_demo.py

# macOS/Linux
{self.venv_path}/bin/python quick_start_demo.py
```

## Failure排除

### 1. 导入Error
如果遇到导入Error，请确保：
- 虚拟Environment已激活
- AllDependency包已正确Install

### 2. CUDASupport
如果NeedGPU加速：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. 内存不足
如果遇到内存Problem，可以：
- 减小批HandleSize
- DecreaseImage分辨率
- UseCPUMode

## ItemStructure

```
camera_tuning/
├── venv/                          # 虚拟Environment
├── AdaptiveISP/                   # 原始论文代码
├── chapter7_adaptive_isp_demo.py  # 主DemonstrationScript
├── isp_pipeline_demo.py          # ISP流水LineDemonstration
├── isp_comparison_tool.py        # 交互式ComparisonTool
├── performance_analyzer.py       # PerformanceAnalyzeTool
├── quick_start_demo.py           # QuickStartDemonstration
├── setup_environment.py          # EnvironmentSettingsScript
├── requirements.txt              # Dependency包List
├── activate_env.bat/.sh          # Environment激活Script
└── README_Chapter7.md            # DetailedDocumentation
```

## TechnologySupport

如果遇到Problem，请Check：
1. PythonVersion >= 3.8
2. 虚拟Environment是否正确激活
3. AllDependency包是否正确Install
4. FilePath是否正确

祝您Use愉快！
"""
        
        guide_path = self.project_root / "Use指南.md"
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(usage_guide)
        
        print(f"✅ Use指南已Create: {guide_path}")
    
    def setup(self):
        """执RowComplete的SettingsFlow"""
        print("=" * 60)
        print("第7章：AI驱动的ISP调优Automation - EnvironmentSettings")
        print("=" * 60)
        
        steps = [
            ("CheckPythonVersion", self.check_python_version),
            ("Create虚拟Environment", self.create_virtual_environment),
            ("InstallDependency包", self.install_dependencies),
            ("Createrequirements.txt", self.create_requirements_file),
            ("Create激活Script", self.create_activation_scripts),
            ("CreateRunScript", self.create_run_scripts),
            ("VerifyInstall", self.verify_installation),
            ("CreateUse指南", self.create_usage_guide),
        ]
        
        for step_name, step_func in steps:
            print(f"\n{'='*20} {step_name} {'='*20}")
            if not step_func():
                print(f"\n❌ SettingsFailed在Step: {step_name}")
                return False
        
        print("\n" + "=" * 60)
        print("🎉 EnvironmentSettingsCompleted！")
        print("=" * 60)
        
        print(f"\n下一步:")
        if self.is_windows:
            print(f"1. Run activate_env.bat 激活Environment")
        else:
            print(f"1. Run source activate_env.sh 激活Environment")
        print(f"2. Run python quick_start_demo.py StartDemonstration")
        
        print(f"\n或者直接Run:")
        python_path = self.get_python_path()
        if self.is_windows:
            print(f"{python_path} quick_start_demo.py")
        else:
            print(f"{python_path} quick_start_demo.py")
        
        return True


def main():
    """主Function"""
    setup = EnvironmentSetup()
    success = setup.setup()
    
    if not success:
        print("\n❌ EnvironmentSettingsFailed，请CheckErrorInformation")
        sys.exit(1)
    
    print("\n✅ EnvironmentSettingsSuccessCompleted！")


if __name__ == "__main__":
    main()
