#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISP流水LineDemonstrationEnvironment

这个模块Provide了一个Simplified的、可编程的ISP流水LineEnvironment，
用于Demonstration和教学目的。它模拟了真实相机ISP的CoreFeature，
包括Denoising、Sharpening、White Balance、色调Map等模块。

每个模块都被Design成一个独立的Function，接收Image和可调Parameter，
并OutputHandle后的Image。这种模块化Design使得ISP流水Line
可以灵活Combine和Debug。

作者：基于AdaptiveISP论文Implementation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
import time


class ISPModule:
    """ISP模块基Class"""
    
    def __init__(self, name: str, parameters: Dict[str, Any]):
        self.name = name
        self.parameters = parameters
        self.enabled = True
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """HandleImage"""
        if not self.enabled:
            return image
        return self._apply_filter(image)
    
    def _apply_filter(self, image: np.ndarray) -> np.ndarray:
        """子ClassNeedImplementation的具Volume滤波逻辑"""
        raise NotImplementedError
    
    def get_parameter_info(self) -> Dict[str, Any]:
        """GetParameterInformation"""
        return {
            'name': self.name,
            'parameters': self.parameters,
            'enabled': self.enabled
        }


class ExposureModule(ISPModule):
    """ExposureAdjustment模块"""
    
    def __init__(self, exposure_value: float = 0.0):
        super().__init__("ExposureAdjustment", {'exposure': exposure_value})
    
    def _apply_filter(self, image: np.ndarray) -> np.ndarray:
        exposure = self.parameters['exposure']
        # ExposureAdjustment：乘以2的exposure次方
        adjusted = image * (2 ** exposure)
        return np.clip(adjusted, 0.0, 1.0)


class GammaModule(ISPModule):
    """Gamma Correction模块"""
    
    def __init__(self, gamma: float = 1.0):
        super().__init__("Gamma Correction", {'gamma': gamma})
    
    def _apply_filter(self, image: np.ndarray) -> np.ndarray:
        gamma = self.parameters['gamma']
        # Gamma Correction：幂Function变换
        corrected = np.power(np.clip(image, 0.001, 1.0), gamma)
        return np.clip(corrected, 0.0, 1.0)


class WhiteBalanceModule(ISPModule):
    """White Balance模块"""
    
    def __init__(self, rgb_gains: List[float] = [1.0, 1.0, 1.0]):
        super().__init__("White Balance", {'rgb_gains': rgb_gains})
    
    def _apply_filter(self, image: np.ndarray) -> np.ndarray:
        gains = np.array(self.parameters['rgb_gains']).reshape(1, 1, 3)
        balanced = image * gains
        return np.clip(balanced, 0.0, 1.0)


class DenoiseModule(ISPModule):
    """Denoising模块"""
    
    def __init__(self, strength: float = 0.5):
        super().__init__("Denoising", {'strength': strength})
    
    def _apply_filter(self, image: np.ndarray) -> np.ndarray:
        strength = self.parameters['strength']
        if strength <= 0:
            return image
        
        # Transform为uint8进RowDenoising
        img_uint8 = (image * 255).astype(np.uint8)
        
        # Use双边滤波Denoising
        d = int(9 * strength)  # 滤波直径
        sigma_color = 75 * strength
        sigma_space = 75 * strength
        
        denoised = cv2.bilateralFilter(img_uint8, d, sigma_color, sigma_space)
        
        return denoised.astype(np.float32) / 255.0


class SharpenModule(ISPModule):
    """Sharpening模块"""
    
    def __init__(self, strength: float = 0.5):
        super().__init__("Sharpening", {'strength': strength})
    
    def _apply_filter(self, image: np.ndarray) -> np.ndarray:
        strength = self.parameters['strength']
        if strength <= 0:
            return image
        
        # CreateSharpening核
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]]) * strength + (1 - strength) * np.eye(3)
        
        # ApplicationSharpening
        sharpened = cv2.filter2D(image, -1, kernel)
        return np.clip(sharpened, 0.0, 1.0)


class ContrastModule(ISPModule):
    """Comparison度Adjustment模块"""
    
    def __init__(self, contrast: float = 0.0):
        super().__init__("Comparison度Adjustment", {'contrast': contrast})
    
    def _apply_filter(self, image: np.ndarray) -> np.ndarray:
        contrast = self.parameters['contrast']
        # Comparison度Adjustment：Line性变换
        adjusted = image * (1 + contrast)
        return np.clip(adjusted, 0.0, 1.0)


class SaturationModule(ISPModule):
    """SaturationAdjustment模块"""
    
    def __init__(self, saturation: float = 1.0):
        super().__init__("SaturationAdjustment", {'saturation': saturation})
    
    def _apply_filter(self, image: np.ndarray) -> np.ndarray:
        saturation = self.parameters['saturation']
        
        # Transform到HSVSpace
        hsv = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        hsv = hsv.astype(np.float32)
        
        # AdjustmentSaturation
        hsv[:, :, 1] *= saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        # Transform回RGB
        rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return rgb.astype(np.float32) / 255.0


class ToneMappingModule(ISPModule):
    """色调Map模块"""
    
    def __init__(self, tone_curve: List[float] = None):
        if tone_curve is None:
            tone_curve = [0.0, 0.25, 0.5, 0.75, 1.0]  # DefaultLine性曲Line
        super().__init__("色调Map", {'tone_curve': tone_curve})
    
    def _apply_filter(self, image: np.ndarray) -> np.ndarray:
        curve = self.parameters['tone_curve']
        
        # CreateFindTable
        x = np.linspace(0, 1, len(curve))
        y = np.array(curve)
        
        # 插ValueCreateComplete的FindTable
        lut = np.interp(np.linspace(0, 1, 256), x, y)
        
        # Application色调Map
        mapped = lut[(image * 255).astype(np.uint8)]
        return mapped.astype(np.float32) / 255.0


class ColorCorrectionModule(ISPModule):
    """颜色校正模块"""
    
    def __init__(self, ccm_matrix: np.ndarray = None):
        if ccm_matrix is None:
            ccm_matrix = np.eye(3)  # Default单位矩阵（无校正）
        super().__init__("颜色校正", {'ccm_matrix': ccm_matrix})
    
    def _apply_filter(self, image: np.ndarray) -> np.ndarray:
        ccm = self.parameters['ccm_matrix']
        
        # Application颜色校正矩阵
        corrected = np.dot(image.reshape(-1, 3), ccm.T).reshape(image.shape)
        return np.clip(corrected, 0.0, 1.0)


class ISPPipeline:
    """ISP流水LineClass"""
    
    def __init__(self):
        self.modules: List[ISPModule] = []
        self.processing_times: Dict[str, float] = {}
    
    def add_module(self, module: ISPModule) -> None:
        """AddISP模块"""
        self.modules.append(module)
    
    def remove_module(self, module_name: str) -> None:
        """移除ISP模块"""
        self.modules = [m for m in self.modules if m.name != module_name]
    
    def enable_module(self, module_name: str) -> None:
        """启用模块"""
        for module in self.modules:
            if module.name == module_name:
                module.enabled = True
                break
    
    def disable_module(self, module_name: str) -> None:
        """禁用模块"""
        for module in self.modules:
            if module.name == module_name:
                module.enabled = False
                break
    
    def process_image(self, image: np.ndarray, measure_time: bool = True) -> np.ndarray:
        """HandleImage"""
        result = image.copy()
        self.processing_times.clear()
        
        for module in self.modules:
            if measure_time:
                start_time = time.time()
            
            result = module.process(result)
            
            if measure_time:
                elapsed = time.time() - start_time
                self.processing_times[module.name] = elapsed
        
        return result
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get流水LineInformation"""
        return {
            'modules': [m.get_parameter_info() for m in self.modules],
            'processing_times': self.processing_times,
            'total_modules': len(self.modules),
            'enabled_modules': len([m for m in self.modules if m.enabled])
        }
    
    def visualize_pipeline(self, save_path: str = None) -> None:
        """Visualization流水Line"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        y_pos = 0
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 
                 'lightpink', 'lightgray', 'lightcyan', 'lightsteelblue']
        
        for i, module in enumerate(self.modules):
            color = colors[i % len(colors)]
            if not module.enabled:
                color = 'lightgray'
            
            # 绘制模块框
            rect = plt.Rectangle((0, y_pos), 10, 1, facecolor=color, 
                               edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Add模块名称
            ax.text(5, y_pos + 0.5, module.name, ha='center', va='center', 
                   fontsize=10, weight='bold')
            
            # Add箭头
            if i < len(self.modules) - 1:
                ax.arrow(5, y_pos, 0, -1.2, head_width=0.3, head_length=0.1, 
                        fc='black', ec='black')
            
            y_pos -= 1.5
        
        ax.set_xlim(-1, 11)
        ax.set_ylim(y_pos - 1, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('ISP流水LineStructure', fontsize=14, weight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_baseline_pipeline() -> ISPPipeline:
    """CreateTraditionalISP流水Line（固定Parameter）"""
    pipeline = ISPPipeline()
    
    # AddTraditionalISP模块，Use固定Parameter
    pipeline.add_module(ExposureModule(exposure_value=0.0))
    pipeline.add_module(WhiteBalanceModule(rgb_gains=[1.0, 1.0, 1.0]))
    pipeline.add_module(GammaModule(gamma=2.2))
    pipeline.add_module(DenoiseModule(strength=0.3))
    pipeline.add_module(SharpenModule(strength=0.4))
    pipeline.add_module(ContrastModule(contrast=0.1))
    pipeline.add_module(SaturationModule(saturation=1.1))
    
    return pipeline


def create_adaptive_pipeline() -> ISPPipeline:
    """CreateAdaptiveISP流水Line（可调Parameter）"""
    pipeline = ISPPipeline()
    
    # AddAll可能的ISP模块，Parameter将在Run时动态Adjustment
    pipeline.add_module(ExposureModule(exposure_value=0.0))
    pipeline.add_module(GammaModule(gamma=1.0))
    pipeline.add_module(WhiteBalanceModule(rgb_gains=[1.0, 1.0, 1.0]))
    pipeline.add_module(SharpenModule(strength=0.0))
    pipeline.add_module(DenoiseModule(strength=0.0))
    pipeline.add_module(ToneMappingModule())
    pipeline.add_module(ContrastModule(contrast=0.0))
    pipeline.add_module(SaturationModule(saturation=1.0))
    pipeline.add_module(ColorCorrectionModule())
    
    return pipeline


def demo_isp_pipeline():
    """DemonstrationISP流水LineFeature"""
    print("=" * 60)
    print("ISP流水LineDemonstration")
    print("=" * 60)
    
    # CreateTestImage
    test_image = create_test_image()
    
    # CreateTraditionalISP流水Line
    print("\n1. CreateTraditionalISP流水Line...")
    baseline_pipeline = create_baseline_pipeline()
    baseline_pipeline.visualize_pipeline()
    
    # HandleImage
    print("\n2. HandleTestImage...")
    baseline_result = baseline_pipeline.process_image(test_image)
    
    # DisplayHandleTime和Information
    pipeline_info = baseline_pipeline.get_pipeline_info()
    print(f"\nTraditionalISP流水LineInformation:")
    print(f"- 模块Number: {pipeline_info['total_modules']}")
    print(f"- 启用模块: {pipeline_info['enabled_modules']}")
    print(f"- HandleTime:")
    total_time = sum(pipeline_info['processing_times'].values())
    for module, time_taken in pipeline_info['processing_times'].items():
        print(f"  * {module}: {time_taken:.4f}s")
    print(f"  * 总计: {total_time:.4f}s")
    
    # CreateAdaptiveISP流水Line
    print("\n3. CreateAdaptiveISP流水Line...")
    adaptive_pipeline = create_adaptive_pipeline()
    
    # 动态AdjustmentParameter（模拟AI调优）
    print("\n4. 动态AdjustmentParameter...")
    adaptive_pipeline.modules[0].parameters['exposure'] = 0.3  # AddExposure
    adaptive_pipeline.modules[1].parameters['gamma'] = 1.8    # Adjustmentgamma
    adaptive_pipeline.modules[4].parameters['strength'] = 0.6  # EnhancementDenoising
    adaptive_pipeline.modules[5].parameters['strength'] = 0.7  # EnhancementSharpening
    
    # HandleImage
    adaptive_result = adaptive_pipeline.process_image(test_image)
    
    # DisplayResultComparison
    print("\n5. ResultComparison...")
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(test_image)
    plt.title('原始Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(baseline_result)
    plt.title('TraditionalISP')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(adaptive_result)
    plt.title('AdaptiveISP')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('isp_pipeline_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nDemonstrationCompleted！")
    print("这个Demonstration展示了ISP流水Line的模块化Design和Parameter调优Process。")


def create_test_image() -> np.ndarray:
    """CreateTestImage"""
    # Create一个Contains各种特征的TestImage
    height, width = 400, 400
    image = np.zeros((height, width, 3), dtype=np.float32)
    
    # Add渐变背景
    for i in range(height):
        for j in range(width):
            image[i, j] = [i/height, j/width, (i+j)/(height+width)]
    
    # Add一些几何形状
    cv2.rectangle(image, (50, 50), (150, 150), (1, 0, 0), -1)
    cv2.circle(image, (300, 100), 50, (0, 1, 0), -1)
    cv2.ellipse(image, (200, 300), (80, 40), 45, 0, 360, (0, 0, 1), -1)
    
    # Add一些Noise
    noise = np.random.normal(0, 0.05, image.shape)
    image = np.clip(image + noise, 0, 1)
    
    return image


if __name__ == "__main__":
    demo_isp_pipeline()
