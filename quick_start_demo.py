#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chapter 7 Quick Start Demo Script

This script provides a quick experience of the complete AI-driven ISP tuning process,
including:
1. Create test images
2. Run traditional ISP processing
3. Run adaptive ISP processing
4. Compare and analyze results
5. Generate visualization reports

Usage:
python quick_start_demo.py
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import sys

# Add current directory to path
sys.path.append('.')

from isp_pipeline_demo import (
    ISPPipeline, ExposureModule, GammaModule, WhiteBalanceModule,
    DenoiseModule, SharpenModule, ContrastModule, SaturationModule,
    create_baseline_pipeline, create_adaptive_pipeline
)


def create_test_scenarios():
    """Create images for multiple test scenarios using real test data"""
    scenarios = {}
    
    # Use real test images from test_data folder
    test_images = ['876.png', '920.png', '926.png', '930.png']
    scenario_names = ['Normal Lighting', 'Low Light', 'High Contrast', 'Noisy Scene']
    
    for i, (img_name, scenario_name) in enumerate(zip(test_images, scenario_names)):
        img_path = f'test_data/{img_name}'
        if os.path.exists(img_path):
            # Load real image
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0
            scenarios[scenario_name] = image
        else:
            # Fallback to synthetic image if real image not found
            if scenario_name == 'Normal Lighting':
                image = create_normal_light_scene()
            elif scenario_name == 'Low Light':
                image = create_low_light_scene()
            elif scenario_name == 'High Contrast':
                image = create_high_contrast_scene()
            else:
                image = create_noisy_scene()
            scenarios[scenario_name] = image
    
    return scenarios


def create_normal_light_scene():
    """CreateNormal LightingTestScene"""
    height, width = 400, 400
    image = np.ones((height, width, 3), dtype=np.float32) * 0.5
    
    # Add some geometric shapes
    cv2.rectangle(image, (50, 50), (150, 150), (0.8, 0.2, 0.2), -1)  # Red rectangle
    cv2.circle(image, (300, 100), 50, (0.2, 0.8, 0.2), -1)  # Green circle
    cv2.ellipse(image, (200, 300), (80, 40), 45, 0, 360, (0.2, 0.2, 0.8), -1)  # Blue ellipse
    
    # Add some text regions
    cv2.rectangle(image, (20, 200), (120, 250), (0.9, 0.9, 0.9), -1)
    cv2.rectangle(image, (280, 200), (380, 250), (0.1, 0.1, 0.1), -1)
    
    return image


def create_low_light_scene():
    """CreateLow LightTestScene"""
    image = create_normal_light_scene()
    # Decrease overall brightness
    image = image * 0.3
    # Add some noise
    noise = np.random.normal(0, 0.02, image.shape)
    image = np.clip(image + noise, 0, 1)
    return image


def create_high_contrast_scene():
    """Create high contrast test scene"""
    height, width = 400, 400
    image = np.zeros((height, width, 3), dtype=np.float32)
    
    # Create高Comparison度的棋盘Mode
    block_size = 40
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            if (i // block_size + j // block_size) % 2 == 0:
                image[i:i+block_size, j:j+block_size] = [0.9, 0.9, 0.9]
            else:
                image[i:i+block_size, j:j+block_size] = [0.1, 0.1, 0.1]
    
    # Add一些彩色Element
    cv2.circle(image, (100, 100), 30, (1, 0, 0), -1)
    cv2.circle(image, (300, 100), 30, (0, 1, 0), -1)
    cv2.circle(image, (200, 300), 30, (0, 0, 1), -1)
    
    return image


def create_noisy_scene():
    """CreateNoiseTestScene"""
    image = create_normal_light_scene()
    # Add大量Noise
    noise = np.random.normal(0, 0.1, image.shape)
    image = np.clip(image + noise, 0, 1)
    return image


def calculate_detection_score(image):
    """CalculateDetection分数（SimplifiedVersion）"""
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    
    # Edge密度
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # Comparison度
    contrast = np.std(gray) / 255.0
    
    # 亮度分布
    brightness = np.mean(gray) / 255.0
    brightness_score = 1.0 - abs(brightness - 0.5) * 2
    
    # 综合评分
    detection_score = (edge_density * 0.4 + contrast * 0.3 + brightness_score * 0.3) * 0.8
    
    return min(max(detection_score, 0.0), 1.0)


def simulate_adaptive_isp(image):
    """模拟AdaptiveISPHandle"""
    # AnalyzeImage特征
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray) / 255.0
    contrast = np.std(gray) / 255.0
    
    # 根据特征AdjustmentParameter
    if brightness < 0.3:  # Low Light
        exposure = 0.5
        gamma = 1.8
        denoise = 0.7
        sharpen = 0.6
        applied_filters = ["ExposureEnhancement", "Gamma Correction", "Denoising", "Sharpening"]
    elif brightness > 0.7:  # 高光
        exposure = -0.3
        gamma = 2.5
        denoise = 0.3
        sharpen = 0.4
        applied_filters = ["ExposureDecrease", "Gamma Correction", "LightDenoising", "LightSharpening"]
    else:  # Normal Lighting
        exposure = 0.1
        gamma = 2.2
        denoise = 0.4
        sharpen = 0.5
        applied_filters = ["LightExposureAdjustment", "Gamma Correction", "Denoising", "Sharpening"]
    
    # 根据Comparison度Adjustment
    if contrast < 0.15:
        contrast_adj = 0.2
        saturation = 1.3
        applied_filters.extend(["Comparison度Enhancement", "SaturationEnhancement"])
    else:
        contrast_adj = 0.0
        saturation = 1.1
        applied_filters.append("SaturationFine-tuning")
    
    # CreateAdaptive流水Line
    pipeline = create_adaptive_pipeline()
    
    # ApplicationParameter
    pipeline.modules[0].parameters['exposure'] = exposure
    pipeline.modules[1].parameters['gamma'] = gamma
    pipeline.modules[4].parameters['strength'] = denoise
    pipeline.modules[5].parameters['strength'] = sharpen
    pipeline.modules[6].parameters['contrast'] = contrast_adj
    pipeline.modules[7].parameters['saturation'] = saturation
    
    # HandleImage
    result = pipeline.process_image(image)
    
    return result, applied_filters


def run_demo():
    """RunCompleteDemonstration"""
    print("=" * 60)
    print("Chapter 7: AI-driven ISP Tuning Automation - Quick Demo")
    print("=" * 60)
    
    # CreateOutputDirectory
    output_dir = "quick_start_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # CreateTestScene
    print("\n1. Creating test scenarios...")
    scenarios = create_test_scenarios()
    print(f"   Created {len(scenarios)} test scenarios")
    
    # Create traditional ISP pipeline
    print("\n2. Initializing ISP pipeline...")
    baseline_pipeline = create_baseline_pipeline()
    print("   Traditional ISP pipeline ready")
    
    # Store results
    all_results = {}
    
    # Process each scenario
    print("\n3. Processing test scenarios...")
    for scenario_name, original_image in scenarios.items():
        print(f"\n   Processing scenario: {scenario_name}")
        
        # TraditionalISPHandle
        start_time = time.time()
        baseline_result = baseline_pipeline.process_image(original_image)
        baseline_time = time.time() - start_time
        
        # AdaptiveISPHandle
        start_time = time.time()
        adaptive_result, applied_filters = simulate_adaptive_isp(original_image)
        adaptive_time = time.time() - start_time
        
        # CalculateDetection分数
        baseline_score = calculate_detection_score(baseline_result)
        adaptive_score = calculate_detection_score(adaptive_result)
        
        # CalculatePerformanceEnhance
        improvement = (adaptive_score - baseline_score) / baseline_score * 100
        
        print(f"     Traditional ISP score: {baseline_score:.3f}, Time: {baseline_time:.3f}s")
        print(f"     Adaptive ISP score: {adaptive_score:.3f}, Time: {adaptive_time:.3f}s")
        print(f"     Performance improvement: {improvement:+.1f}%")
        print(f"     Applied filters: {', '.join(applied_filters)}")
        
        # SaveResult
        all_results[scenario_name] = {
            'original': original_image,
            'baseline': baseline_result,
            'adaptive': adaptive_result,
            'baseline_score': baseline_score,
            'adaptive_score': adaptive_score,
            'baseline_time': baseline_time,
            'adaptive_time': adaptive_time,
            'improvement': improvement,
            'applied_filters': applied_filters
        }
    
    # GenerateVisualizationReport
    print("\n4. Generating visualization report...")
    create_comparison_visualization(all_results, output_dir)
    
    # Generate statistics summary
    print("\n5. Generating statistics summary...")
    generate_summary_report(all_results, output_dir)
    
    print(f"\nDemo completed! Results saved to: {output_dir}")
    print("\nMain findings:")
    
    # CalculateAverageEnhance
    improvements = [result['improvement'] for result in all_results.values()]
    avg_improvement = np.mean(improvements)
    print(f"- Average detection performance improvement: {avg_improvement:.1f}%")
    print(f"- Best scenario improvement: {max(improvements):.1f}%")
    print(f"- Worst scenario improvement: {min(improvements):.1f}%")
    
    # Analyze filter usage
    all_filters = []
    for result in all_results.values():
        all_filters.extend(result['applied_filters'])
    
    from collections import Counter
    filter_counts = Counter(all_filters)
    print(f"- Most commonly used filters: {filter_counts.most_common(3)}")
    
    print("\nThis demonstration shows how AI automatically analyzes image features,")
    print("selects optimal ISP parameter combinations, and implements task-oriented image optimization.")


def create_comparison_visualization(all_results, output_dir):
    """CreateComparisonVisualization"""
    n_scenarios = len(all_results)
    fig, axes = plt.subplots(n_scenarios, 3, figsize=(15, 4*n_scenarios))
    
    if n_scenarios == 1:
        axes = axes.reshape(1, -1)
    
    for i, (scenario_name, result) in enumerate(all_results.items()):
        # 原始Image
        axes[i, 0].imshow(result['original'])
        axes[i, 0].set_title(f'{scenario_name}\n原始Image')
        axes[i, 0].axis('off')
        
        # TraditionalISPResult
        axes[i, 1].imshow(result['baseline'])
        axes[i, 1].set_title(f'TraditionalISP\n分数: {result["baseline_score"]:.3f}')
        axes[i, 1].axis('off')
        
        # AdaptiveISPResult
        axes[i, 2].imshow(result['adaptive'])
        axes[i, 2].set_title(f'AdaptiveISP\n分数: {result["adaptive_score"]:.3f} ({result["improvement"]:+.1f}%)')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scenario_comparison.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # CreatePerformanceComparison图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    scenario_names = list(all_results.keys())
    baseline_scores = [all_results[name]['baseline_score'] for name in scenario_names]
    adaptive_scores = [all_results[name]['adaptive_score'] for name in scenario_names]
    improvements = [all_results[name]['improvement'] for name in scenario_names]
    
    # Detection分数Comparison
    x = range(len(scenario_names))
    width = 0.35
    ax1.bar([i - width/2 for i in x], baseline_scores, width, label='TraditionalISP', alpha=0.8)
    ax1.bar([i + width/2 for i in x], adaptive_scores, width, label='AdaptiveISP', alpha=0.8)
    ax1.set_xlabel('TestScene')
    ax1.set_ylabel('Detection分数')
    ax1.set_title('DetectionPerformanceComparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_names, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # PerformanceEnhance
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    ax2.bar(scenario_names, improvements, color=colors, alpha=0.7)
    ax2.set_xlabel('TestScene')
    ax2.set_ylabel('PerformanceEnhance (%)')
    ax2.set_title('PerformanceEnhanceComparison')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()


def generate_summary_report(all_results, output_dir):
    """Generate摘要Report"""
    report_path = os.path.join(output_dir, 'summary_report.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("AI驱动ISP调优DemonstrationReport\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"DemonstrationTime: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"TestSceneNumber: {len(all_results)}\n\n")
        
        f.write("DetailedResult:\n")
        f.write("-" * 40 + "\n")
        
        for scenario_name, result in all_results.items():
            f.write(f"\nScene: {scenario_name}\n")
            f.write(f"  TraditionalISP分数: {result['baseline_score']:.4f}\n")
            f.write(f"  AdaptiveISP分数: {result['adaptive_score']:.4f}\n")
            f.write(f"  PerformanceEnhance: {result['improvement']:+.2f}%\n")
            f.write(f"  HandleTime: {result['baseline_time']:.3f}s -> {result['adaptive_time']:.3f}s\n")
            f.write(f"  Application滤波器: {', '.join(result['applied_filters'])}\n")
        
        # Statistics摘要
        improvements = [result['improvement'] for result in all_results.values()]
        baseline_scores = [result['baseline_score'] for result in all_results.values()]
        adaptive_scores = [result['adaptive_score'] for result in all_results.values()]
        
        f.write(f"\nStatistics摘要:\n")
        f.write("-" * 40 + "\n")
        f.write(f"AveragePerformanceEnhance: {np.mean(improvements):.2f}%\n")
        f.write(f"MaximumPerformanceEnhance: {np.max(improvements):.2f}%\n")
        f.write(f"MinimumPerformanceEnhance: {np.min(improvements):.2f}%\n")
        f.write(f"PerformanceEnhanceStandard差: {np.std(improvements):.2f}%\n")
        f.write(f"AverageTraditionalISP分数: {np.mean(baseline_scores):.4f}\n")
        f.write(f"AverageAdaptiveISP分数: {np.mean(adaptive_scores):.4f}\n")
        
        # 滤波器UseStatistics
        all_filters = []
        for result in all_results.values():
            all_filters.extend(result['applied_filters'])
        
        from collections import Counter
        filter_counts = Counter(all_filters)
        
        f.write(f"\n滤波器UseStatistics:\n")
        f.write("-" * 40 + "\n")
        for filter_name, count in filter_counts.most_common():
            f.write(f"  {filter_name}: {count} 次 ({count/len(all_filters)*100:.1f}%)\n")
        
        f.write(f"\n结论:\n")
        f.write("-" * 40 + "\n")
        f.write("这个Demonstration展示了AI驱动的ISP调优相比Traditional固定ParameterMethod\n")
        f.write("在DetectionPerformance上的显著Enhance。AdaptiveISP能够根据Image特征\n")
        f.write("自动Select最优的HandleParameter，Implementation任务导向的ImageOptimize。\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("ReportGenerateCompleted\n")
        f.write("=" * 60 + "\n")


if __name__ == "__main__":
    run_demo()
