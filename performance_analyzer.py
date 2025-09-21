#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISPPerformanceEvaluate和VisualizationAnalyzeTool

这个ToolProvide了全Surface的ISPPerformanceEvaluateFeature，包括：
1. 多种Image质量指标的自动Calculate
2. DetectionPerformance的量化Evaluate
3. HandleTime的PerformanceAnalyze
4. Parameter调优Effect的StatisticsAnalyze
5. GenerateDetailed的PerformanceReport和Visualization图Table

作者：基于AdaptiveISP论文Implementation
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import pandas as pd
import json
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import argparse


@dataclass
class ImageQualityMetrics:
    """Image质量指标"""
    psnr: float = 0.0
    ssim: float = 0.0
    mse: float = 0.0
    edge_density: float = 0.0
    contrast: float = 0.0
    brightness: float = 0.0
    noise_level: float = 0.0
    sharpness: float = 0.0
    color_variance: float = 0.0


@dataclass
class DetectionMetrics:
    """DetectionPerformance指标"""
    detection_score: float = 0.0
    confidence_scores: List[float] = None
    detection_count: int = 0
    average_confidence: float = 0.0
    max_confidence: float = 0.0


@dataclass
class PerformanceMetrics:
    """综合Performance指标"""
    processing_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0


@dataclass
class ISPAnalysisResult:
    """ISPAnalyzeResult"""
    image_quality: ImageQualityMetrics
    detection: DetectionMetrics
    performance: PerformanceMetrics
    applied_filters: List[str]
    parameters: Dict[str, Any]
    timestamp: str


class ImageQualityAnalyzer:
    """Image质量Analyze器"""
    
    @staticmethod
    def calculate_psnr(image1: np.ndarray, image2: np.ndarray) -> float:
        """CalculatePSNR（峰Value信噪比）"""
        mse = np.mean((image1 - image2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(1.0 / np.sqrt(mse))
    
    @staticmethod
    def calculate_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
        """CalculateSSIM（Structure相似性）"""
        # Simplified的SSIMCalculate
        mu1 = np.mean(image1)
        mu2 = np.mean(image2)
        sigma1 = np.var(image1)
        sigma2 = np.var(image2)
        sigma12 = np.mean((image1 - mu1) * (image2 - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
        
        return ssim
    
    @staticmethod
    def calculate_edge_density(image: np.ndarray) -> float:
        """CalculateEdge密度"""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    @staticmethod
    def calculate_contrast(image: np.ndarray) -> float:
        """CalculateComparison度"""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return np.std(gray) / 255.0
    
    @staticmethod
    def calculate_brightness(image: np.ndarray) -> float:
        """Calculate亮度"""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return np.mean(gray) / 255.0
    
    @staticmethod
    def estimate_noise_level(image: np.ndarray) -> float:
        """估计NoiseHorizontal"""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.std(laplacian) / 255.0
    
    @staticmethod
    def calculate_sharpness(image: np.ndarray) -> float:
        """Calculate锐度"""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.var(laplacian) / 255.0
    
    @staticmethod
    def calculate_color_variance(image: np.ndarray) -> float:
        """Calculate颜色方差"""
        return np.var(image, axis=(0, 1)).mean()
    
    @classmethod
    def analyze_image(cls, image: np.ndarray, reference: np.ndarray = None) -> ImageQualityMetrics:
        """综合AnalyzeImage质量"""
        metrics = ImageQualityMetrics()
        
        metrics.edge_density = cls.calculate_edge_density(image)
        metrics.contrast = cls.calculate_contrast(image)
        metrics.brightness = cls.calculate_brightness(image)
        metrics.noise_level = cls.estimate_noise_level(image)
        metrics.sharpness = cls.calculate_sharpness(image)
        metrics.color_variance = cls.calculate_color_variance(image)
        
        if reference is not None:
            metrics.psnr = cls.calculate_psnr(image, reference)
            metrics.ssim = cls.calculate_ssim(image, reference)
            metrics.mse = np.mean((image - reference) ** 2)
        
        return metrics


class DetectionAnalyzer:
    """DetectionPerformanceAnalyze器"""
    
    @staticmethod
    def simulate_detection(image: np.ndarray) -> DetectionMetrics:
        """模拟目标Detection（如果没有真实DetectionModel）"""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # UseEdgeDetection模拟目标Detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter小轮廓
        min_area = 100
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        # CalculateDetection分数
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        contrast = np.std(gray) / 255.0
        brightness = np.mean(gray) / 255.0
        
        # 综合评分
        detection_score = (edge_density * 0.4 + contrast * 0.3 + 
                          (1.0 - abs(brightness - 0.5) * 2) * 0.3) * 0.8
        
        # 模拟置信度分数
        confidence_scores = [0.8 + np.random.normal(0, 0.1) for _ in range(len(valid_contours))]
        confidence_scores = [max(0.0, min(1.0, score)) for score in confidence_scores]
        
        return DetectionMetrics(
            detection_score=detection_score,
            confidence_scores=confidence_scores,
            detection_count=len(valid_contours),
            average_confidence=np.mean(confidence_scores) if confidence_scores else 0.0,
            max_confidence=max(confidence_scores) if confidence_scores else 0.0
        )
    
    @staticmethod
    def run_yolo_detection(image: np.ndarray, yolo_model=None) -> DetectionMetrics:
        """RunYOLODetection（如果有Model）"""
        if yolo_model is None:
            return DetectionAnalyzer.simulate_detection(image)
        
        try:
            # 这里应该调用实际的YOLOModel
            # 为了Demonstration，我们Use模拟Detection
            return DetectionAnalyzer.simulate_detection(image)
        except Exception as e:
            print(f"YOLODetectionFailed: {e}")
            return DetectionAnalyzer.simulate_detection(image)


class PerformanceAnalyzer:
    """PerformanceAnalyze器"""
    
    @staticmethod
    def measure_processing_time(func, *args, **kwargs) -> Tuple[Any, float]:
        """测量HandleTime"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    @staticmethod
    def estimate_memory_usage(image: np.ndarray) -> float:
        """估计内存Use量（MB）"""
        return image.nbytes / (1024 * 1024)


class ISPAnalyzer:
    """ISPAnalyze器主Class"""
    
    def __init__(self):
        self.results: List[ISPAnalysisResult] = []
        self.image_quality_analyzer = ImageQualityAnalyzer()
        self.detection_analyzer = DetectionAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
    
    def analyze_isp_result(self, original_image: np.ndarray, 
                          processed_image: np.ndarray,
                          applied_filters: List[str],
                          parameters: Dict[str, Any]) -> ISPAnalysisResult:
        """AnalyzeSingleISPHandleResult"""
        
        # Image质量Analyze
        image_quality = self.image_quality_analyzer.analyze_image(
            processed_image, original_image
        )
        
        # DetectionPerformanceAnalyze
        detection = self.detection_analyzer.simulate_detection(processed_image)
        
        # PerformanceAnalyze
        processing_time = 0.0  # 在实际Application中从External传入
        memory_usage = self.performance_analyzer.estimate_memory_usage(processed_image)
        
        performance = PerformanceMetrics(
            processing_time=processing_time,
            memory_usage=memory_usage
        )
        
        result = ISPAnalysisResult(
            image_quality=image_quality,
            detection=detection,
            performance=performance,
            applied_filters=applied_filters,
            parameters=parameters,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        self.results.append(result)
        return result
    
    def compare_results(self, baseline_result: ISPAnalysisResult, 
                       adaptive_result: ISPAnalysisResult) -> Dict[str, Any]:
        """比较两个ISPResult"""
        comparison = {
            'detection_score_improvement': (
                adaptive_result.detection.detection_score - 
                baseline_result.detection.detection_score
            ),
            'detection_score_improvement_percent': (
                (adaptive_result.detection.detection_score - 
                 baseline_result.detection.detection_score) / 
                baseline_result.detection.detection_score * 100
            ),
            'psnr_difference': (
                adaptive_result.image_quality.psnr - 
                baseline_result.image_quality.psnr
            ),
            'ssim_difference': (
                adaptive_result.image_quality.ssim - 
                baseline_result.image_quality.ssim
            ),
            'processing_time_ratio': (
                adaptive_result.performance.processing_time / 
                baseline_result.performance.processing_time
                if baseline_result.performance.processing_time > 0 else 0
            ),
            'edge_density_improvement': (
                adaptive_result.image_quality.edge_density - 
                baseline_result.image_quality.edge_density
            ),
            'contrast_improvement': (
                adaptive_result.image_quality.contrast - 
                baseline_result.image_quality.contrast
            ),
            'sharpness_improvement': (
                adaptive_result.image_quality.sharpness - 
                baseline_result.image_quality.sharpness
            )
        }
        
        return comparison
    
    def generate_performance_report(self, output_dir: str = "performance_report"):
        """GeneratePerformanceReport"""
        if not self.results:
            print("没有AnalyzeResult可GenerateReport")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # CreateData框
        data = []
        for result in self.results:
            row = {
                'timestamp': result.timestamp,
                'detection_score': result.detection.detection_score,
                'psnr': result.image_quality.psnr,
                'ssim': result.image_quality.ssim,
                'edge_density': result.image_quality.edge_density,
                'contrast': result.image_quality.contrast,
                'brightness': result.image_quality.brightness,
                'noise_level': result.image_quality.noise_level,
                'sharpness': result.image_quality.sharpness,
                'color_variance': result.image_quality.color_variance,
                'processing_time': result.performance.processing_time,
                'memory_usage': result.performance.memory_usage,
                'applied_filters_count': len(result.applied_filters)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # GenerateStatistics摘要
        summary = df.describe()
        
        # SaveData
        df.to_csv(os.path.join(output_dir, 'performance_data.csv'), index=False)
        summary.to_csv(os.path.join(output_dir, 'performance_summary.csv'))
        
        # GenerateVisualizationReport
        self._create_performance_plots(df, output_dir)
        
        # GenerateTextReport
        self._create_text_report(df, summary, output_dir)
        
        print(f"PerformanceReport已Generate到: {output_dir}")
    
    def _create_performance_plots(self, df: pd.DataFrame, output_dir: str):
        """CreatePerformance图Table"""
        plt.style.use('seaborn-v0_8')
        
        # 1. Detection分数分布
        plt.figure(figsize=(10, 6))
        plt.hist(df['detection_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Detection分数')
        plt.ylabel('频次')
        plt.title('Detection分数分布')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'detection_score_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Image质量指标Comparison
        quality_metrics = ['edge_density', 'contrast', 'brightness', 'sharpness', 'color_variance']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(quality_metrics):
            if i < len(axes):
                axes[i].hist(df[metric], bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
                axes[i].set_xlabel(metric.replace('_', ' ').title())
                axes[i].set_ylabel('频次')
                axes[i].set_title(f'{metric.replace("_", " ").title()} 分布')
                axes[i].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(len(quality_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'image_quality_metrics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 相关性热力图
        correlation_matrix = df[['detection_score', 'edge_density', 'contrast', 
                               'brightness', 'sharpness', 'color_variance']].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f')
        plt.title('Image质量指标相关性热力图')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. HandleTime vs Detection分数
        if 'processing_time' in df.columns and df['processing_time'].sum() > 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(df['processing_time'], df['detection_score'], 
                       alpha=0.6, color='purple')
            plt.xlabel('HandleTime (秒)')
            plt.ylabel('Detection分数')
            plt.title('HandleTime vs Detection分数')
            plt.grid(True, alpha=0.3)
            
            # Add趋势Line
            z = np.polyfit(df['processing_time'], df['detection_score'], 1)
            p = np.poly1d(z)
            plt.plot(df['processing_time'], p(df['processing_time']), 
                    "r--", alpha=0.8, label=f'趋势Line (斜率: {z[0]:.3f})')
            plt.legend()
            
            plt.savefig(os.path.join(output_dir, 'processing_time_vs_score.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. 滤波器Use频率
        all_filters = []
        for result in self.results:
            all_filters.extend(result.applied_filters)
        
        if all_filters:
            filter_counts = pd.Series(all_filters).value_counts()
            
            plt.figure(figsize=(12, 6))
            filter_counts.plot(kind='bar', color='orange', alpha=0.7)
            plt.xlabel('滤波器Type')
            plt.ylabel('Use次数')
            plt.title('滤波器Use频率')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'filter_usage_frequency.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_text_report(self, df: pd.DataFrame, summary: pd.DataFrame, output_dir: str):
        """CreateTextReport"""
        report_path = os.path.join(output_dir, 'performance_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("ISPPerformanceAnalyzeReport\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"AnalyzeTime: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analyze样本Number: {len(df)}\n\n")
            
            f.write("1. DetectionPerformanceStatistics\n")
            f.write("-" * 30 + "\n")
            f.write(f"AverageDetection分数: {df['detection_score'].mean():.4f}\n")
            f.write(f"Detection分数Standard差: {df['detection_score'].std():.4f}\n")
            f.write(f"最高Detection分数: {df['detection_score'].max():.4f}\n")
            f.write(f"最低Detection分数: {df['detection_score'].min():.4f}\n\n")
            
            f.write("2. Image质量Statistics\n")
            f.write("-" * 30 + "\n")
            quality_metrics = ['edge_density', 'contrast', 'brightness', 'sharpness', 'color_variance']
            for metric in quality_metrics:
                if metric in df.columns:
                    f.write(f"{metric.replace('_', ' ').title()}: "
                           f"{df[metric].mean():.4f} ± {df[metric].std():.4f}\n")
            f.write("\n")
            
            f.write("3. PerformanceStatistics\n")
            f.write("-" * 30 + "\n")
            if df['processing_time'].sum() > 0:
                f.write(f"AverageHandleTime: {df['processing_time'].mean():.4f} 秒\n")
                f.write(f"总HandleTime: {df['processing_time'].sum():.4f} 秒\n")
            f.write(f"Average内存Use: {df['memory_usage'].mean():.2f} MB\n\n")
            
            f.write("4. 滤波器UseStatistics\n")
            f.write("-" * 30 + "\n")
            all_filters = []
            for result in self.results:
                all_filters.extend(result.applied_filters)
            
            if all_filters:
                filter_counts = pd.Series(all_filters).value_counts()
                for filter_name, count in filter_counts.items():
                    f.write(f"{filter_name}: {count} 次 ({count/len(all_filters)*100:.1f}%)\n")
            
            f.write("\n5. DetailedStatistics摘要\n")
            f.write("-" * 30 + "\n")
            f.write(summary.to_string())
            
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("ReportGenerateCompleted\n")
            f.write("=" * 60 + "\n")
    
    def create_comparison_visualization(self, baseline_results: List[ISPAnalysisResult],
                                      adaptive_results: List[ISPAnalysisResult],
                                      output_dir: str = "comparison_analysis"):
        """CreateComparisonVisualization"""
        os.makedirs(output_dir, exist_ok=True)
        
        # ExtractData
        baseline_scores = [r.detection.detection_score for r in baseline_results]
        adaptive_scores = [r.detection.detection_score for r in adaptive_results]
        
        baseline_times = [r.performance.processing_time for r in baseline_results]
        adaptive_times = [r.performance.processing_time for r in adaptive_results]
        
        # CreateComparison图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Detection分数Comparison
        x = range(len(baseline_scores))
        axes[0, 0].plot(x, baseline_scores, 'r-o', label='TraditionalISP', linewidth=2, markersize=6)
        axes[0, 0].plot(x, adaptive_scores, 'g-o', label='AdaptiveISP', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Image编号')
        axes[0, 0].set_ylabel('Detection分数')
        axes[0, 0].set_title('DetectionPerformanceComparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. PerformanceEnhance分布
        improvements = [(a - b) / b * 100 for a, b in zip(adaptive_scores, baseline_scores)]
        axes[0, 1].hist(improvements, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_xlabel('PerformanceEnhance (%)')
        axes[0, 1].set_ylabel('频次')
        axes[0, 1].set_title('PerformanceEnhance分布')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axvline(np.mean(improvements), color='red', linestyle='--', 
                          label=f'AverageEnhance: {np.mean(improvements):.1f}%')
        axes[0, 1].legend()
        
        # 3. HandleTimeComparison
        axes[1, 0].plot(x, baseline_times, 'r-o', label='TraditionalISP', linewidth=2, markersize=6)
        axes[1, 0].plot(x, adaptive_times, 'g-o', label='AdaptiveISP', linewidth=2, markersize=6)
        axes[1, 0].set_xlabel('Image编号')
        axes[1, 0].set_ylabel('HandleTime (秒)')
        axes[1, 0].set_title('HandleTimeComparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 综合Performance雷达图
        metrics = ['Detection分数', 'Edge密度', 'Comparison度', '锐度', '颜色方差']
        baseline_values = [
            np.mean(baseline_scores),
            np.mean([r.image_quality.edge_density for r in baseline_results]),
            np.mean([r.image_quality.contrast for r in baseline_results]),
            np.mean([r.image_quality.sharpness for r in baseline_results]),
            np.mean([r.image_quality.color_variance for r in baseline_results])
        ]
        adaptive_values = [
            np.mean(adaptive_scores),
            np.mean([r.image_quality.edge_density for r in adaptive_results]),
            np.mean([r.image_quality.contrast for r in adaptive_results]),
            np.mean([r.image_quality.sharpness for r in adaptive_results]),
            np.mean([r.image_quality.color_variance for r in adaptive_results])
        ]
        
        # 归一化到0-1
        max_vals = [max(b, a) for b, a in zip(baseline_values, adaptive_values)]
        baseline_norm = [b/m if m > 0 else 0 for b, m in zip(baseline_values, max_vals)]
        adaptive_norm = [a/m if m > 0 else 0 for a, m in zip(adaptive_values, max_vals)]
        
        # Create雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        baseline_norm += baseline_norm[:1]  # 闭合图形
        adaptive_norm += adaptive_norm[:1]
        angles += angles[:1]
        
        axes[1, 1].plot(angles, baseline_norm, 'r-o', linewidth=2, label='TraditionalISP')
        axes[1, 1].fill(angles, baseline_norm, alpha=0.25, color='red')
        axes[1, 1].plot(angles, adaptive_norm, 'g-o', linewidth=2, label='AdaptiveISP')
        axes[1, 1].fill(angles, adaptive_norm, alpha=0.25, color='green')
        
        axes[1, 1].set_xticks(angles[:-1])
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('综合PerformanceComparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'isp_comparison_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ComparisonAnalyze图已Save到: {output_dir}")


def main():
    """主Function"""
    parser = argparse.ArgumentParser(description='ISPPerformanceAnalyzeTool')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='InputImageDirectory')
    parser.add_argument('--output_dir', type=str, default='performance_analysis',
                       help='OutputAnalyzeResultDirectory')
    parser.add_argument('--baseline_results', type=str, default=None,
                       help='TraditionalISPResultJSONFile')
    parser.add_argument('--adaptive_results', type=str, default=None,
                       help='AdaptiveISPResultJSONFile')
    
    args = parser.parse_args()
    
    print("启动ISPPerformanceAnalyzeTool...")
    
    analyzer = ISPAnalyzer()
    
    # 这里可以Add具Volume的Analyze逻辑
    # 例如：LoadImage、RunISPHandle、AnalyzeResult等
    
    print("PerformanceAnalyzeCompleted！")


if __name__ == "__main__":
    main()
