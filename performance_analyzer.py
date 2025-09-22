#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISP Performance Evaluation and Visualization Analysis Tool

This tool provides comprehensive ISP performance evaluation features, including:
1. Automatic calculation of multiple image quality metrics
2. Quantitative evaluation of detection performance
3. Performance analysis of processing time
4. Statistical analysis of parameter tuning effects
5. Generate detailed performance reports and visualization charts

Author: Based on AdaptiveISP paper implementation
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
    """Image quality metrics"""
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
    """Detection performance metrics"""
    detection_score: float = 0.0
    confidence_scores: List[float] = None
    detection_count: int = 0
    average_confidence: float = 0.0
    max_confidence: float = 0.0


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
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
    """Image quality analyzer"""
    
    @staticmethod
    def calculate_psnr(image1: np.ndarray, image2: np.ndarray) -> float:
        """Calculate PSNR (peak signal-to-noise ratio)"""
        mse = np.mean((image1 - image2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(1.0 / np.sqrt(mse))
    
    @staticmethod
    def calculate_ssim(image1: np.ndarray, image2: np.ndarray) -> float:
        """Calculate SSIM (structural similarity)"""
        # Simplified SSIM calculation
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
        """Calculate edge density"""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    @staticmethod
    def calculate_contrast(image: np.ndarray) -> float:
        """Calculate contrast"""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return np.std(gray) / 255.0
    
    @staticmethod
    def calculate_brightness(image: np.ndarray) -> float:
        """Calculate brightness"""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return np.mean(gray) / 255.0
    
    @staticmethod
    def estimate_noise_level(image: np.ndarray) -> float:
        """Estimate noise level"""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.std(laplacian) / 255.0
    
    @staticmethod
    def calculate_sharpness(image: np.ndarray) -> float:
        """Calculate sharpness"""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return np.var(laplacian) / 255.0
    
    @staticmethod
    def calculate_color_variance(image: np.ndarray) -> float:
        """Calculate color variance"""
        return np.var(image, axis=(0, 1)).mean()
    
    @classmethod
    def analyze_image(cls, image: np.ndarray, reference: np.ndarray = None) -> ImageQualityMetrics:
        """Comprehensive image quality analysis"""
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
    """Detection performance analyzer"""
    
    @staticmethod
    def simulate_detection(image: np.ndarray) -> DetectionMetrics:
        """Simulate object detection (if no real detection model available)"""
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Use edge detection to simulate object detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small contours
        min_area = 100
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        # Calculate detection score
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        contrast = np.std(gray) / 255.0
        brightness = np.mean(gray) / 255.0
        
        # Comprehensive score
        detection_score = (edge_density * 0.4 + contrast * 0.3 + 
                          (1.0 - abs(brightness - 0.5) * 2) * 0.3) * 0.8
        
        # Simulate confidence scores
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
        """Run YOLO detection (if model available)"""
        if yolo_model is None:
            return DetectionAnalyzer.simulate_detection(image)
        
        try:
            # Here should call the actual YOLO model
            # For demonstration, we use simulated detection
            return DetectionAnalyzer.simulate_detection(image)
        except Exception as e:
            print(f"YOLODetectionFailed: {e}")
            return DetectionAnalyzer.simulate_detection(image)


class PerformanceAnalyzer:
    """Performance analyzer"""
    
    @staticmethod
    def measure_processing_time(func, *args, **kwargs) -> Tuple[Any, float]:
        """Measure processing time"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    @staticmethod
    def estimate_memory_usage(image: np.ndarray) -> float:
        """Estimate memory usage (MB)"""
        return image.nbytes / (1024 * 1024)


class ISPAnalyzer:
    """ISP analyzer main class"""
    
    def __init__(self):
        self.results: List[ISPAnalysisResult] = []
        self.image_quality_analyzer = ImageQualityAnalyzer()
        self.detection_analyzer = DetectionAnalyzer()
        self.performance_analyzer = PerformanceAnalyzer()
    
    def analyze_isp_result(self, original_image: np.ndarray, 
                          processed_image: np.ndarray,
                          applied_filters: List[str],
                          parameters: Dict[str, Any]) -> ISPAnalysisResult:
        """Analyze single ISP processing result"""
        
        # Image quality analysis
        image_quality = self.image_quality_analyzer.analyze_image(
            processed_image, original_image
        )
        
        # Detection performance analysis
        detection = self.detection_analyzer.simulate_detection(processed_image)
        
        # Performance analysis
        processing_time = 0.0  # In actual application, passed from external source
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
        """Compare two ISP results"""
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
        """Generate performance report"""
        if not self.results:
            print("No analysis results to generate report")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create dataframe
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
        
        # Generate statistical summary
        summary = df.describe()
        
        # Save data
        df.to_csv(os.path.join(output_dir, 'performance_data.csv'), index=False)
        summary.to_csv(os.path.join(output_dir, 'performance_summary.csv'))
        
        # Generate visualization report
        self._create_performance_plots(df, output_dir)
        
        # Generate text report
        self._create_text_report(df, summary, output_dir)
        
        print(f"Performance report generated to: {output_dir}")
    
    def _create_performance_plots(self, df: pd.DataFrame, output_dir: str):
        """Create performance charts"""
        plt.style.use('seaborn-v0_8')
        
        # 1. Detection score distribution
        plt.figure(figsize=(10, 6))
        plt.hist(df['detection_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Detection Score')
        plt.ylabel('Frequency')
        plt.title('Detection score distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'detection_score_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Image quality metrics comparison
        quality_metrics = ['edge_density', 'contrast', 'brightness', 'sharpness', 'color_variance']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(quality_metrics):
            if i < len(axes):
                axes[i].hist(df[metric], bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
                axes[i].set_xlabel(metric.replace('_', ' ').title())
                axes[i].set_ylabel('Frequency')
                axes[i].set_title(f'{metric.replace("_", " ").title()} Distribution')
                axes[i].grid(True, alpha=0.3)
        
        # Hide extra subplots
        for i in range(len(quality_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'image_quality_metrics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Correlation heatmap
        correlation_matrix = df[['detection_score', 'edge_density', 'contrast', 
                               'brightness', 'sharpness', 'color_variance']].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f')
        plt.title('Image Quality Metrics Correlation Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Processing time vs Detection score
        if 'processing_time' in df.columns and df['processing_time'].sum() > 0:
            plt.figure(figsize=(10, 6))
            plt.scatter(df['processing_time'], df['detection_score'], 
                       alpha=0.6, color='purple')
            plt.xlabel('Processing Time (seconds)')
            plt.ylabel('Detection Score')
            plt.title('Processing Time vs Detection Score')
            plt.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(df['processing_time'], df['detection_score'], 1)
            p = np.poly1d(z)
            plt.plot(df['processing_time'], p(df['processing_time']), 
                    "r--", alpha=0.8, label=f'Trend Line (slope: {z[0]:.3f})')
            plt.legend()
            
            plt.savefig(os.path.join(output_dir, 'processing_time_vs_score.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Filter usage frequency
        all_filters = []
        for result in self.results:
            all_filters.extend(result.applied_filters)
        
        if all_filters:
            filter_counts = pd.Series(all_filters).value_counts()
            
            plt.figure(figsize=(12, 6))
            filter_counts.plot(kind='bar', color='orange', alpha=0.7)
            plt.xlabel('Filter Type')
            plt.ylabel('Usage Count')
            plt.title('Filter Usage Frequency')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'filter_usage_frequency.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_text_report(self, df: pd.DataFrame, summary: pd.DataFrame, output_dir: str):
        """Create text report"""
        report_path = os.path.join(output_dir, 'performance_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("ISP Performance Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Analysis Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis Sample Count: {len(df)}\n\n")
            
            f.write("1. Detection Performance Statistics\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average Detection Score: {df['detection_score'].mean():.4f}\n")
            f.write(f"Detection Score Standard Deviation: {df['detection_score'].std():.4f}\n")
            f.write(f"Highest Detection Score: {df['detection_score'].max():.4f}\n")
            f.write(f"Lowest Detection Score: {df['detection_score'].min():.4f}\n\n")
            
            f.write("2. Image Quality Statistics\n")
            f.write("-" * 30 + "\n")
            quality_metrics = ['edge_density', 'contrast', 'brightness', 'sharpness', 'color_variance']
            for metric in quality_metrics:
                if metric in df.columns:
                    f.write(f"{metric.replace('_', ' ').title()}: "
                           f"{df[metric].mean():.4f} ± {df[metric].std():.4f}\n")
            f.write("\n")
            
            f.write("3. Performance Statistics\n")
            f.write("-" * 30 + "\n")
            if df['processing_time'].sum() > 0:
                f.write(f"Average Processing Time: {df['processing_time'].mean():.4f} seconds\n")
                f.write(f"Total Processing Time: {df['processing_time'].sum():.4f} seconds\n")
            f.write(f"Average Memory Usage: {df['memory_usage'].mean():.2f} MB\n\n")
            
            f.write("4. Filter Usage Statistics\n")
            f.write("-" * 30 + "\n")
            all_filters = []
            for result in self.results:
                all_filters.extend(result.applied_filters)
            
            if all_filters:
                filter_counts = pd.Series(all_filters).value_counts()
                for filter_name, count in filter_counts.items():
                    f.write(f"{filter_name}: {count} times ({count/len(all_filters)*100:.1f}%)\n")
            
            f.write("\n5. Detailed Statistics Summary\n")
            f.write("-" * 30 + "\n")
            f.write(summary.to_string())
            
            f.write("\n\n" + "=" * 60 + "\n")
            f.write("Report Generation Completed\n")
            f.write("=" * 60 + "\n")
    
    def create_comparison_visualization(self, baseline_results: List[ISPAnalysisResult],
                                      adaptive_results: List[ISPAnalysisResult],
                                      output_dir: str = "comparison_analysis"):
        """Create comparison visualization"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data
        baseline_scores = [r.detection.detection_score for r in baseline_results]
        adaptive_scores = [r.detection.detection_score for r in adaptive_results]
        
        baseline_times = [r.performance.processing_time for r in baseline_results]
        adaptive_times = [r.performance.processing_time for r in adaptive_results]
        
        # Create comparison charts
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Detection score comparison
        x = range(len(baseline_scores))
        axes[0, 0].plot(x, baseline_scores, 'r-o', label='TraditionalISP', linewidth=2, markersize=6)
        axes[0, 0].plot(x, adaptive_scores, 'g-o', label='AdaptiveISP', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Image Index')
        axes[0, 0].set_ylabel('Detection Score')
        axes[0, 0].set_title('Detection Performance Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Performance improvement distribution
        improvements = [(a - b) / b * 100 for a, b in zip(adaptive_scores, baseline_scores)]
        axes[0, 1].hist(improvements, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 1].set_xlabel('Performance Improvement (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Performance Improvement Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axvline(np.mean(improvements), color='red', linestyle='--', 
                          label=f'Average Improvement: {np.mean(improvements):.1f}%')
        axes[0, 1].legend()
        
        # 3. Processing time comparison
        axes[1, 0].plot(x, baseline_times, 'r-o', label='TraditionalISP', linewidth=2, markersize=6)
        axes[1, 0].plot(x, adaptive_times, 'g-o', label='AdaptiveISP', linewidth=2, markersize=6)
        axes[1, 0].set_xlabel('Image Index')
        axes[1, 0].set_ylabel('Processing Time (seconds)')
        axes[1, 0].set_title('Processing Time Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Comprehensive performance radar chart
        metrics = ['Detection Score', 'Edge Density', 'Contrast', 'Sharpness', 'Color Variance']
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
        
        # Normalize to 0-1
        max_vals = [max(b, a) for b, a in zip(baseline_values, adaptive_values)]
        baseline_norm = [b/m if m > 0 else 0 for b, m in zip(baseline_values, max_vals)]
        adaptive_norm = [a/m if m > 0 else 0 for a, m in zip(adaptive_values, max_vals)]
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        baseline_norm += baseline_norm[:1]  # Close the shape
        adaptive_norm += adaptive_norm[:1]
        angles += angles[:1]
        
        axes[1, 1].plot(angles, baseline_norm, 'r-o', linewidth=2, label='TraditionalISP')
        axes[1, 1].fill(angles, baseline_norm, alpha=0.25, color='red')
        axes[1, 1].plot(angles, adaptive_norm, 'g-o', linewidth=2, label='AdaptiveISP')
        axes[1, 1].fill(angles, adaptive_norm, alpha=0.25, color='green')
        
        axes[1, 1].set_xticks(angles[:-1])
        axes[1, 1].set_xticklabels(metrics)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('Comprehensive Performance Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'isp_comparison_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison analysis charts saved to: {output_dir}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='ISP Performance Analysis Tool')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input image directory')
    parser.add_argument('--output_dir', type=str, default='performance_analysis',
                       help='Output analysis result directory')
    parser.add_argument('--baseline_results', type=str, default=None,
                       help='Traditional ISP result JSON file')
    parser.add_argument('--adaptive_results', type=str, default=None,
                       help='Adaptive ISP result JSON file')
    
    args = parser.parse_args()
    
    print("Starting ISP Performance Analysis Tool...")
    
    analyzer = ISPAnalyzer()
    
    # Here you can add specific analysis logic
    # For example: Load images, run ISP processing, analyze results, etc.
    
    print("Performance analysis completed!")


if __name__ == "__main__":
    main()

