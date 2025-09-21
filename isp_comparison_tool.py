#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISPComparisonVisualizationTool

This tool provides an interactive interface for comparing traditional ISP and AdaptiveISP effects.
Users can:
1. Load images for real-time comparison
2. Adjust traditional ISP parameters to observe effect changes
3. View AdaptiveISP's automatic tuning process
4. Analyze performance metrics and detection results
5. Generate detailed comparison reports

Author: Based on AdaptiveISP paper implementation
"""

import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button, RadioButtons
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import json
import time
from typing import Dict, List, Tuple, Any

# Import our modules
from isp_pipeline_demo import (
    ISPPipeline, ExposureModule, GammaModule, WhiteBalanceModule,
    DenoiseModule, SharpenModule, ContrastModule, SaturationModule,
    create_baseline_pipeline, create_adaptive_pipeline
)


class ISPComparisonTool:
    """ISPComparisonVisualizationTool"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI-driven ISP Tuning Comparison Tool")
        self.root.geometry("1400x900")
        
        # Data storage
        self.original_image = None
        self.baseline_result = None
        self.adaptive_result = None
        self.baseline_pipeline = None
        self.adaptive_pipeline = None
        
        # Performance metrics
        self.performance_metrics = {
            'baseline_score': 0.0,
            'adaptive_score': 0.0,
            'processing_times': {'baseline': 0.0, 'adaptive': 0.0},
            'applied_filters': []
        }
        
        # Create interface
        self._create_widgets()
        self._setup_pipelines()
    
    def _create_widgets(self):
        """Create user interface"""
        # Main framework
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left control panel
        control_frame = ttk.LabelFrame(main_frame, text="Control panel", width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_frame.pack_propagate(False)
        
        # ImageLoad
        load_frame = ttk.Frame(control_frame)
        load_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(load_frame, text="LoadImage", command=self._load_image).pack(fill=tk.X)
        ttk.Button(load_frame, text="UseTestImage", command=self._load_test_image).pack(fill=tk.X, pady=(5, 0))
        
        # TraditionalISPParameterControl
        self._create_baseline_controls(control_frame)
        
        # AdaptiveISPControl
        self._create_adaptive_controls(control_frame)
        
        # PerformanceDisplay
        self._create_performance_display(control_frame)
        
        # RightImageDisplayRegion
        image_frame = ttk.LabelFrame(main_frame, text="ImageComparison")
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create image display labels
        self.image_labels = {}
        for i, (title, key) in enumerate([("Original image", "original"), 
                                        ("TraditionalISP", "baseline"), 
                                        ("AdaptiveISP", "adaptive")]):
            frame = ttk.Frame(image_frame)
            frame.grid(row=0, column=i, padx=5, pady=5, sticky="nsew")
            
            ttk.Label(frame, text=title, font=("Arial", 12, "bold")).pack()
            
            label = tk.Label(frame, text="No image", width=30, height=20, 
                           bg="lightgray", relief="sunken")
            label.pack(padx=5, pady=5)
            self.image_labels[key] = label
        
        # Configure grid weights
        image_frame.grid_columnconfigure(0, weight=1)
        image_frame.grid_columnconfigure(1, weight=1)
        image_frame.grid_columnconfigure(2, weight=1)
        image_frame.grid_rowconfigure(0, weight=1)
        
        # Bottom status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief="sunken")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
    
    def _create_baseline_controls(self, parent):
        """CreateTraditionalISPParameterControl"""
        baseline_frame = ttk.LabelFrame(parent, text="TraditionalISPParameter")
        baseline_frame.pack(fill=tk.X, pady=5)
        
        self.baseline_vars = {}
        self.baseline_sliders = {}
        
        # ExposureAdjustment
        exposure_frame = ttk.Frame(baseline_frame)
        exposure_frame.pack(fill=tk.X, pady=2)
        ttk.Label(exposure_frame, text="Exposure:").pack(side=tk.LEFT)
        self.baseline_vars['exposure'] = tk.DoubleVar(value=0.0)
        slider = tk.Scale(exposure_frame, from_=-2.0, to=2.0, resolution=0.1,
                         orient=tk.HORIZONTAL, variable=self.baseline_vars['exposure'],
                         command=self._update_baseline)
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.baseline_sliders['exposure'] = slider
        
        # Gamma Correction
        gamma_frame = ttk.Frame(baseline_frame)
        gamma_frame.pack(fill=tk.X, pady=2)
        ttk.Label(gamma_frame, text="Gamma:").pack(side=tk.LEFT)
        self.baseline_vars['gamma'] = tk.DoubleVar(value=2.2)
        slider = tk.Scale(gamma_frame, from_=0.5, to=3.0, resolution=0.1,
                         orient=tk.HORIZONTAL, variable=self.baseline_vars['gamma'],
                         command=self._update_baseline)
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.baseline_sliders['gamma'] = slider
        
        # White BalanceR
        wb_r_frame = ttk.Frame(baseline_frame)
        wb_r_frame.pack(fill=tk.X, pady=2)
        ttk.Label(wb_r_frame, text="White BalanceR:").pack(side=tk.LEFT)
        self.baseline_vars['wb_r'] = tk.DoubleVar(value=1.0)
        slider = tk.Scale(wb_r_frame, from_=0.5, to=2.0, resolution=0.1,
                         orient=tk.HORIZONTAL, variable=self.baseline_vars['wb_r'],
                         command=self._update_baseline)
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.baseline_sliders['wb_r'] = slider
        
        # White BalanceG
        wb_g_frame = ttk.Frame(baseline_frame)
        wb_g_frame.pack(fill=tk.X, pady=2)
        ttk.Label(wb_g_frame, text="White BalanceG:").pack(side=tk.LEFT)
        self.baseline_vars['wb_g'] = tk.DoubleVar(value=1.0)
        slider = tk.Scale(wb_g_frame, from_=0.5, to=2.0, resolution=0.1,
                         orient=tk.HORIZONTAL, variable=self.baseline_vars['wb_g'],
                         command=self._update_baseline)
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.baseline_sliders['wb_g'] = slider
        
        # White BalanceB
        wb_b_frame = ttk.Frame(baseline_frame)
        wb_b_frame.pack(fill=tk.X, pady=2)
        ttk.Label(wb_b_frame, text="White BalanceB:").pack(side=tk.LEFT)
        self.baseline_vars['wb_b'] = tk.DoubleVar(value=1.0)
        slider = tk.Scale(wb_b_frame, from_=0.5, to=2.0, resolution=0.1,
                         orient=tk.HORIZONTAL, variable=self.baseline_vars['wb_b'],
                         command=self._update_baseline)
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.baseline_sliders['wb_b'] = slider
        
        # Denoising强度
        denoise_frame = ttk.Frame(baseline_frame)
        denoise_frame.pack(fill=tk.X, pady=2)
        ttk.Label(denoise_frame, text="Denoising:").pack(side=tk.LEFT)
        self.baseline_vars['denoise'] = tk.DoubleVar(value=0.3)
        slider = tk.Scale(denoise_frame, from_=0.0, to=1.0, resolution=0.1,
                         orient=tk.HORIZONTAL, variable=self.baseline_vars['denoise'],
                         command=self._update_baseline)
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.baseline_sliders['denoise'] = slider
        
        # Sharpening强度
        sharpen_frame = ttk.Frame(baseline_frame)
        sharpen_frame.pack(fill=tk.X, pady=2)
        ttk.Label(sharpen_frame, text="Sharpening:").pack(side=tk.LEFT)
        self.baseline_vars['sharpen'] = tk.DoubleVar(value=0.4)
        slider = tk.Scale(sharpen_frame, from_=0.0, to=1.0, resolution=0.1,
                         orient=tk.HORIZONTAL, variable=self.baseline_vars['sharpen'],
                         command=self._update_baseline)
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.baseline_sliders['sharpen'] = slider
        
        # Contrast
        contrast_frame = ttk.Frame(baseline_frame)
        contrast_frame.pack(fill=tk.X, pady=2)
        ttk.Label(contrast_frame, text="Contrast:").pack(side=tk.LEFT)
        self.baseline_vars['contrast'] = tk.DoubleVar(value=0.1)
        slider = tk.Scale(contrast_frame, from_=-0.5, to=0.5, resolution=0.1,
                         orient=tk.HORIZONTAL, variable=self.baseline_vars['contrast'],
                         command=self._update_baseline)
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.baseline_sliders['contrast'] = slider
        
        # Saturation
        saturation_frame = ttk.Frame(baseline_frame)
        saturation_frame.pack(fill=tk.X, pady=2)
        ttk.Label(saturation_frame, text="Saturation:").pack(side=tk.LEFT)
        self.baseline_vars['saturation'] = tk.DoubleVar(value=1.1)
        slider = tk.Scale(saturation_frame, from_=0.5, to=2.0, resolution=0.1,
                         orient=tk.HORIZONTAL, variable=self.baseline_vars['saturation'],
                         command=self._update_baseline)
        slider.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        self.baseline_sliders['saturation'] = slider
    
    def _create_adaptive_controls(self, parent):
        """CreateAdaptiveISPControl"""
        adaptive_frame = ttk.LabelFrame(parent, text="AdaptiveISPControl")
        adaptive_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(adaptive_frame, text="RunAdaptiveISP", 
                  command=self._run_adaptive_isp).pack(fill=tk.X, pady=2)
        ttk.Button(adaptive_frame, text="Reset parameters", 
                  command=self._reset_adaptive_params).pack(fill=tk.X, pady=2)
        
        # Display applied filters
        self.filter_display = tk.Text(adaptive_frame, height=6, width=30)
        self.filter_display.pack(fill=tk.X, pady=2)
    
    def _create_performance_display(self, parent):
        """Create performance display"""
        perf_frame = ttk.LabelFrame(parent, text="Performance metrics")
        perf_frame.pack(fill=tk.X, pady=5)
        
        # Detection score
        self.score_frame = ttk.Frame(perf_frame)
        self.score_frame.pack(fill=tk.X, pady=2)
        
        self.baseline_score_var = tk.StringVar(value="TraditionalISP: --")
        self.adaptive_score_var = tk.StringVar(value="AdaptiveISP: --")
        
        ttk.Label(self.score_frame, textvariable=self.baseline_score_var).pack()
        ttk.Label(self.score_frame, textvariable=self.adaptive_score_var).pack()
        
        # HandleTime
        self.time_frame = ttk.Frame(perf_frame)
        self.time_frame.pack(fill=tk.X, pady=2)
        
        self.baseline_time_var = tk.StringVar(value="TraditionalISPTime: --")
        self.adaptive_time_var = tk.StringVar(value="AdaptiveISPTime: --")
        
        ttk.Label(self.time_frame, textvariable=self.baseline_time_var).pack()
        ttk.Label(self.time_frame, textvariable=self.adaptive_time_var).pack()
        
        # PerformanceEnhance
        self.improvement_var = tk.StringVar(value="PerformanceEnhance: --")
        ttk.Label(perf_frame, textvariable=self.improvement_var, 
                 font=("Arial", 10, "bold")).pack(pady=2)
        
        # SaveResult按钮
        ttk.Button(perf_frame, text="Save comparison result", 
                  command=self._save_results).pack(fill=tk.X, pady=2)
    
    def _setup_pipelines(self):
        """Setup ISP pipeline"""
        self.baseline_pipeline = create_baseline_pipeline()
        self.adaptive_pipeline = create_adaptive_pipeline()
    
    def _load_image(self):
        """LoadImage"""
        file_path = filedialog.askopenfilename(
            title="SelectImageFile",
            filetypes=[("ImageFile", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        
        if file_path:
            try:
                # LoadImage
                image = cv2.imread(file_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.original_image = image.astype(np.float32) / 255.0
                
                # DisplayOriginal image
                self._display_image(self.original_image, "original")
                
                # Automatic processing
                self._update_baseline()
                self._run_adaptive_isp()
                
                self.status_var.set(f"Image loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"LoadImageFailed: {str(e)}")
    
    def _load_test_image(self):
        """LoadTestImage"""
        # CreateTestImage
        height, width = 400, 400
        image = np.zeros((height, width, 3), dtype=np.float32)
        
        # Add gradient background
        for i in range(height):
            for j in range(width):
                image[i, j] = [i/height, j/width, (i+j)/(height+width)]
        
        # Add geometric shapes
        cv2.rectangle(image, (50, 50), (150, 150), (1, 0, 0), -1)
        cv2.circle(image, (300, 100), 50, (0, 1, 0), -1)
        cv2.ellipse(image, (200, 300), (80, 40), 45, 0, 360, (0, 0, 1), -1)
        
        # AddNoise
        noise = np.random.normal(0, 0.05, image.shape)
        image = np.clip(image + noise, 0, 1)
        
        self.original_image = image
        self._display_image(self.original_image, "original")
        
        # Automatic processing
        self._update_baseline()
        self._run_adaptive_isp()
        
        self.status_var.set("Test image loaded")
    
    def _display_image(self, image, key):
        """DisplayImage"""
        if image is None:
            return
        
        # AdjustmentImageSize
        display_size = (200, 200)
        image_display = cv2.resize(image, display_size)
        
        # Convert to PIL image
        image_pil = Image.fromarray((image_display * 255).astype(np.uint8))
        image_tk = ImageTk.PhotoImage(image_pil)
        
        # Update label
        self.image_labels[key].configure(image=image_tk, text="")
        self.image_labels[key].image = image_tk  # Keep reference
    
    def _update_baseline(self, *args):
        """UpdateTraditionalISPResult"""
        if self.original_image is None:
            return
        
        try:
            # Update pipeline parameters
            self.baseline_pipeline.modules[0].parameters['exposure'] = self.baseline_vars['exposure'].get()
            self.baseline_pipeline.modules[2].parameters['gamma'] = self.baseline_vars['gamma'].get()
            
            wb_gains = [
                self.baseline_vars['wb_r'].get(),
                self.baseline_vars['wb_g'].get(),
                self.baseline_vars['wb_b'].get()
            ]
            self.baseline_pipeline.modules[1].parameters['rgb_gains'] = wb_gains
            
            self.baseline_pipeline.modules[3].parameters['strength'] = self.baseline_vars['denoise'].get()
            self.baseline_pipeline.modules[4].parameters['strength'] = self.baseline_vars['sharpen'].get()
            self.baseline_pipeline.modules[5].parameters['contrast'] = self.baseline_vars['contrast'].get()
            self.baseline_pipeline.modules[6].parameters['saturation'] = self.baseline_vars['saturation'].get()
            
            # HandleImage
            start_time = time.time()
            self.baseline_result = self.baseline_pipeline.process_image(self.original_image)
            processing_time = time.time() - start_time
            
            # DisplayResult
            self._display_image(self.baseline_result, "baseline")
            
            # CalculatePerformance metrics
            baseline_score = self._calculate_detection_score(self.baseline_result)
            self.performance_metrics['baseline_score'] = baseline_score
            self.performance_metrics['processing_times']['baseline'] = processing_time
            
            # UpdateDisplay
            self.baseline_score_var.set(f"TraditionalISP: {baseline_score:.3f}")
            self.baseline_time_var.set(f"TraditionalISPTime: {processing_time:.3f}s")
            
            self._update_improvement()
            
        except Exception as e:
            self.status_var.set(f"UpdateTraditionalISPFailed: {str(e)}")
    
    def _run_adaptive_isp(self):
        """RunAdaptiveISP"""
        if self.original_image is None:
            return
        
        try:
            # Simulate AdaptiveISP parameter optimization process
            applied_filters = []
            
            # Automatically adjust parameters based on image features
            # Here we use simplified heuristic methods, in actual applications deep learning models would be used
            
            # Analyze image features
            gray = cv2.cvtColor((self.original_image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            brightness = np.mean(gray) / 255.0
            contrast = np.std(gray) / 255.0
            noise_level = self._estimate_noise_level(gray)
            
            # Adjust parameters based on features
            if brightness < 0.3:  # Low light image
                exposure = 0.5
                gamma = 1.8
                denoise = 0.7
                sharpen = 0.6
                applied_filters.append("ExposureEnhancement")
                applied_filters.append("Gamma Correction")
                applied_filters.append("Denoising")
                applied_filters.append("Sharpening")
            elif brightness > 0.7:  # High light image
                exposure = -0.3
                gamma = 2.5
                denoise = 0.3
                sharpen = 0.4
                applied_filters.append("ExposureDecrease")
                applied_filters.append("Gamma Correction")
                applied_filters.append("LightDenoising")
                applied_filters.append("LightSharpening")
            else:  # Normal Lighting
                exposure = 0.1
                gamma = 2.2
                denoise = 0.4
                sharpen = 0.5
                applied_filters.append("LightExposureAdjustment")
                applied_filters.append("Gamma Correction")
                applied_filters.append("Denoising")
                applied_filters.append("Sharpening")
            
            # 根据ContrastAdjustment
            if contrast < 0.15:
                contrast_adj = 0.2
                saturation = 1.3
                applied_filters.append("ContrastEnhancement")
                applied_filters.append("Saturation enhancement")
            else:
                contrast_adj = 0.0
                saturation = 1.1
                applied_filters.append("SaturationFine-tuning")
            
            # Adjust based on noise level
            if noise_level > 0.05:
                denoise = max(denoise, 0.6)
                applied_filters.append("Enhanced denoising")
            
            # Update adaptive pipeline parameters
            self.adaptive_pipeline.modules[0].parameters['exposure'] = exposure
            self.adaptive_pipeline.modules[1].parameters['gamma'] = gamma
            self.adaptive_pipeline.modules[4].parameters['strength'] = denoise
            self.adaptive_pipeline.modules[5].parameters['strength'] = sharpen
            self.adaptive_pipeline.modules[6].parameters['contrast'] = contrast_adj
            self.adaptive_pipeline.modules[7].parameters['saturation'] = saturation
            
            # HandleImage
            start_time = time.time()
            self.adaptive_result = self.adaptive_pipeline.process_image(self.original_image)
            processing_time = time.time() - start_time
            
            # DisplayResult
            self._display_image(self.adaptive_result, "adaptive")
            
            # CalculatePerformance metrics
            adaptive_score = self._calculate_detection_score(self.adaptive_result)
            self.performance_metrics['adaptive_score'] = adaptive_score
            self.performance_metrics['processing_times']['adaptive'] = processing_time
            self.performance_metrics['applied_filters'] = applied_filters
            
            # UpdateDisplay
            self.adaptive_score_var.set(f"AdaptiveISP: {adaptive_score:.3f}")
            self.adaptive_time_var.set(f"AdaptiveISPTime: {processing_time:.3f}s")
            
            # Update filter display
            filter_text = "Applied filters:\n" + "\n".join(applied_filters)
            self.filter_display.delete(1.0, tk.END)
            self.filter_display.insert(1.0, filter_text)
            
            self._update_improvement()
            
            self.status_var.set("AdaptiveISPHandleCompleted")
            
        except Exception as e:
            self.status_var.set(f"AdaptiveISPHandleFailed: {str(e)}")
    
    def _estimate_noise_level(self, image):
        """Estimate image noise level"""
        # Use Laplacian operator to estimate noise
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        noise_level = np.std(laplacian) / 255.0
        return noise_level
    
    def _calculate_detection_score(self, image):
        """CalculateDetection score（模拟）"""
        # Simplified detection quality evaluation
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Contrast
        contrast = np.std(gray) / 255.0
        
        # Brightness distribution
        brightness = np.mean(gray) / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2  # 接近0.5最好
        
        # Comprehensive score
        detection_score = (edge_density * 0.4 + contrast * 0.3 + brightness_score * 0.3) * 0.8
        
        return min(max(detection_score, 0.0), 1.0)
    
    def _update_improvement(self):
        """UpdatePerformanceEnhanceDisplay"""
        baseline_score = self.performance_metrics['baseline_score']
        adaptive_score = self.performance_metrics['adaptive_score']
        
        if baseline_score > 0:
            improvement = (adaptive_score - baseline_score) / baseline_score * 100
            self.improvement_var.set(f"PerformanceEnhance: {improvement:+.1f}%")
        else:
            self.improvement_var.set("PerformanceEnhance: --")
    
    def _reset_adaptive_params(self):
        """Reset adaptive parameters"""
        # Reset adaptive pipeline parameters
        for module in self.adaptive_pipeline.modules:
            if hasattr(module, 'parameters'):
                if 'exposure' in module.parameters:
                    module.parameters['exposure'] = 0.0
                if 'gamma' in module.parameters:
                    module.parameters['gamma'] = 1.0
                if 'strength' in module.parameters:
                    module.parameters['strength'] = 0.0
                if 'contrast' in module.parameters:
                    module.parameters['contrast'] = 0.0
                if 'saturation' in module.parameters:
                    module.parameters['saturation'] = 1.0
        
        self.status_var.set("Adaptive parameters reset")
    
    def _save_results(self):
        """Save comparison result"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "No image to save")
            return
        
        # SelectSaveDirectory
        save_dir = filedialog.askdirectory(title="SelectSaveDirectory")
        if not save_dir:
            return
        
        try:
            # SaveImage
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Create comparison chart
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(self.original_image)
            axes[0].set_title('Original image')
            axes[0].axis('off')
            
            axes[1].imshow(self.baseline_result)
            axes[1].set_title(f'TraditionalISP (Score: {self.performance_metrics["baseline_score"]:.3f})')
            axes[1].axis('off')
            
            axes[2].imshow(self.adaptive_result)
            axes[2].set_title(f'AdaptiveISP (Score: {self.performance_metrics["adaptive_score"]:.3f})')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'isp_comparison_{timestamp}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # SavePerformanceData
            performance_data = {
                'timestamp': timestamp,
                'baseline_score': self.performance_metrics['baseline_score'],
                'adaptive_score': self.performance_metrics['adaptive_score'],
                'processing_times': self.performance_metrics['processing_times'],
                'applied_filters': self.performance_metrics['applied_filters'],
                'improvement_percentage': (self.performance_metrics['adaptive_score'] - 
                                        self.performance_metrics['baseline_score']) / 
                                       self.performance_metrics['baseline_score'] * 100
            }
            
            with open(os.path.join(save_dir, f'performance_data_{timestamp}.json'), 'w') as f:
                json.dump(performance_data, f, indent=2)
            
            messagebox.showinfo("Success", f"Result saved to: {save_dir}")
            self.status_var.set(f"Result saved to: {save_dir}")
            
        except Exception as e:
            messagebox.showerror("Error", f"SaveFailed: {str(e)}")
    
    def run(self):
        """RunApplicationProgram"""
        self.root.mainloop()


def main():
    """Main function"""
    print("Starting ISP comparison visualization tool...")
    app = ISPComparisonTool()
    app.run()


if __name__ == "__main__":
    main()

