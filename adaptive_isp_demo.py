#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chapter 7: Camera Tuning - Case Study: AI-Driven ISP Tuning Automation

This demo script demonstrates how to use AdaptiveISP for intelligent image signal processor tuning,
automatically optimizing ISP parameters through deep reinforcement learning to improve object detection performance.

Main features:
1. Real-time adaptive ISP tuning
2. Traditional ISP vs Adaptive ISP comparison
3. Performance evaluation and visualization
4. Parameter tuning process demonstration

Author: Based on AdaptiveISP paper implementation
"""

import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import argparse
import json
import time
from pathlib import Path

# AddAdaptiveISPPath
sys.path.append('AdaptiveISP')
sys.path.append('AdaptiveISP/yolov3')

# Import AdaptiveISP core modules
try:
    from agent import Agent
    from config import cfg
    from isp.filters import *
    import importlib
    ADAPTIVEISP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Unable to import AdaptiveISP modules: {e}")
    print("Will use simplified version for demonstration")
    ADAPTIVEISP_AVAILABLE = False

class AdaptiveISPDemo:
    """AI-driven ISP tuning demonstration class"""
    
    def __init__(self, isp_weights_path=None, yolo_weights_path=None):
        """
        Initialize AdaptiveISP demonstration system
        
        Args:
            isp_weights_path: AdaptiveISP pre-trained model path
            yolo_weights_path: YOLO detection model path
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize configuration
        if ADAPTIVEISP_AVAILABLE:
            self.cfg = cfg
            self.filters_number = len(cfg.filters)
            self.z_dim = 3 + self.filters_number * cfg.z_dim_per_filter
            self.num_state_dim = 3 + self.filters_number
        else:
            # Simplified configuration
            self.cfg = None
            self.filters_number = 8
            self.z_dim = 16
            self.num_state_dim = 8
        
        # Initialize models
        self._init_models(isp_weights_path, yolo_weights_path)
        
        # Traditional ISP parameters (fixed configuration)
        self.baseline_params = self._get_baseline_isp_params()
        
        # Performance statistics
        self.performance_stats = {
            'adaptive_maps': [],
            'baseline_maps': [],
            'adaptive_times': [],
            'baseline_times': []
        }
    
    def _init_models(self, isp_weights_path, yolo_weights_path):
        """Initialize ISP and detection models"""
        
        if ADAPTIVEISP_AVAILABLE:
            # Initialize AdaptiveISP agent
            print("Initializing AdaptiveISP agent...")
            self.isp_agent = Agent(self.cfg, shape=(6 + self.filters_number, 64, 64)).to(self.device)
            
            # Load pre-trained weights (if provided)
            if isp_weights_path and os.path.exists(isp_weights_path):
                print(f"Loading ISP model weights: {isp_weights_path}")
                checkpoint = torch.load(isp_weights_path, map_location=self.device)
                self.isp_agent.load_state_dict(checkpoint['agent_model'])
            else:
                print("Warning: No ISP pre-trained weights provided, using randomly initialized model")
            
            self.isp_agent.eval()
            
            # Get filter names
            self.filter_names = [f.get_short_name() for f in self.isp_agent.filters]
            print(f"Available ISP filters: {self.filter_names}")
        else:
            print("Using simplified version, skipping AdaptiveISP model initialization")
            self.isp_agent = None
            self.filter_names = ['E', 'G', 'W', 'Shr', 'NLM', 'T', 'Ct', 'S+']
        
        # Initialize YOLO detection model
        print("Initializing YOLO detection model...")
        try:
            if ADAPTIVEISP_AVAILABLE:
                from yolov3.models.yolo import Model
                from yolov3.utils.downloads import attempt_download
                
                # Download or load YOLO weights
                if yolo_weights_path and os.path.exists(yolo_weights_path):
                    weights_path = yolo_weights_path
                else:
                    weights_path = attempt_download('yolov3.pt')
                
                # Load YOLO model
                checkpoint = torch.load(weights_path, map_location=self.device)
                self.yolo_model = Model('yolov3/models/yolov3.yaml', ch=3, nc=80).to(self.device)
                
                # Load weights
                if 'model' in checkpoint:
                    self.yolo_model.load_state_dict(checkpoint['model'].state_dict())
                else:
                    self.yolo_model.load_state_dict(checkpoint)
                
                self.yolo_model.eval()
                print("YOLO model loaded successfully")
            else:
                print("Using simplified detection simulator")
                self.yolo_model = None
            
        except Exception as e:
            print(f"YOLO model loading failed: {e}")
            print("Will use simplified detection simulator")
            self.yolo_model = None
    
    def _get_baseline_isp_params(self):
        """Get traditional ISP fixed parameter configuration"""
        return {
            'exposure': 0.0,
            'gamma': 1.0,
            'white_balance': [1.0, 1.0, 1.0],
            'contrast': 0.0,
            'saturation': 0.5,
            'sharpen': 0.3,
            'denoise': 0.2,
            'tone_curve': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        }
    
    def _apply_baseline_isp(self, image):
        """Apply traditional ISP processing"""
        # Simplified traditional ISP processing
        processed = image.copy()
        
        # Exposure adjustment
        exposure = self.baseline_params['exposure']
        processed = processed * (2 ** exposure)
        
        # Gamma correction
        gamma = self.baseline_params['gamma']
        processed = np.power(np.clip(processed, 0.001, 1.0), gamma)
        
        # White balance
        wb = np.array(self.baseline_params['white_balance']).reshape(1, 1, 3)
        processed = processed * wb
        
        # Contrast adjustment
        contrast = self.baseline_params['contrast']
        processed = processed * (1 + contrast)
        
        # Saturation adjustment
        saturation = self.baseline_params['saturation']
        gray = np.dot(processed, [0.299, 0.587, 0.114])[:, :, np.newaxis]
        processed = gray + saturation * (processed - gray)
        
        # Sharpening
        sharpen = self.baseline_params['sharpen']
        if sharpen > 0:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            processed = cv2.filter2D(processed, -1, kernel * sharpen + (1-sharpen) * np.eye(3))
        
        # Denoising
        denoise = self.baseline_params['denoise']
        if denoise > 0:
            processed = cv2.bilateralFilter((processed * 255).astype(np.uint8), 9, 75, 75).astype(np.float32) / 255
        
        return np.clip(processed, 0.0, 1.0)
    
    def _apply_adaptive_isp(self, image, steps=5):
        """Apply adaptive ISP processing"""
        if not ADAPTIVEISP_AVAILABLE or self.isp_agent is None:
            return self._apply_simplified_adaptive_isp(image)
        
        # Convert to tensor
        if isinstance(image, np.ndarray):
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        else:
            img_tensor = image
        
        # Generate noise and states
        noises = self._get_noise(1, "uniform", self.z_dim)
        states = self._get_initial_states(1, self.num_state_dim, self.filters_number)
        
        noises = torch.from_numpy(noises).to(self.device)
        states = torch.from_numpy(states).to(self.device)
        
        # Step-by-step application of ISP filters
        processed_tensor = img_tensor
        applied_filters = []
        
        for step in range(steps):
            noise = noises[step:step+1] if step < len(noises) else noises[-1:]
            (processed_tensor, new_states, _, _), debug_info, _ = self.isp_agent(
                (processed_tensor, noise, states), 1.0
            )
            
            # Record applied filters
            filter_id = debug_info['selected_filter'].item()
            applied_filters.append({
                'step': step,
                'filter_id': filter_id,
                'filter_name': self.filter_names[filter_id],
                'parameters': debug_info['filter_debug_info'][filter_id]['filter_parameters'].cpu().numpy()
            })
            
            states = new_states
            
            # Check if stopped
            if new_states[0, 1] > 0:  # STATE_STOPPED_DIM
                break
        
        # Convert back to numpy
        processed_image = processed_tensor[0].permute(1, 2, 0).cpu().numpy()
        return np.clip(processed_image, 0.0, 1.0), applied_filters
    
    def _apply_simplified_adaptive_isp(self, image):
        """Apply simplified adaptive ISP processing"""
        # Analyze image features
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray) / 255.0
        contrast = np.std(gray) / 255.0
        
        # Adjust parameters based on features
        applied_filters = []
        
        if brightness < 0.3:  # Low-light image
            exposure = 0.5
            gamma = 1.8
            denoise = 0.7
            sharpen = 0.6
            applied_filters.append("Exposure enhancement")
            applied_filters.append("Gamma correction")
            applied_filters.append("Denoising")
            applied_filters.append("Sharpening")
        elif brightness > 0.7:  # High-light image
            exposure = -0.3
            gamma = 2.5
            denoise = 0.3
            sharpen = 0.4
            applied_filters.append("Exposure reduction")
            applied_filters.append("Gamma correction")
            applied_filters.append("Light denoising")
            applied_filters.append("Light sharpening")
        else:  # Normal lighting
            exposure = 0.1
            gamma = 2.2
            denoise = 0.4
            sharpen = 0.5
            applied_filters.append("Light exposure adjustment")
            applied_filters.append("Gamma correction")
            applied_filters.append("Denoising")
            applied_filters.append("Sharpening")
        
        # Adjust based on contrast
        if contrast < 0.15:
            contrast_adj = 0.2
            saturation = 1.3
            applied_filters.extend(["Contrast enhancement", "Saturation enhancement"])
        else:
            contrast_adj = 0.0
            saturation = 1.1
            applied_filters.append("Saturation fine-tuning")
        
        # Apply parameters
        processed = image.copy()
        
        # Exposure adjustment
        processed = processed * (2 ** exposure)
        
        # Gamma correction
        processed = np.power(np.clip(processed, 0.001, 1.0), gamma)
        
        # Denoising
        if denoise > 0:
            processed = cv2.bilateralFilter((processed * 255).astype(np.uint8), 9, 75, 75).astype(np.float32) / 255
        
        # Sharpening
        if sharpen > 0:
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            processed = cv2.filter2D(processed, -1, kernel * sharpen + (1-sharpen) * np.eye(3))
        
        # Contrast adjustment
        processed = processed * (1 + contrast_adj)
        
        # Saturation adjustment
        gray = np.dot(processed, [0.299, 0.587, 0.114])[:, :, np.newaxis]
        processed = gray + saturation * (processed - gray)
        
        return np.clip(processed, 0.0, 1.0), applied_filters
    
    def _get_noise(self, batch_size, z_type, z_dim):
        """Generate noise vector"""
        if z_type == "uniform":
            return np.random.uniform(0, 1, (batch_size, z_dim)).astype(np.float32)
        else:
            return np.random.normal(0, 1, (batch_size, z_dim)).astype(np.float32)
    
    def _get_initial_states(self, batch_size, num_state_dim, filters_number):
        """Generate initial states"""
        states = np.zeros((batch_size, num_state_dim), dtype=np.float32)
        return states
    
    def _simulate_detection(self, image):
        """Simulate object detection (if no YOLO model available)"""
        # Simplified detection simulation: calculate detection confidence based on image features
        gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Calculate edge density as a proxy indicator for detection quality
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Calculate contrast
        contrast = np.std(gray) / 255.0
        
        # Comprehensive scoring (simulate mAP)
        detection_score = (edge_density * 0.6 + contrast * 0.4) * 0.8
        
        return detection_score, []
    
    def _run_detection(self, image):
        """Run object detection"""
        if self.yolo_model is None:
            return self._simulate_detection(image)
        
        # Use YOLO for actual detection
        try:
            # Preprocess image
            img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            
            # Inference
            with torch.no_grad():
                pred = self.yolo_model(img_tensor)
            
            # Simplified mAP calculation (more complex evaluation needed in actual applications)
            confidence_scores = pred[0][:, 4].cpu().numpy()
            if len(confidence_scores) > 0:
                avg_confidence = np.mean(confidence_scores)
                detection_score = min(avg_confidence * 2, 1.0)  # Normalize to [0,1]
            else:
                detection_score = 0.0
            
            # Extract detection boxes
            detections = []
            if len(pred[0]) > 0:
                for detection in pred[0]:
                    if detection[4] > 0.5:  # Confidence threshold
                        x1, y1, x2, y2 = detection[:4].cpu().numpy()
                        detections.append([x1, y1, x2, y2, detection[4].item()])
            
            return detection_score, detections
            
        except Exception as e:
            print(f"Detection process error: {e}")
            return self._simulate_detection(image)
    
    def process_image(self, image_path, save_results=True, output_dir="results"):
        """
        Process single image, compare traditional ISP and adaptive ISP effects
        
        Args:
            image_path: Input image path
            save_results: Whether to save results
            output_dir: Output directory
        """
        print(f"\nProcessing image: {image_path}")
        
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = image_path
        
        image = image.astype(np.float32) / 255.0
        
        # Apply traditional ISP
        start_time = time.time()
        baseline_result = self._apply_baseline_isp(image)
        baseline_time = time.time() - start_time
        
        # Apply adaptive ISP
        start_time = time.time()
        adaptive_result, applied_filters = self._apply_adaptive_isp(image)
        adaptive_time = time.time() - start_time
        
        # Run detection
        baseline_score, baseline_detections = self._run_detection(baseline_result)
        adaptive_score, adaptive_detections = self._run_detection(adaptive_result)
        
        # Update performance statistics
        self.performance_stats['baseline_maps'].append(baseline_score)
        self.performance_stats['adaptive_maps'].append(adaptive_score)
        self.performance_stats['baseline_times'].append(baseline_time)
        self.performance_stats['adaptive_times'].append(adaptive_time)
        
        # Print results
        print(f"Traditional ISP - Detection score: {baseline_score:.4f}, Processing time: {baseline_time:.4f}s")
        print(f"Adaptive ISP - Detection score: {adaptive_score:.4f}, Processing time: {adaptive_time:.4f}s")
        print(f"Performance improvement: {((adaptive_score - baseline_score) / baseline_score * 100):.2f}%")
        print(f"Applied filter sequence: {applied_filters}")
        
        # Save results
        if save_results:
            self._save_results(image, baseline_result, adaptive_result, 
                             applied_filters, baseline_score, adaptive_score,
                             baseline_detections, adaptive_detections,
                             output_dir, Path(image_path).stem if isinstance(image_path, str) else "demo")
        
        return {
            'original': image,
            'baseline': baseline_result,
            'adaptive': adaptive_result,
            'baseline_score': baseline_score,
            'adaptive_score': adaptive_score,
            'applied_filters': applied_filters,
            'baseline_detections': baseline_detections,
            'adaptive_detections': adaptive_detections
        }
    
    def _save_results(self, original, baseline, adaptive, applied_filters, 
                     baseline_score, adaptive_score, baseline_detections, 
                     adaptive_detections, output_dir, image_name):
        """Save processing results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create comparison chart
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Traditional ISP result
        axes[0, 1].imshow(baseline)
        axes[0, 1].set_title(f'Traditional ISP (Detection Score: {baseline_score:.3f})')
        axes[0, 1].axis('off')
        
        # Adaptive ISP result
        axes[0, 2].imshow(adaptive)
        axes[0, 2].set_title(f'Adaptive ISP (Detection Score: {adaptive_score:.3f})')
        axes[0, 2].axis('off')
        
        # Draw detection boxes (if any)
        axes[1, 0].imshow(baseline)
        axes[1, 0].set_title('Traditional ISP Detection Results')
        axes[1, 0].axis('off')
        for det in baseline_detections:
            x1, y1, x2, y2, conf = det
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='red', facecolor='none')
            axes[1, 0].add_patch(rect)
        
        axes[1, 1].imshow(adaptive)
        axes[1, 1].set_title('Adaptive ISP Detection Results')
        axes[1, 1].axis('off')
        for det in adaptive_detections:
            x1, y1, x2, y2, conf = det
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor='green', facecolor='none')
            axes[1, 1].add_patch(rect)
        
        # Filter sequence visualization
        axes[1, 2].text(0.1, 0.9, 'Adaptive ISP Filter Sequence:', transform=axes[1, 2].transAxes, 
                       fontsize=12, weight='bold')
        y_pos = 0.8
        for i, filter_info in enumerate(applied_filters):
            if isinstance(filter_info, str):
                axes[1, 2].text(0.1, y_pos, f'{i+1}. {filter_info}', 
                               transform=axes[1, 2].transAxes, fontsize=10)
            else:
                axes[1, 2].text(0.1, y_pos, f'{i+1}. {filter_info["filter_name"]}', 
                               transform=axes[1, 2].transAxes, fontsize=10)
            y_pos -= 0.1
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{image_name}_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save滤波器Parameter
        filter_params = {
            'applied_filters': applied_filters,
            'baseline_score': baseline_score,
            'adaptive_score': adaptive_score,
            'performance_improvement': (adaptive_score - baseline_score) / baseline_score * 100
        }
        
        with open(os.path.join(output_dir, f'{image_name}_parameters.json'), 'w') as f:
            json.dump(filter_params, f, indent=2, default=str)
        
        print(f"Results saved to: {output_dir}")
    
    def batch_process(self, image_dir, output_dir="batch_results"):
        """Batch process images"""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(image_dir).glob(f'*{ext}'))
            image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
        
        if not image_files:
            print(f"No image files found in directory {image_dir}")
            return
        
        print(f"Found {len(image_files)} images, starting batch processing...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, image_file in enumerate(image_files):
            print(f"\nProcessing progress: {i+1}/{len(image_files)}")
            try:
                self.process_image(str(image_file), save_results=True, 
                                 output_dir=os.path.join(output_dir, f'image_{i+1}'))
            except Exception as e:
                print(f"Error processing image {image_file}: {e}")
        
        # Generate batch processing report
        self._generate_batch_report(output_dir)
    
    def _generate_batch_report(self, output_dir):
        """Generate batch processing report"""
        if len(self.performance_stats['baseline_maps']) == 0:
            return
        
        # Calculate statistics
        baseline_avg = np.mean(self.performance_stats['baseline_maps'])
        adaptive_avg = np.mean(self.performance_stats['adaptive_maps'])
        improvement = (adaptive_avg - baseline_avg) / baseline_avg * 100
        
        baseline_time_avg = np.mean(self.performance_stats['baseline_times'])
        adaptive_time_avg = np.mean(self.performance_stats['adaptive_times'])
        
        # Create report
        report = {
            'summary': {
                'total_images': len(self.performance_stats['baseline_maps']),
                'baseline_avg_score': baseline_avg,
                'adaptive_avg_score': adaptive_avg,
                'average_improvement': improvement,
                'baseline_avg_time': baseline_time_avg,
                'adaptive_avg_time': adaptive_time_avg
            },
            'detailed_results': {
                'baseline_scores': self.performance_stats['baseline_maps'],
                'adaptive_scores': self.performance_stats['adaptive_maps'],
                'baseline_times': self.performance_stats['baseline_times'],
                'adaptive_times': self.performance_stats['adaptive_times']
            }
        }
        
        # Save report
        with open(os.path.join(output_dir, 'batch_report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create performance comparison chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Detection score comparison
        x = range(1, len(self.performance_stats['baseline_maps']) + 1)
        ax1.plot(x, self.performance_stats['baseline_maps'], 'r-o', label='Traditional ISP', linewidth=2)
        ax1.plot(x, self.performance_stats['adaptive_maps'], 'g-o', label='Adaptive ISP', linewidth=2)
        ax1.set_xlabel('Image Number')
        ax1.set_ylabel('Detection Score')
        ax1.set_title('Detection Performance Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Processing time comparison
        ax2.plot(x, self.performance_stats['baseline_times'], 'r-o', label='Traditional ISP', linewidth=2)
        ax2.plot(x, self.performance_stats['adaptive_times'], 'g-o', label='Adaptive ISP', linewidth=2)
        ax2.set_xlabel('Image Number')
        ax2.set_ylabel('Processing Time (seconds)')
        ax2.set_title('Processing Time Comparison')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nBatch processing completed!")
        print(f"Average detection score improvement: {improvement:.2f}%")
        print(f"Traditional ISP average score: {baseline_avg:.4f}")
        print(f"Adaptive ISP average score: {adaptive_avg:.4f}")
        print(f"Detailed report saved to: {output_dir}")


def create_test_image():
    """Create test image"""
    height, width = 400, 400
    image = np.ones((height, width, 3), dtype=np.uint8) * 128
    
    # Add some geometric shapes
    cv2.rectangle(image, (50, 50), (150, 150), (255, 0, 0), -1)  # Red rectangle
    cv2.circle(image, (300, 100), 50, (0, 255, 0), -1)  # Green circle
    cv2.ellipse(image, (200, 300), (80, 40), 45, 0, 360, (0, 0, 255), -1)  # Blue ellipse
    
    return image


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='AI-driven ISP tuning automation demonstration')
    parser.add_argument('--input', type=str, default='test_data/876.png',
                       help='Input image path or directory')
    parser.add_argument('--isp_weights', type=str, default=None,
                       help='AdaptiveISP pre-trained model path')
    parser.add_argument('--yolo_weights', type=str, default=None,
                       help='YOLO detection model path')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--batch', action='store_true',
                       help='Batch processing mode')
    parser.add_argument('--create_test', action='store_true',
                       help='Create test image')
    
    args = parser.parse_args()
    
    # Initialize demonstration system
    print("=" * 60)
    print("Chapter 7: Camera Tuning - AI-driven ISP Tuning Automation Demo")
    print("=" * 60)
    
    # Check if input image exists
    if not os.path.exists(args.input):
        print(f"Input image not found: {args.input}")
        if args.create_test:
            print("Creating synthetic test image...")
            test_image = create_test_image()
            cv2.imwrite(args.input, cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
            print(f"Test image created: {args.input}")
        else:
            print("Please provide a valid image path or use --create_test to generate a synthetic image")
            sys.exit(1)
    else:
        print(f"Using input image: {args.input}")
    
    demo = AdaptiveISPDemo(args.isp_weights, args.yolo_weights)
    
    if args.batch or os.path.isdir(args.input):
        # Batch processing mode
        demo.batch_process(args.input, args.output)
    else:
        # Single image processing mode
        demo.process_image(args.input, save_results=True, output_dir=args.output)
    
    print("\nDemo completed!")
    print("This case study demonstrates how AI automatically optimizes ISP parameters through reinforcement learning,")
    print("task-performance oriented, achieving a paradigm shift from 'looks good' to 'works well'.")


if __name__ == "__main__":
    main()