"""
GUI Interface for testing driving decision model inference with visualization.
This provides an interactive way to test your trained model on segmentation masks.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageTk
import torchvision.transforms as transforms
import os
import json
from typing import Dict, Optional, Tuple

from decision_model import create_model
from decision_dataset import LABEL_MAPPING


class InferenceGUI:
    """GUI for testing model inference with visualization."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("RoadNet - Driving Decision Inference Test")
        self.root.geometry("1200x800")
        
        # Model and transform
        self.model = None
        self.device = torch.device('cpu')  # Use CPU for GUI stability
        self.transform = None
        self.current_image_path = None
        self.current_prediction = None
        self.current_confidence = None
        
        # Colors for visualization
        self.decision_colors = {
            'front': '#4CAF50',  # Green
            'left': '#2196F3',   # Blue
            'right': '#FF9800'   # Orange
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="🚗 RoadNet Driving Decision Inference", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Model loading
        ttk.Label(control_frame, text="1. Load Model:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.model_path_var = tk.StringVar()
        model_entry = ttk.Entry(control_frame, textvariable=self.model_path_var, width=30)
        model_entry.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Button(control_frame, text="Browse Model", 
                  command=self.browse_model).grid(row=1, column=1, padx=(5, 0), pady=(0, 5))
        ttk.Button(control_frame, text="Load Model", 
                  command=self.load_model).grid(row=2, column=0, sticky=tk.W, pady=(0, 10))
        
        # Model status
        self.model_status_var = tk.StringVar(value="No model loaded")
        ttk.Label(control_frame, textvariable=self.model_status_var, 
                 foreground="red").grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        # Image loading
        ttk.Label(control_frame, text="2. Load Image:").grid(row=4, column=0, sticky=tk.W, pady=(0, 5))
        self.image_path_var = tk.StringVar()
        image_entry = ttk.Entry(control_frame, textvariable=self.image_path_var, width=30)
        image_entry.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Button(control_frame, text="Browse Image", 
                  command=self.browse_image).grid(row=5, column=1, padx=(5, 0), pady=(0, 5))
        
        # Quick test buttons
        ttk.Label(control_frame, text="Quick Test:").grid(row=6, column=0, sticky=tk.W, pady=(10, 5))
        ttk.Button(control_frame, text="Test Sample Images", 
                  command=self.test_sample_images).grid(row=7, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Inference button
        self.inference_button = ttk.Button(control_frame, text="🚀 Run Inference", 
                                         command=self.run_inference, state='disabled')
        self.inference_button.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(20, 10))
        
        # Results frame
        results_frame = ttk.LabelFrame(control_frame, text="Results", padding="10")
        results_frame.grid(row=9, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        self.prediction_var = tk.StringVar(value="No prediction")
        self.confidence_var = tk.StringVar(value="")
        
        ttk.Label(results_frame, text="Prediction:").grid(row=0, column=0, sticky=tk.W)
        self.prediction_label = ttk.Label(results_frame, textvariable=self.prediction_var, 
                                         font=('Arial', 12, 'bold'))
        self.prediction_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        ttk.Label(results_frame, text="Confidence:").grid(row=2, column=0, sticky=tk.W)
        ttk.Label(results_frame, textvariable=self.confidence_var).grid(row=3, column=0, sticky=tk.W, pady=(0, 5))
        
        # Right panel - Visualization
        viz_frame = ttk.LabelFrame(main_frame, text="Visualization", padding="10")
        viz_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 5))
        self.fig.suptitle('Segmentation Mask and Prediction')
        
        # Setup axes
        self.ax1.set_title('Input Segmentation Mask')
        self.ax1.axis('off')
        
        self.ax2.set_title('Prediction Confidence')
        self.ax2.set_xlabel('Decision')
        self.ax2.set_ylabel('Confidence')
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for control frame
        control_frame.columnconfigure(0, weight=1)
        viz_frame.columnconfigure(0, weight=1)
        viz_frame.rowconfigure(0, weight=1)
        
        # Initial empty plot
        self.clear_visualization()
    
    def clear_visualization(self):
        """Clear the visualization plots."""
        self.ax1.clear()
        self.ax1.set_title('Input Segmentation Mask')
        self.ax1.text(0.5, 0.5, 'No image loaded', ha='center', va='center', 
                     transform=self.ax1.transAxes, fontsize=12, color='gray')
        self.ax1.axis('off')
        
        self.ax2.clear()
        self.ax2.set_title('Prediction Confidence')
        self.ax2.set_xlabel('Decision')
        self.ax2.set_ylabel('Confidence')
        self.ax2.text(0.5, 0.5, 'No prediction', ha='center', va='center', 
                     transform=self.ax2.transAxes, fontsize=12, color='gray')
        
        self.canvas.draw()
    
    def browse_model(self):
        """Browse for model file."""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("PyTorch Models", "*.pth"), ("All Files", "*.*")],
            initialdir="./checkpoints"
        )
        if filename:
            self.model_path_var.set(filename)
    
    def browse_image(self):
        """Browse for image file."""
        filename = filedialog.askopenfilename(
            title="Select Segmentation Mask",
            filetypes=[("PNG Files", "*.png"), ("JPEG Files", "*.jpg"), ("All Files", "*.*")],
            initialdir="../data/sequence.0"
        )
        if filename:
            self.image_path_var.set(filename)
    
    def load_model(self):
        """Load the selected model."""
        model_path = self.model_path_var.get()
        if not model_path or not os.path.exists(model_path):
            messagebox.showerror("Error", "Please select a valid model file")
            return
        
        try:
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Determine model type (try to infer from file or use ResNet as default)
            model_type = 'resnet'  # Default to ResNet
            
            # Create model
            self.model = create_model(model_type, num_classes=3)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Create transform (same as training validation transform)
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Update status
            val_acc = checkpoint.get('val_accuracy', 'Unknown')
            self.model_status_var.set(f"✅ Model loaded (Val Acc: {val_acc:.1f}%)")
            self.model_status_var.set(f"✅ Model loaded successfully")
            
            # Enable inference button
            self.inference_button.config(state='normal')
            
            messagebox.showinfo("Success", f"Model loaded successfully!\nValidation Accuracy: {val_acc:.1f}%")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.model_status_var.set("❌ Failed to load model")
    
    def test_sample_images(self):
        """Test with sample images from the dataset."""
        sample_dir = "../data/sequence.0"
        if not os.path.exists(sample_dir):
            messagebox.showerror("Error", f"Sample directory not found: {sample_dir}")
            return
        
        # Find segmentation mask files
        mask_files = [f for f in os.listdir(sample_dir) if f.endswith('.camera.semantic segmentation.png')]
        if not mask_files:
            messagebox.showerror("Error", "No segmentation mask files found in sample directory")
            return
        
        # Show selection dialog
        selection_window = tk.Toplevel(self.root)
        selection_window.title("Select Sample Image")
        selection_window.geometry("400x300")
        
        ttk.Label(selection_window, text="Select a sample image:").pack(pady=10)
        
        listbox = tk.Listbox(selection_window, height=10)
        listbox.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        for file in sorted(mask_files)[:20]:  # Show first 20 files
            listbox.insert(tk.END, file)
        
        def select_sample():
            selection = listbox.curselection()
            if selection:
                selected_file = mask_files[selection[0]]
                self.image_path_var.set(os.path.join(sample_dir, selected_file))
                selection_window.destroy()
        
        ttk.Button(selection_window, text="Select", command=select_sample).pack(pady=10)
    
    def run_inference(self):
        """Run inference on the selected image."""
        if self.model is None:
            messagebox.showerror("Error", "Please load a model first")
            return
        
        image_path = self.image_path_var.get()
        if not image_path or not os.path.exists(image_path):
            messagebox.showerror("Error", "Please select a valid image file")
            return
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            original_image = np.array(image)
            
            # Apply transform
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Get prediction name
            prediction_name = LABEL_MAPPING[predicted_class]
            
            # Update results
            self.current_prediction = prediction_name
            self.current_confidence = confidence
            self.current_image_path = image_path
            
            self.prediction_var.set(f"{prediction_name.upper()}")
            self.confidence_var.set(f"{confidence*100:.1f}%")
            
            # Update prediction label color
            color = self.decision_colors.get(prediction_name, 'black')
            self.prediction_label.config(foreground=color)
            
            # Update visualization
            self.update_visualization(original_image, probabilities[0].cpu().numpy(), prediction_name)
            
        except Exception as e:
            messagebox.showerror("Error", f"Inference failed: {str(e)}")
    
    def update_visualization(self, image: np.ndarray, probabilities: np.ndarray, prediction: str):
        """Update the visualization with image and results."""
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        
        # Plot original image
        self.ax1.imshow(image)
        self.ax1.set_title(f'Input Segmentation Mask\n{os.path.basename(self.current_image_path)}')
        self.ax1.axis('off')
        
        # Add prediction overlay
        bbox_props = dict(boxstyle="round,pad=0.3", facecolor=self.decision_colors[prediction], alpha=0.8)
        self.ax1.text(0.02, 0.98, f'Prediction: {prediction.upper()}', 
                     transform=self.ax1.transAxes, fontsize=12, fontweight='bold',
                     verticalalignment='top', bbox=bbox_props, color='white')
        
        # Plot confidence bar chart
        labels = list(LABEL_MAPPING.values())
        colors = [self.decision_colors[label] for label in labels]
        bars = self.ax2.bar(labels, probabilities * 100, color=colors, alpha=0.7)
        
        # Highlight predicted class
        predicted_idx = labels.index(prediction)
        bars[predicted_idx].set_alpha(1.0)
        bars[predicted_idx].set_edgecolor('black')
        bars[predicted_idx].set_linewidth(2)
        
        self.ax2.set_title('Prediction Confidence')
        self.ax2.set_ylabel('Confidence (%)')
        self.ax2.set_ylim(0, 100)
        
        # Add percentage labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            self.ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                         f'{prob*100:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Adjust layout and draw
        self.fig.tight_layout()
        self.canvas.draw()


def main():
    """Main function to run the GUI."""
    root = tk.Tk()
    app = InferenceGUI(root)
    
    # Set up proper closing
    def on_closing():
        plt.close('all')
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
