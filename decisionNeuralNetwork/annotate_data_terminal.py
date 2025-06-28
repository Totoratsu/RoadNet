"""
Terminal-based data annotation utility for manually labeling driving decisions.
This version works better in terminal environments and on Mac.
"""

import os
import json
import sys
from PIL import Image
from typing import Dict, List, Optional
import argparse


class TerminalDataAnnotator:
    """
    Terminal-based tool for annotating driving decisions on segmentation masks.
    Works without GUI dependencies.
    """
    
    def __init__(self, data_dir: str, labels_file: str = 'driving_labels.json'):
        """
        Initialize the annotator.
        
        Args:
            data_dir: Directory containing segmentation masks
            labels_file: File to save/load labels
        """
        self.data_dir = data_dir
        self.labels_file = os.path.join(data_dir, labels_file)
        
        # Load existing labels
        self.labels = self._load_labels()
        
        # Get list of segmentation files
        self.segmentation_files = self._get_segmentation_files()
        
        print(f"Found {len(self.segmentation_files)} segmentation files")
        print(f"Loaded {len(self.labels)} existing labels")
    
    def _load_labels(self) -> Dict[str, int]:
        """Load existing labels from file."""
        if os.path.exists(self.labels_file):
            with open(self.labels_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_labels(self):
        """Save labels to file."""
        with open(self.labels_file, 'w') as f:
            json.dump(self.labels, f, indent=2)
        print(f"✅ Labels saved to {self.labels_file}")
    
    def _get_segmentation_files(self) -> List[str]:
        """Get sorted list of segmentation files."""
        files = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.camera.semantic segmentation.png'):
                files.append(filename)
        return sorted(files, key=lambda x: int(x.split('.')[0].replace('step', '')))
    
    def _extract_step_number(self, filename: str) -> int:
        """Extract step number from filename."""
        return int(filename.split('.')[0].replace('step', ''))
    
    def _open_image_externally(self, image_path: str):
        """Open image using system default viewer."""
        try:
            if sys.platform == "darwin":  # macOS
                os.system(f"open '{image_path}'")
            elif sys.platform == "linux":
                os.system(f"xdg-open '{image_path}'")
            elif sys.platform == "win32":
                os.system(f"start '{image_path}'")
        except Exception as e:
            print(f"⚠️  Could not open image externally: {e}")
            print(f"Please manually open: {image_path}")
    
    def _show_image_info(self, filename: str, step_number: int):
        """Show image information and open it externally."""
        img_path = os.path.join(self.data_dir, filename)
        
        # Get image info
        try:
            img = Image.open(img_path)
            width, height = img.size
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return False
        
        print(f"\n{'='*60}")
        print(f"📸 IMAGE: {filename}")
        print(f"📍 Step: {step_number}")
        print(f"📏 Size: {width}x{height}")
        print(f"📂 Path: {img_path}")
        
        # Show current label if exists
        current_label = self.labels.get(str(step_number), None)
        if current_label is not None:
            label_names = ['FRONT', 'LEFT', 'RIGHT']
            print(f"🏷️  Current label: {label_names[current_label]}")
        else:
            print(f"🏷️  Current label: UNLABELED")
        
        print('='*60)
        
        # Open image externally
        print("🖼️  Opening image in system viewer...")
        self._open_image_externally(img_path)
        
        return True
    
    def annotate_batch(self, start_idx: int = 0, batch_size: int = 10):
        """
        Annotate a batch of images interactively.
        
        Args:
            start_idx: Starting index
            batch_size: Number of images to annotate in this session
        """
        end_idx = min(start_idx + batch_size, len(self.segmentation_files))
        
        print("\n" + "="*80)
        print("🚗 DRIVING DECISION ANNOTATION TOOL - TERMINAL VERSION")
        print("="*80)
        print("📋 INSTRUCTIONS:")
        print("   • An image will open in your system's default viewer")
        print("   • Look at the segmentation mask and decide what action to take")
        print("   • Consider: What should the car do in this situation?")
        print("   • Type your decision in the terminal:")
        print("     ➤ 0 = FRONT (go straight)")
        print("     ➤ 1 = LEFT  (turn left)")
        print("     ➤ 2 = RIGHT (turn right)")
        print("     ➤ s = SKIP this image")
        print("     ➤ q = QUIT and save")
        print("     ➤ u = UNDO last annotation")
        print("     ➤ r = REOPEN current image")
        print("="*80)
        
        for i in range(start_idx, end_idx):
            filename = self.segmentation_files[i]
            step_number = self._extract_step_number(filename)
            
            print(f"\n🔄 ANNOTATING {i+1}/{len(self.segmentation_files)}")
            
            # Show the image info and open it
            if not self._show_image_info(filename, step_number):
                continue
            
            # Get user input
            while True:
                try:
                    user_input = input("\n👉 Your decision (0/1/2/s/q/u/r): ").strip().lower()
                    
                    if user_input == 'q':
                        self._save_labels()
                        print("👋 Quitting and saving...")
                        return
                    elif user_input == 's':
                        print("⏭️  Skipped")
                        break
                    elif user_input == 'u':
                        if self.labels:
                            last_key = list(self.labels.keys())[-1]
                            del self.labels[last_key]
                            print(f"↩️  Undid label for step {last_key}")
                        else:
                            print("❌ No labels to undo")
                    elif user_input == 'r':
                        print("🔄 Reopening image...")
                        self._open_image_externally(os.path.join(self.data_dir, filename))
                    elif user_input in ['0', '1', '2']:
                        label = int(user_input)
                        self.labels[str(step_number)] = label
                        label_names = ['FRONT', 'LEFT', 'RIGHT']
                        print(f"✅ Labeled as: {label_names[label]}")
                        break
                    else:
                        print("❌ Invalid input. Please use: 0, 1, 2, s, q, u, or r")
                except KeyboardInterrupt:
                    print("\n\n⚠️  Interrupted by user. Saving progress...")
                    self._save_labels()
                    return
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        # Auto-save after batch
        self._save_labels()
        print(f"\n✅ BATCH COMPLETED!")
        print(f"📊 Progress: {len(self.labels)}/{len(self.segmentation_files)} images labeled")
        self.show_statistics()
    
    def show_statistics(self):
        """Show annotation statistics."""
        if not self.labels:
            print("📊 No labels found.")
            return
        
        label_counts = {0: 0, 1: 0, 2: 0}  # front, left, right
        for label in self.labels.values():
            label_counts[label] += 1
        
        total = len(self.labels)
        label_names = ['FRONT', 'LEFT', 'RIGHT']
        
        print("\n" + "="*50)
        print("📊 ANNOTATION STATISTICS")
        print("="*50)
        print(f"📈 Total labeled: {total}/{len(self.segmentation_files)}")
        print(f"📋 Progress: {total/len(self.segmentation_files)*100:.1f}%")
        print("\n🏷️  Label distribution:")
        for i, name in enumerate(label_names):
            count = label_counts[i]
            percentage = count/total*100 if total > 0 else 0
            bar = "█" * int(percentage/5)  # Simple progress bar
            print(f"   {name:>5}: {count:>3} ({percentage:>5.1f}%) {bar}")
        print("="*50)
    
    def find_unlabeled(self) -> List[str]:
        """Find files that haven't been labeled yet."""
        unlabeled = []
        for filename in self.segmentation_files:
            step_number = self._extract_step_number(filename)
            if str(step_number) not in self.labels:
                unlabeled.append(filename)
        return unlabeled
    
    def continue_annotation(self):
        """Continue annotation from where left off."""
        unlabeled = self.find_unlabeled()
        if not unlabeled:
            print("🎉 All images have been labeled!")
            self.show_statistics()
            return
        
        # Find the index of the first unlabeled image
        first_unlabeled = unlabeled[0]
        start_idx = self.segmentation_files.index(first_unlabeled)
        
        print(f"🔄 Continuing from image {start_idx + 1}/{len(self.segmentation_files)}")
        print(f"📋 Remaining: {len(unlabeled)} images")
        self.annotate_batch(start_idx=start_idx, batch_size=len(unlabeled))


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description='Terminal-based annotation for driving decisions')
    parser.add_argument('data_dir', help='Directory containing segmentation masks')
    parser.add_argument('--labels_file', default='driving_labels.json', help='Labels file name')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for annotation')
    parser.add_argument('--start_idx', type=int, default=0, help='Starting index')
    parser.add_argument('--stats', action='store_true', help='Show statistics only')
    parser.add_argument('--continue', action='store_true', dest='continue_annotation', 
                       help='Continue from where left off')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        print(f"❌ Error: Directory {args.data_dir} does not exist")
        return
    
    annotator = TerminalDataAnnotator(args.data_dir, args.labels_file)
    
    if args.stats:
        annotator.show_statistics()
    elif args.continue_annotation:
        annotator.continue_annotation()
    else:
        annotator.annotate_batch(args.start_idx, args.batch_size)


if __name__ == "__main__":
    main()
