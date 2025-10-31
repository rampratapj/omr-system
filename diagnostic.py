"""
OMR Diagnostic Tool
Use this to debug and troubleshoot OMR sheet detection issues
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class OMRDiagnostic:
    """Diagnose OMR sheet detection issues"""
    
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = None
        self.gray_image = None
        self.processed_image = None
        self.bubbles = []
        
    def load_image(self):
        """Load image"""
        try:
            self.original_image = cv2.imread(self.image_path)
            if self.original_image is None:
                print("❌ ERROR: Cannot read image file")
                return False
            
            height, width = self.original_image.shape[:2]
            print(f"✓ Image loaded: {width}x{height} pixels")
            return True
        except Exception as e:
            print(f"❌ ERROR: {str(e)}")
            return False
    
    def check_image_quality(self):
        """Check image quality"""
        print("\n" + "="*50)
        print("IMAGE QUALITY CHECK")
        print("="*50)
        
        if self.original_image is None:
            print("❌ Image not loaded")
            return
        
        height, width = self.original_image.shape[:2]
        file_size = Path(self.image_path).stat().st_size / (1024*1024)
        
        print(f"Resolution: {width}x{height} pixels")
        print(f"File Size: {file_size:.2f} MB")
        
        # Calculate DPI estimate
        dpi = width / 8.27  # A4 width is 8.27 inches
        print(f"Estimated DPI: {dpi:.0f}")
        
        if dpi < 150:
            print("⚠️  WARNING: Low DPI - may cause detection issues")
            print("   Recommendation: Scan at 300+ DPI")
        else:
            print("✓ DPI adequate")
        
        # Check if image is mostly blank
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        print(f"\nMean Brightness: {mean_brightness:.2f}/255")
        
        if mean_brightness > 240:
            print("⚠️  WARNING: Image too bright - may be blank")
        elif mean_brightness < 20:
            print("⚠️  WARNING: Image too dark - may be over-exposed")
        else:
            print("✓ Brightness OK")
    
    def debug_preprocessing(self):
        """Debug image preprocessing steps"""
        print("\n" + "="*50)
        print("PREPROCESSING DEBUG")
        print("="*50)
        
        if self.original_image is None:
            print("❌ Image not loaded")
            return
        
        # Step 1: Convert to grayscale
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        print("✓ Step 1: Converted to grayscale")
        
        # Step 2: Apply Gaussian blur
        blur = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
        print("✓ Step 2: Applied Gaussian blur")
        
        # Step 3: Binary thresholding
        threshold_value = 127
        _, thresh = cv2.threshold(blur, threshold_value, 255, cv2.THRESH_BINARY)
        self.processed_image = thresh
        print(f"✓ Step 3: Applied threshold (value={threshold_value})")
        
        # Analyze threshold result
        white_pixels = np.sum(thresh == 255)
        black_pixels = np.sum(thresh == 0)
        total_pixels = thresh.size
        
        white_ratio = (white_pixels / total_pixels) * 100
        black_ratio = (black_pixels / total_pixels) * 100
        
        print(f"\n  White pixels: {white_ratio:.1f}%")
        print(f"  Black pixels: {black_ratio:.1f}%")
        
        if white_ratio > 90:
            print("  ⚠️  Almost all white - image may be blank or too bright")
        elif black_ratio > 90:
            print("  ⚠️  Almost all black - image may be too dark")
        else:
            print("  ✓ Good balance of black and white")
    
    def debug_bubble_detection(self):
        """Debug bubble detection"""
        print("\n" + "="*50)
        print("BUBBLE DETECTION DEBUG")
        print("="*50)
        
        if self.processed_image is None:
            print("❌ Image not preprocessed")
            return
        
        # Find contours
        contours, _ = cv2.findContours(
            self.processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        print(f"✓ Found {len(contours)} total contours")
        
        if len(contours) == 0:
            print("❌ No contours found - likely image quality issue")
            return
        
        # Analyze contours
        bubble_list = []
        print("\nContour Analysis (first 20 contours):")
        print("-" * 60)
        print(f"{'#':<4} {'Area':<8} {'Width':<8} {'Height':<8} {'Ratio':<8} {'Status'}")
        print("-" * 60)
        
        for idx, contour in enumerate(contours[:20]):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Check criteria
            is_valid = (200 < area < 5000) and (0.7 < aspect_ratio < 1.3)
            status = "✓ VALID" if is_valid else "✗ INVALID"
            
            print(f"{idx:<4} {area:<8.0f} {w:<8} {h:<8} {aspect_ratio:<8.2f} {status}")
            
            if is_valid:
                bubble_list.append({
                    'x': x, 'y': y, 'w': w, 'h': h,
                    'area': area, 'center': (x + w//2, y + h//2)
                })
        
        print("-" * 60)
        print(f"\nValid bubbles found: {len(bubble_list)}")
        
        if len(bubble_list) == 0:
            print("\n❌ NO VALID BUBBLES DETECTED")
            print("\nPossible causes:")
            print("1. Image quality too low - rescan at 300+ DPI")
            print("2. Bubbles too small or too large")
            print("3. Incorrect threshold value (currently 127)")
            print("4. Poor contrast between bubbles and background")
            print("\nTry adjusting:")
            print("  - BUBBLE_AREA_MIN (currently 200)")
            print("  - BUBBLE_AREA_MAX (currently 5000)")
            print("  - THRESHOLD_VALUE (currently 127)")
            return
        
        self.bubbles = sorted(bubble_list, key=lambda b: (b['y'], b['x']))
        print(f"✓ Detected {len(self.bubbles)} valid bubbles")
        
        # Group bubbles by row
        y_positions = sorted(set(b['y'] for b in self.bubbles))
        groups = []
        current_group = [y_positions[0]]
        
        for y in y_positions[1:]:
            if y - current_group[-1] > 30:
                groups.append(current_group)
                current_group = [y]
            else:
                current_group.append(y)
        
        if current_group:
            groups.append(current_group)
        
        print(f"\nGrouped into {len(groups)} questions (rows)")
        print("First 10 row positions:")
        for i, group in enumerate(groups[:10]):
            print(f"  Question {i+1}: Y={group[0]:.0f}")
    
    def debug_answer_extraction(self):
        """Debug answer extraction"""
        print("\n" + "="*50)
        print("ANSWER EXTRACTION DEBUG")
        print("="*50)
        
        if not self.bubbles:
            print("❌ No bubbles to process")
            return
        
        if self.gray_image is None:
            print("❌ Gray image not available")
            return
        
        # Group bubbles by rows
        y_positions = sorted(set(b['y'] for b in self.bubbles))
        groups = []
        current_group = [y_positions[0]]
        
        for y in y_positions[1:]:
            if y - current_group[-1] > 30:
                groups.append(current_group)
                current_group = [y]
            else:
                current_group.append(y)
        
        if current_group:
            groups.append(current_group)
        
        print(f"Processing {len(groups[:10])} questions (showing first 10):\n")
        print(f"{'Q':<4} {'Opt1':<8} {'Opt2':<8} {'Opt3':<8} {'Opt4':<8} {'Marked':<8}")
        print("-" * 52)
        
        for group_idx, group in enumerate(groups[:10]):
            row_bubbles = [b for b in self.bubbles if any(
                abs(b['y'] - y) < 20 for y in group
            )]
            row_bubbles = sorted(row_bubbles, key=lambda b: b['x'])
            
            darknesses = []
            marked = -1
            max_darkness = 0
            
            for opt_idx, bubble in enumerate(row_bubbles[:4]):
                roi = self.gray_image[
                    bubble['y']:bubble['y']+bubble['h'],
                    bubble['x']:bubble['x']+bubble['w']
                ]
                darkness = 255 - np.mean(roi)
                darknesses.append(darkness)
                
                if darkness > max_darkness:
                    max_darkness = darkness
                    marked = opt_idx
            
            if max_darkness <= 100:
                marked = -1
            
            mark_char = chr(65 + marked) if marked >= 0 else "—"
            
            opt_str = " ".join([f"{d:.0f}" for d in darknesses[:4]])
            print(f"{group_idx+1:<4} {opt_str:<32} {mark_char:<8}")
        
        print("-" * 52)
        print(f"\nDarkness threshold: 100")
        print("(Values > 100 are marked as filled)")
    
    def visualize(self, output_path="diagnostic_output.png"):
        """Create diagnostic visualization"""
        print("\n" + "="*50)
        print("CREATING VISUALIZATION")
        print("="*50)
        
        if self.original_image is None:
            print("❌ Image not loaded")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis('off')
        
        # Grayscale image
        if self.gray_image is not None:
            axes[0, 1].imshow(self.gray_image, cmap='gray')
            axes[0, 1].set_title("Grayscale")
            axes[0, 1].axis('off')
        
        # Threshold image
        if self.processed_image is not None:
            axes[1, 0].imshow(self.processed_image, cmap='gray')
            axes[1, 0].set_title("Threshold Binary")
            axes[1, 0].axis('off')
        
        # Detected bubbles
        if self.bubbles:
            detected_img = self.original_image.copy()
            for bubble in self.bubbles:
                x, y, w, h = bubble['x'], bubble['y'], bubble['w'], bubble['h']
                cv2.rectangle(detected_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            axes[1, 1].imshow(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB))
            axes[1, 1].set_title(f"Detected Bubbles ({len(self.bubbles)})")
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"✓ Visualization saved: {output_path}")
        plt.close()
    
    def full_diagnosis(self, output_path="diagnostic_output.png"):
        """Run complete diagnosis"""
        print("\n" + "="*70)
        print("OMR DIAGNOSTIC TOOL - FULL ANALYSIS")
        print("="*70)
        
        if not self.load_image():
            return
        
        self.check_image_quality()
        self.debug_preprocessing()
        self.debug_bubble_detection()
        self.debug_answer_extraction()
        self.visualize(output_path)
        
        print("\n" + "="*70)
        print("DIAGNOSIS COMPLETE")
        print("="*70)
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 70)
        
        if not self.bubbles:
            print("\n1. NO BUBBLES DETECTED - Try these steps:")
            print("   a) Increase scanner DPI to 300-400")
            print("   b) Ensure OMR sheet is not blank")
            print("   c) Check that bubbles are clearly drawn")
            print("   d) Adjust BUBBLE_AREA_MIN/MAX in config.py:")
            print("      - Decrease MIN to detect smaller bubbles")
            print("      - Increase MAX to detect larger bubbles")
            print("   e) Try different THRESHOLD_VALUE (try 100-150)")
        else:
            print(f"\n✓ {len(self.bubbles)} bubbles detected successfully")
        
        print("\n2. Image Processing Tips:")
        print("   - Use high contrast OMR sheets")
        print("   - Scan in black & white mode (not grayscale)")
        print("   - Ensure proper lighting when scanning")
        print("   - Clean scanner glass before scanning")
        
        print("\n3. Sample OMR Mark Requirements:")
        print("   - Fill 60%+ of bubble")
        print("   - Use HB or 2B pencil")
        print("   - Mark only one option per question")
        print("   - No stray marks outside bubbles")

# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║     OMR DIAGNOSTIC TOOL - TROUBLESHOOTING GUIDE     ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    # Get image path from command line or prompt
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter path to OMR sheet image: ").strip()
    
    # Run diagnosis
    diagnostic = OMRDiagnostic(image_path)
    diagnostic.full_diagnosis()
    
    print("\n" + "="*70)
    print("Check 'diagnostic_output.png' for visual analysis")
    print("="*70)
