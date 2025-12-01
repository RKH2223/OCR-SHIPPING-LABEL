"""
Simple OCR Test Script
Just change the IMAGE_PATH below and run: python test.py
"""

import sys
import os
from pathlib import Path

# ADD YOUR IMAGE PATH HERE ⬇️⬇️⬇️
IMAGE_PATH = r"D:\ocr-shipping-label\ReverseWay Bill\reverseWaybill-162533794288078400_1.jpg" 

# Set to True to see full extracted text
SHOW_FULL_TEXT = False

# ============================================
# Don't modify below this line
# ============================================

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocessing import ImagePreprocessor
from ocr_engine import OCREngine
from text_extraction import TextExtractor


def main():
    """Run OCR extraction"""
    
    print("\n" + "="*70)
    print("🔍 OCR SHIPPING LABEL EXTRACTION TEST")
    print("="*70)
    print(f"\n📁 Image: {IMAGE_PATH}")
    
    # Check if file exists
    if not os.path.exists(IMAGE_PATH):
        print(f"\n❌ ERROR: Image file not found!")
        print(f"   Path: {IMAGE_PATH}")
        print(f"\n💡 Make sure the path is correct and file exists")
        return
    
    try:
        # Step 1: Initialize
        print("\n" + "-"*70)
        print("📦 Step 1: Initializing OCR components...")
        preprocessor = ImagePreprocessor()
        ocr_engine = OCREngine()
        extractor = TextExtractor()
        print("   ✅ Components initialized")
        
        # Step 2: Preprocess
        print("\n" + "-"*70)
        print("🔧 Step 2: Preprocessing image...")
        variants = preprocessor.preprocess(IMAGE_PATH)
        print(f"   ✅ Generated {len(variants)} image variants")
        
        # Step 3: OCR
        print("\n" + "-"*70)
        print("🔍 Step 3: Running OCR (this may take 10-30 seconds)...")
        ocr_text, ocr_confidence = ocr_engine.extract_text_optimized(variants)
        print(f"   ✅ OCR completed with {ocr_confidence:.1f}% confidence")
        
        if SHOW_FULL_TEXT:
            print("\n" + "="*70)
            print("📄 FULL EXTRACTED TEXT:")
            print("="*70)
            print(ocr_text)
            print("="*70)
        
        # Step 4: Extract target
        print("\n" + "-"*70)
        print("🎯 Step 4: Extracting target pattern (containing '_1_')...")
        all_ocr_results = ocr_engine.extract_text_all_methods(variants)
        target_text, extraction_confidence = extractor.batch_extract(all_ocr_results)
        
        # Calculate final confidence
        final_confidence = (ocr_confidence * 0.4 + extraction_confidence * 0.6)
        
        # Display results
        print("\n" + "="*70)
        print("🎉 FINAL RESULTS")
        print("="*70)
        
        if target_text:
            print(f"\n✅ SUCCESS! Target pattern found!\n")
            print(f"📌 EXTRACTED TEXT:")
            print(f"   ╔{'═'*60}╗")
            print(f"   ║  {target_text:<58}║")
            print(f"   ╚{'═'*60}╝\n")
            
            print(f"📊 CONFIDENCE METRICS:")
            print(f"   • OCR Confidence:        {ocr_confidence:6.1f}%")
            print(f"   • Extraction Confidence: {extraction_confidence:6.1f}%")
            print(f"   • Overall Confidence:    {final_confidence:6.1f}%")
            
            # Confidence rating
            if final_confidence >= 90:
                rating = "🟢 EXCELLENT"
            elif final_confidence >= 75:
                rating = "🟡 GOOD"
            elif final_confidence >= 50:
                rating = "🟠 FAIR"
            else:
                rating = "🔴 POOR"
            print(f"   • Quality Rating:        {rating}")
            
            # Validate format
            import re
            is_valid = bool(re.match(r'^\d{15,}_1_[a-zA-Z0-9]{0,10}$', target_text))
            print(f"\n✓ Format Validation: {'PASS ✅' if is_valid else 'FAIL ❌'}")
            
            # Expected format
            print(f"\n📋 Expected Format: <15+ digits>_1_<alphanumeric>")
            print(f"   Example: 163233702292313922_1_lWV")
            
        else:
            print(f"\n❌ FAILED: Target pattern '_1_' not found\n")
            print(f"💡 SUGGESTIONS:")
            print(f"   1. Check if the image contains the target pattern")
            print(f"   2. Ensure image quality is good (not blurry)")
            print(f"   3. Set SHOW_FULL_TEXT = True to see what was extracted")
            print(f"   4. Try with a different image")
            
            if not SHOW_FULL_TEXT:
                print(f"\n📄 First 500 characters of extracted text:")
                print("-"*70)
                print(ocr_text[:500] if ocr_text else "No text extracted")
                print("-"*70)
        
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERROR OCCURRED:")
        print(f"   {str(e)}")
        print(f"\n🐛 FULL ERROR DETAILS:")
        import traceback
        traceback.print_exc()
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()