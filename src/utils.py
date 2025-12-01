"""
Utility functions for OCR system
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_results(results: Dict[str, Any], output_dir: str = "results") -> str:
    """
    Save extraction results to JSON file
    
    Args:
        results: Dictionary containing extraction results
        output_dir: Directory to save results
        
    Returns:
        Path to saved file
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ocr_result_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Add metadata
    results['saved_at'] = datetime.now().isoformat()
    results['version'] = '1.0.0'
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to: {filepath}")
    return filepath


def load_results(filepath: str) -> Dict[str, Any]:
    """Load results from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_accuracy(predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
    """
    Calculate accuracy metrics
    
    Args:
        predictions: List of predicted target texts
        ground_truth: List of actual target texts
        
    Returns:
        Dictionary with accuracy metrics
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have same length")
    
    total = len(predictions)
    if total == 0:
        return {'accuracy': 0.0, 'precision': 0.0, 'total': 0}
    
    # Exact matches
    exact_matches = sum(
        1 for pred, truth in zip(predictions, ground_truth)
        if pred and pred == truth
    )
    
    # Partial matches (contains the core pattern)
    partial_matches = sum(
        1 for pred, truth in zip(predictions, ground_truth)
        if pred and truth and extract_core_pattern(pred) == extract_core_pattern(truth)
    )
    
    # Calculate metrics
    accuracy = (exact_matches / total) * 100
    partial_accuracy = (partial_matches / total) * 100
    
    # Precision (correct predictions / total predictions made)
    predictions_made = sum(1 for p in predictions if p is not None and p != "")
    precision = (exact_matches / predictions_made * 100) if predictions_made > 0 else 0.0
    
    return {
        'accuracy': round(accuracy, 2),
        'partial_accuracy': round(partial_accuracy, 2),
        'precision': round(precision, 2),
        'exact_matches': exact_matches,
        'partial_matches': partial_matches,
        'total': total,
        'predictions_made': predictions_made
    }


def extract_core_pattern(text: str) -> str:
    """Extract core pattern (numeric + _1_) from text"""
    import re
    
    if not text:
        return ""
    
    # Extract the main numeric part and _1_
    match = re.search(r'(\d{15,}_1_)', text)
    if match:
        return match.group(1)
    
    return ""


def create_accuracy_report(results: List[Dict], output_path: str = "results/accuracy_report.json"):
    """
    Create detailed accuracy report
    
    Args:
        results: List of result dictionaries
        output_path: Path to save report
    """
    total = len(results)
    successful = sum(1 for r in results if r.get('status') == 'SUCCESS')
    failed = sum(1 for r in results if r.get('status') == 'FAILED')
    errors = sum(1 for r in results if 'ERROR' in r.get('status', ''))
    
    # Calculate average confidence
    confidences = [r.get('confidence', 0) for r in results if r.get('status') == 'SUCCESS']
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    # Group by confidence ranges
    high_conf = sum(1 for c in confidences if c >= 80)
    medium_conf = sum(1 for c in confidences if 60 <= c < 80)
    low_conf = sum(1 for c in confidences if c < 60)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_images': total,
            'successful_extractions': successful,
            'failed_extractions': failed,
            'errors': errors,
            'accuracy_percentage': round((successful / total * 100), 2) if total > 0 else 0.0
        },
        'confidence_stats': {
            'average_confidence': round(avg_confidence, 2),
            'high_confidence_count': high_conf,
            'medium_confidence_count': medium_conf,
            'low_confidence_count': low_conf
        },
        'detailed_results': results
    }
    
    # Save report
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Accuracy report saved to: {output_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("ACCURACY REPORT SUMMARY")
    print("="*50)
    print(f"Total Images: {total}")
    print(f"Successful: {successful} ({successful/total*100:.1f}%)")
    print(f"Failed: {failed} ({failed/total*100:.1f}%)")
    print(f"Errors: {errors}")
    print(f"Average Confidence: {avg_confidence:.1f}%")
    print("="*50 + "\n")
    
    return report


def validate_target_format(text: str) -> bool:
    """
    Validate if extracted text matches expected format
    
    Args:
        text: Extracted target text
        
    Returns:
        True if valid format
    """
    import re
    
    if not text:
        return False
    
    # Check if matches expected pattern
    pattern = r'^\d{15,}_1_[a-zA-Z0-9]{0,10}$'
    return bool(re.match(pattern, text))


def format_confidence_display(confidence: float) -> str:
    """Format confidence for display with emoji"""
    if confidence >= 90:
        return f"🟢 {confidence:.1f}% (Excellent)"
    elif confidence >= 75:
        return f"🟡 {confidence:.1f}% (Good)"
    elif confidence >= 50:
        return f"🟠 {confidence:.1f}% (Fair)"
    else:
        return f"🔴 {confidence:.1f}% (Poor)"


def batch_process_directory(
    input_dir: str, 
    output_dir: str = "results",
    ground_truth_file: str = None
) -> Dict[str, Any]:
    """
    Process all images in a directory
    
    Args:
        input_dir: Directory containing images
        output_dir: Directory to save results
        ground_truth_file: Optional JSON file with ground truth labels
        
    Returns:
        Processing results and accuracy metrics
    """
    from src.preprocessing import ImagePreprocessor
    from src.ocr_engine import OCREngine
    from src.text_extraction import TextExtractor
    
    # Initialize components
    preprocessor = ImagePreprocessor()
    ocr_engine = OCREngine()
    extractor = TextExtractor()
    
    # Load ground truth if provided
    ground_truth = {}
    if ground_truth_file and os.path.exists(ground_truth_file):
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
    
    # Get all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = [
        f for f in Path(input_dir).iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    logger.info(f"Found {len(image_files)} images to process")
    
    results = []
    predictions = []
    truths = []
    
    for idx, image_path in enumerate(image_files, 1):
        logger.info(f"Processing {idx}/{len(image_files)}: {image_path.name}")
        
        try:
            # Process image
            variants = preprocessor.preprocess(str(image_path))
            ocr_text, ocr_conf = ocr_engine.extract_text_optimized(variants)
            all_ocr = ocr_engine.extract_text_all_methods(variants)
            target, ext_conf = extractor.batch_extract(all_ocr)
            
            final_conf = (ocr_conf * 0.4 + ext_conf * 0.6) if target else 0.0
            
            result = {
                'filename': image_path.name,
                'target_text': target if target else "NOT FOUND",
                'confidence': round(final_conf, 2),
                'status': 'SUCCESS' if target else 'FAILED'
            }
            
            # Add ground truth comparison if available
            if image_path.name in ground_truth:
                truth = ground_truth[image_path.name]
                result['ground_truth'] = truth
                result['match'] = (target == truth) if target else False
                
                predictions.append(target)
                truths.append(truth)
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {str(e)}")
            results.append({
                'filename': image_path.name,
                'target_text': "ERROR",
                'confidence': 0.0,
                'status': f'ERROR: {str(e)}'
            })
    
    # Calculate accuracy if ground truth available
    accuracy_metrics = None
    if predictions and truths:
        accuracy_metrics = calculate_accuracy(predictions, truths)
    
    # Create comprehensive report
    report = create_accuracy_report(results, os.path.join(output_dir, "accuracy_report.json"))
    
    if accuracy_metrics:
        report['ground_truth_accuracy'] = accuracy_metrics
    
    return report