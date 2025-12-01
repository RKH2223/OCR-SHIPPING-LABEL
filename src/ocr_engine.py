"""
OCR Engine with multiple strategies for maximum accuracy
"""

import pytesseract
import re
import logging
from typing import List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCREngine:
    """Multi-strategy OCR engine for shipping labels"""
    
    def __init__(self):
        # Multiple PSM (Page Segmentation Mode) configurations
        self.psm_modes = [
            '--psm 6',   # Assume uniform block of text
            '--psm 4',   # Assume single column of text
            '--psm 11',  # Sparse text. Find as much text as possible
            '--psm 3',   # Fully automatic page segmentation
            '--psm 12',  # Sparse text with OSD
        ]
        
        # OCR configurations with different parameters
        self.ocr_configs = [
            '--oem 3 --psm 6',  # Default LSTM + PSM 6
            '--oem 3 --psm 11', # LSTM + Sparse text
            '--oem 1 --psm 6',  # LSTM only
            # '--oem 0 --psm 6',  # Legacy Tesseract
        ]
    
    def extract_text_all_methods(self, image_variants: List) -> List[str]:
        """
        Extract text using all available methods and image variants
        
        Args:
            image_variants: List of preprocessed images
            
        Returns:
            List of extracted text from all combinations
        """
        all_results = []
        
        for idx, image in enumerate(image_variants):
            logger.info(f"Processing image variant {idx + 1}/{len(image_variants)}")
            
            for config in self.ocr_configs:
                try:
                    text = pytesseract.image_to_string(image, config=config)
                    if text.strip():
                        all_results.append(text)
                        logger.debug(f"Extracted text with config '{config}': {len(text)} chars")
                except Exception as e:
                    logger.warning(f"OCR failed with config '{config}': {str(e)}")
                    continue
        
        return all_results
    
    def extract_with_data(self, image_variants: List) -> List[dict]:
        """
        Extract text with detailed data (confidence, bounding boxes)
        
        Args:
            image_variants: List of preprocessed images
            
        Returns:
            List of detailed OCR results
        """
        detailed_results = []
        
        for image in image_variants:
            try:
                # Get detailed data
                data = pytesseract.image_to_data(
                    image, 
                    output_type=pytesseract.Output.DICT,
                    config='--psm 6'
                )
                
                # Extract high-confidence text
                high_conf_text = []
                for i, conf in enumerate(data['conf']):
                    if int(conf) > 30:  # Confidence threshold
                        text = data['text'][i].strip()
                        if text:
                            high_conf_text.append({
                                'text': text,
                                'confidence': conf,
                                'left': data['left'][i],
                                'top': data['top'][i],
                                'width': data['width'][i],
                                'height': data['height'][i]
                            })
                
                if high_conf_text:
                    detailed_results.append(high_conf_text)
                    
            except Exception as e:
                logger.warning(f"Detailed extraction failed: {str(e)}")
                continue
        
        return detailed_results
    
    def combine_results(self, text_results: List[str]) -> str:
        """
        Combine multiple OCR results intelligently
        
        Args:
            text_results: List of text extractions
            
        Returns:
            Best combined result
        """
        if not text_results:
            return ""
        
        # Count frequency of each line across all results
        line_frequency = {}
        
        for text in text_results:
            lines = text.split('\n')
            for line in lines:
                clean_line = line.strip()
                if clean_line:
                    line_frequency[clean_line] = line_frequency.get(clean_line, 0) + 1
        
        # Sort by frequency and length
        sorted_lines = sorted(
            line_frequency.items(), 
            key=lambda x: (x[1], len(x[0])), 
            reverse=True
        )
        
        # Combine most frequent lines
        combined_text = '\n'.join([line for line, freq in sorted_lines])
        
        return combined_text
    
    def extract_text_optimized(self, image_variants: List) -> Tuple[str, float]:
        """
        Extract text with optimization for speed and accuracy balance
        
        Args:
            image_variants: List of preprocessed images
            
        Returns:
            Tuple of (best_text, confidence_score)
        """
        results = []
        
        # Try first two variants with best configs
        for image in image_variants[:3]:
            for config in self.ocr_configs[:2]:
                try:
                    text = pytesseract.image_to_string(image, config=config)
                    if text.strip():
                        results.append(text)
                except:
                    continue
        
        if not results:
            # Fallback: try all combinations
            results = self.extract_text_all_methods(image_variants)
        
        # Combine and return best result
        best_text = self.combine_results(results)
        confidence = self._calculate_confidence(results, best_text)
        
        return best_text, confidence
    
    def _calculate_confidence(self, all_results: List[str], best_result: str) -> float:
        """Calculate confidence score based on consistency"""
        if not all_results or not best_result:
            return 0.0
        
        # Check how many results contain key patterns
        target_pattern_count = sum(
            1 for text in all_results 
            if '_1_' in text.replace(' ', '')
        )
        
        # Confidence based on consistency
        consistency = target_pattern_count / len(all_results) if all_results else 0
        
        # Check if best result has target pattern
        has_target = 1.0 if '_1_' in best_result.replace(' ', '') else 0.5
        
        confidence = (consistency * 0.6 + has_target * 0.4) * 100
        
        return min(confidence, 100.0)