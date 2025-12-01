"""
Target text extraction module
Extracts lines containing "_1_" pattern with high accuracy
"""

import re
import logging
from typing import Optional, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextExtractor:
    """Extract target pattern from OCR text"""
    
    def __init__(self):
        # Pattern variations we're looking for
        self.target_patterns = [
            r'\d{15,}_1_[a-zA-Z0-9]{0,10}',  # Main pattern: digits_1_alphanumeric
            r'\d{15,}_1_',                    # Pattern without suffix
            r'\d{15,}\s*_\s*1\s*_\s*[a-zA-Z0-9]{0,10}',  # With spaces
            r'\d{15,}[-_]1[-_][a-zA-Z0-9]{0,10}',  # With hyphens
            r'\d{15,}[_\-]1[_\-]',           # Flexible separators
        ]
        
        self.min_digit_length = 15  # Minimum length of numeric part
    
    def extract_target_line(self, ocr_text: str) -> Optional[str]:
        """
        Extract line containing _1_ pattern
        
        Args:
            ocr_text: Full OCR extracted text
            
        Returns:
            Extracted target line or None
        """
        if not ocr_text:
            return None
        
        # Try direct pattern matching first
        result = self._direct_pattern_match(ocr_text)
        if result:
            logger.info(f"Found via direct pattern: {result}")
            return result
        
        # Try line-by-line search
        result = self._line_by_line_search(ocr_text)
        if result:
            logger.info(f"Found via line search: {result}")
            return result
        
        # Try fuzzy matching
        result = self._fuzzy_pattern_match(ocr_text)
        if result:
            logger.info(f"Found via fuzzy match: {result}")
            return result
        
        logger.warning("No target pattern found")
        return None
    
    def _direct_pattern_match(self, text: str) -> Optional[str]:
        """Direct regex pattern matching"""
        # Remove all whitespace for matching
        clean_text = re.sub(r'\s+', '', text)
        
        for pattern in self.target_patterns:
            matches = re.findall(pattern, clean_text)
            if matches:
                # Return longest match
                best_match = max(matches, key=len)
                return self._clean_extracted_text(best_match)
        
        return None
    
    def _line_by_line_search(self, text: str) -> Optional[str]:
        """Search each line for the pattern"""
        lines = text.split('\n')
        candidates = []
        
        for line in lines:
            clean_line = re.sub(r'\s+', '', line.strip())
            
            # Check if line contains _1_ or similar pattern
            if '_1_' in clean_line or '_1-' in clean_line or '-1_' in clean_line:
                # Verify it has enough digits
                digits = re.findall(r'\d+', clean_line)
                if digits and len(max(digits, key=len)) >= self.min_digit_length:
                    candidates.append(clean_line)
        
        if candidates:
            # Return longest candidate (most complete)
            best_candidate = max(candidates, key=len)
            return self._clean_extracted_text(best_candidate)
        
        return None
    
    def _fuzzy_pattern_match(self, text: str) -> Optional[str]:
        """
        Fuzzy matching for OCR errors
        Handles common OCR mistakes like:
        - 'l' instead of '1'
        - 'O' instead of '0'
        - Missing or extra characters
        """
        # Replace common OCR errors
        corrected_text = text.replace(' ', '')
        corrected_text = corrected_text.replace('l', '1').replace('I', '1')
        corrected_text = corrected_text.replace('O', '0')
        
        # Try matching again
        for pattern in self.target_patterns:
            matches = re.findall(pattern, corrected_text)
            if matches:
                best_match = max(matches, key=len)
                return self._clean_extracted_text(best_match)
        
        # Look for pattern with flexible separators
        flexible_pattern = r'\d{15,}[_\-\s]{0,2}[1l][_\-\s]{0,2}[a-zA-Z0-9]{0,10}'
        matches = re.findall(flexible_pattern, corrected_text)
        
        if matches:
            best_match = max(matches, key=len)
            # Normalize separators to _1_
            normalized = re.sub(r'[_\-\s]+1[_\-\s]+', '_1_', best_match)
            return self._clean_extracted_text(normalized)
        
        return None
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', '', text)
        
        # Ensure proper format: digits_1_alphanumeric
        # Normalize separators
        cleaned = re.sub(r'[_\-]+1[_\-]+', '_1_', cleaned)
        
        # Remove any leading/trailing non-alphanumeric except underscores
        cleaned = re.sub(r'^[^0-9]+', '', cleaned)
        cleaned = re.sub(r'[^a-zA-Z0-9_]+$', '', cleaned)
        
        return cleaned
    
    def extract_with_confidence(self, ocr_text: str) -> Tuple[Optional[str], float]:
        """
        Extract target with confidence score
        
        Args:
            ocr_text: Full OCR text
            
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        result = self.extract_target_line(ocr_text)
        
        if not result:
            return None, 0.0
        
        # Calculate confidence based on pattern quality
        confidence = self._calculate_extraction_confidence(result)
        
        return result, confidence
    
    def _calculate_extraction_confidence(self, extracted: str) -> float:
        """Calculate confidence score for extraction"""
        score = 0.0
        
        # Check if it matches strict pattern
        if re.match(r'^\d{15,}_1_[a-zA-Z0-9]{1,10}$', extracted):
            score += 60.0
        elif re.match(r'^\d{15,}_1_$', extracted):
            score += 50.0
        else:
            score += 30.0
        
        # Check numeric part length
        digits = re.findall(r'\d+', extracted)
        if digits:
            longest_digit = max(digits, key=len)
            if len(longest_digit) >= 18:
                score += 20.0
            elif len(longest_digit) >= 15:
                score += 15.0
        
        # Check suffix quality
        suffix_match = re.search(r'_1_([a-zA-Z0-9]+)$', extracted)
        if suffix_match:
            suffix = suffix_match.group(1)
            if 1 <= len(suffix) <= 5:
                score += 20.0
            elif len(suffix) > 5:
                score += 10.0
        
        return min(score, 100.0)
    
    def batch_extract(self, ocr_results: List[str]) -> Tuple[Optional[str], float]:
        """
        Extract from multiple OCR results and pick best one
        
        Args:
            ocr_results: List of OCR text results
            
        Returns:
            Best extraction with confidence
        """
        candidates = []
        
        for ocr_text in ocr_results:
            result, confidence = self.extract_with_confidence(ocr_text)
            if result:
                candidates.append((result, confidence))
        
        if not candidates:
            return None, 0.0
        
        # Sort by confidence and pick best
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_result, best_confidence = candidates[0]
        
        # Verify consistency across results
        same_results = sum(1 for r, c in candidates if r == best_result)
        consistency_bonus = (same_results / len(candidates)) * 10
        
        final_confidence = min(best_confidence + consistency_bonus, 100.0)
        
        return best_result, final_confidence