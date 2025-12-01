"""
Image Preprocessing Module for OCR
Handles various image qualities and orientations
"""

import cv2
import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Advanced image preprocessing for optimal OCR results"""
    
    def __init__(self):
        self.processed_images = []
    
    def preprocess(self, image_path):
        """
        Apply multiple preprocessing techniques and return all variants
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of preprocessed image variants
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            logger.info(f"Processing image: {image_path}")
            
            # Store all variants
            variants = []
            
            # Variant 1: Standard grayscale with OTSU thresholding
            variants.append(self._standard_preprocessing(img.copy()))
            
            # Variant 2: Enhanced contrast
            variants.append(self._enhanced_contrast(img.copy()))
            
            # Variant 3: Adaptive thresholding
            variants.append(self._adaptive_threshold(img.copy()))
            
            # Variant 4: Denoised version
            variants.append(self._denoise_and_sharpen(img.copy()))
            
            # Variant 5: Binary with morphological operations
            variants.append(self._morphological_preprocessing(img.copy()))
            
            self.processed_images = variants
            return variants
            
        except Exception as e:
            logger.error(f"Preprocessing error: {str(e)}")
            raise
    
    def _standard_preprocessing(self, img):
        """Standard grayscale + OTSU thresholding"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    
    def _enhanced_contrast(self, img):
        """CLAHE for contrast enhancement"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    
    def _adaptive_threshold(self, img):
        """Adaptive thresholding for varying lighting"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Blur to reduce noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        adaptive = cv2.adaptiveThreshold(
            blur, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 2
        )
        return adaptive
    
    def _denoise_and_sharpen(self, img):
        """Denoise and sharpen for better text clarity"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        
        # Threshold
        _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
    
    def _morphological_preprocessing(self, img):
        """Morphological operations to enhance text"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        
        return morph
    
    def rotate_if_needed(self, image):
        """Detect and correct image orientation"""
        try:
            # Detect text orientation using contours
            coords = np.column_stack(np.where(image > 0))
            angle = cv2.minAreaRect(coords)[-1]
            
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            # Rotate if needed
            if abs(angle) > 0.5:
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(
                    image, M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE
                )
                return rotated
            
            return image
        except:
            return image
    
    def resize_for_ocr(self, image, target_height=1000):
        """Resize image to optimal size for OCR"""
        height, width = image.shape[:2]
        
        if height < target_height:
            # Upscale small images
            scale = target_height / height
            new_width = int(width * scale)
            resized = cv2.resize(
                image, 
                (new_width, target_height), 
                interpolation=cv2.INTER_CUBIC
            )
            return resized
        
        return image