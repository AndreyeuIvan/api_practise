"""
GameTextExtractor: Reusable OCR helper for game screenshots using EasyOCR and OpenCV.

This module provides a Singleton class for extracting text from game screenshots with
configurable preprocessing options using OpenCV.
"""

import threading
from typing import List, Optional, Union
import numpy as np
import cv2


class GameTextExtractor:
    """
    Singleton class for extracting text from game screenshots using EasyOCR.
    
    The EasyOCR model is loaded only once per process. Provides configurable
    preprocessing options including grayscale conversion, denoising, and thresholding.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, langs: Optional[List[str]] = None, use_gpu: Optional[bool] = None):
        """
        Create or return the singleton instance.
        
        Args:
            langs: List of language codes for OCR (default: ['en'])
            use_gpu: Whether to use GPU acceleration. If None, auto-detect using torch.cuda.is_available()
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, langs: Optional[List[str]] = None, use_gpu: Optional[bool] = None):
        """
        Initialize the GameTextExtractor (only runs once due to Singleton pattern).
        
        Args:
            langs: List of language codes for OCR (default: ['en'])
            use_gpu: Whether to use GPU acceleration. If None, auto-detect using torch.cuda.is_available()
        """
        if self._initialized:
            return
            
        self.langs = langs if langs is not None else ['en']
        
        # Determine GPU usage
        if use_gpu is None:
            try:
                import torch
                self.use_gpu = torch.cuda.is_available()
            except ImportError:
                self.use_gpu = False
        else:
            self.use_gpu = use_gpu
        
        # Initialize EasyOCR reader
        import easyocr
        self.reader = easyocr.Reader(self.langs, gpu=self.use_gpu)
        
        self._initialized = True
    
    def extract_text(
        self,
        image: Union[str, np.ndarray],
        preprocess: bool = True,
        grayscale: bool = True,
        denoise: Optional[str] = 'fastnlmeans',
        thresholding: Optional[str] = 'binary',
        threshold_params: Optional[dict] = None
    ) -> List[str]:
        """
        Extract text from an image with optional preprocessing.
        
        Args:
            image: File path (str) or numpy array (ndarray) of the image
            preprocess: Whether to apply preprocessing (default: True)
            grayscale: Whether to convert to grayscale (default: True)
            denoise: Denoising method - 'fastnlmeans', 'gaussian', or None (default: 'fastnlmeans')
            thresholding: Thresholding method - 'binary', 'adaptive', or None (default: 'binary')
            threshold_params: Optional dict of parameters for thresholding
        
        Returns:
            List of recognized text strings
            
        Raises:
            FileNotFoundError: If image path cannot be read
        """
        # Load image if path is provided
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise FileNotFoundError(f"Cannot read image file: {image}")
        else:
            img = image.copy()
        
        # Apply preprocessing if requested
        if preprocess:
            if grayscale:
                img = to_grayscale(img)
            
            if denoise:
                img = denoise_image(img, method=denoise)
            
            if thresholding and grayscale:
                params = threshold_params if threshold_params else {}
                img = threshold_image(img, mode=thresholding, **params)
        
        # Perform OCR
        results = self.reader.readtext(img)
        
        # Extract text from results (format: [(bbox, text, confidence), ...])
        texts = [text for (_, text, _) in results]
        
        return texts


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale.
    
    Args:
        img: Input image (BGR or RGB)
    
    Returns:
        Grayscale image (single channel)
    """
    if len(img.shape) == 2:
        # Already grayscale
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def denoise_image(
    img: np.ndarray,
    method: Optional[str] = 'fastnlmeans',
    h: int = 10,
    templateWindowSize: int = 7,
    searchWindowSize: int = 21,
    ksize: tuple = (5, 5),
    sigmaX: float = 0
) -> np.ndarray:
    """
    Apply denoising to an image.
    
    Args:
        img: Input image
        method: Denoising method - 'fastnlmeans', 'gaussian', or None
        h: Filter strength for fastNlMeans (default: 10)
        templateWindowSize: Template patch size for fastNlMeans (default: 7)
        searchWindowSize: Search area size for fastNlMeans (default: 21)
        ksize: Kernel size for Gaussian blur (default: (5, 5))
        sigmaX: Gaussian kernel standard deviation (default: 0, auto-calculated)
    
    Returns:
        Denoised image
    """
    if method is None:
        return img
    
    if method == 'fastnlmeans':
        if len(img.shape) == 2:
            # Grayscale image
            return cv2.fastNlMeansDenoising(img, None, h, templateWindowSize, searchWindowSize)
        else:
            # Color image
            return cv2.fastNlMeansDenoisingColored(img, None, h, h, templateWindowSize, searchWindowSize)
    
    elif method == 'gaussian':
        return cv2.GaussianBlur(img, ksize, sigmaX)
    
    else:
        raise ValueError(f"Unknown denoising method: {method}. Use 'fastnlmeans', 'gaussian', or None")


def threshold_image(
    gray_img: np.ndarray,
    mode: Optional[str] = 'binary',
    thresh_val: Optional[int] = None,
    max_value: int = 255,
    use_otsu: bool = True,
    invert: bool = False,
    blockSize: int = 11,
    C: int = 2,
    method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
) -> np.ndarray:
    """
    Apply thresholding to a grayscale image.
    
    Args:
        gray_img: Input grayscale image (single channel)
        mode: Thresholding mode - 'binary', 'adaptive', or None
        thresh_val: Threshold value for binary thresholding (default: None, uses OTSU if use_otsu=True)
        max_value: Maximum value for thresholding (default: 255)
        use_otsu: Use Otsu's method for automatic threshold calculation (default: True)
        invert: Invert the threshold (default: False)
        blockSize: Block size for adaptive thresholding (default: 11)
        C: Constant subtracted from mean in adaptive thresholding (default: 2)
        method: Adaptive thresholding method (default: cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    
    Returns:
        Thresholded grayscale image
        
    Raises:
        ValueError: If input is not grayscale
    """
    # Validate input is grayscale
    if len(gray_img.shape) != 2:
        raise ValueError("Input image must be grayscale (single channel)")
    
    if mode is None:
        return gray_img
    
    if mode == 'binary':
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        
        if use_otsu and thresh_val is None:
            thresh_type |= cv2.THRESH_OTSU
            thresh_val = 0  # OTSU will determine the threshold
        elif thresh_val is None:
            thresh_val = 127  # Default threshold value
        
        _, thresholded = cv2.threshold(gray_img, thresh_val, max_value, thresh_type)
        return thresholded
    
    elif mode == 'adaptive':
        thresh_type = cv2.ADAPTIVE_THRESH_MEAN_C if method == cv2.ADAPTIVE_THRESH_MEAN_C else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        binary_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        
        thresholded = cv2.adaptiveThreshold(
            gray_img,
            max_value,
            thresh_type,
            binary_type,
            blockSize,
            C
        )
        return thresholded
    
    else:
        raise ValueError(f"Unknown thresholding mode: {mode}. Use 'binary', 'adaptive', or None")
