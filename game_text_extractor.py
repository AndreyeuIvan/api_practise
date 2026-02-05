"""
GameTextExtractor: A reusable OCR helper for game screenshots using EasyOCR and OpenCV.

This module provides a Singleton class for extracting text from game screenshots
using EasyOCR with configurable OpenCV preprocessing options.
"""

import threading
from typing import List, Optional, Union, Any
import numpy as np
import cv2


class GameTextExtractor:
    """
    Singleton class for extracting text from game screenshots using EasyOCR.
    
    The EasyOCR model is loaded only once per process to optimize performance.
    Provides configurable OpenCV preprocessing for improved OCR accuracy.
    
    Example:
        >>> extractor = GameTextExtractor(langs=['en'], use_gpu=False)
        >>> texts = extractor.extract_text('screenshot.png', preprocess=True)
        >>> print(texts)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, langs: Optional[List[str]] = None, use_gpu: Optional[bool] = None):
        """
        Create or return the singleton instance.
        
        Args:
            langs: List of language codes for OCR (default: ['en'])
            use_gpu: Whether to use GPU. If None, auto-detect using torch.cuda.is_available()
        
        Returns:
            The singleton instance of GameTextExtractor
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(GameTextExtractor, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, langs: Optional[List[str]] = None, use_gpu: Optional[bool] = None):
        """
        Initialize the GameTextExtractor.
        
        Args:
            langs: List of language codes for OCR (default: ['en'])
            use_gpu: Whether to use GPU. If None, auto-detect using torch.cuda.is_available()
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
        
        # Import and initialize EasyOCR reader
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
        Extract text from an image using EasyOCR with optional preprocessing.
        
        Args:
            image: Either a file path (str) or numpy array (ndarray)
            preprocess: Whether to apply preprocessing (default: True)
            grayscale: Whether to convert to grayscale (default: True)
            denoise: Denoising method - 'fastnlmeans', 'gaussian', or None (default: 'fastnlmeans')
            thresholding: Thresholding method - 'binary', 'adaptive', or None (default: 'binary')
            threshold_params: Additional parameters for thresholding (optional)
        
        Returns:
            List of recognized text strings
        
        Raises:
            FileNotFoundError: If the provided file path cannot be read
        """
        # Load image
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise FileNotFoundError(f"Cannot read image file: {image}")
        else:
            img = image.copy()
        
        # Apply preprocessing if requested
        if preprocess:
            # Convert to grayscale
            if grayscale:
                img = to_grayscale(img)
            
            # Apply denoising
            if denoise:
                img = denoise_image(img, method=denoise)
            
            # Apply thresholding
            if thresholding:
                if threshold_params is None:
                    threshold_params = {}
                img = threshold_image(img, mode=thresholding, **threshold_params)
        
        # Perform OCR
        results = self.reader.readtext(img)
        
        # Extract text from results
        texts = [text for (_, text, _) in results]
        
        return texts


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale.
    
    Args:
        img: Input image (BGR or grayscale)
    
    Returns:
        Grayscale image (single channel)
    """
    if len(img.shape) == 2:
        # Already grayscale
        return img
    elif len(img.shape) == 3:
        # Convert BGR to grayscale
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")


def denoise_image(
    img: np.ndarray,
    method: Optional[str] = 'fastnlmeans',
    h: int = 10,
    template_window_size: int = 7,
    search_window_size: int = 21,
    kernel_size: tuple = (5, 5),
    sigma_x: int = 0
) -> np.ndarray:
    """
    Apply denoising to an image.
    
    Args:
        img: Input image (grayscale or color)
        method: Denoising method - 'fastnlmeans', 'gaussian', or None
        h: Filter strength for fastNlMeans (default: 10)
        template_window_size: Template patch size for fastNlMeans (default: 7)
        search_window_size: Search area size for fastNlMeans (default: 21)
        kernel_size: Kernel size for Gaussian blur (default: (5, 5))
        sigma_x: Standard deviation for Gaussian blur (default: 0 - calculated from kernel)
    
    Returns:
        Denoised image
    """
    if method is None:
        return img
    
    if method == 'fastnlmeans':
        if len(img.shape) == 2:
            # Grayscale image
            return cv2.fastNlMeansDenoising(
                img,
                None,
                h=h,
                templateWindowSize=template_window_size,
                searchWindowSize=search_window_size
            )
        else:
            # Color image
            return cv2.fastNlMeansDenoisingColored(
                img,
                None,
                h=h,
                hColor=h,
                templateWindowSize=template_window_size,
                searchWindowSize=search_window_size
            )
    elif method == 'gaussian':
        return cv2.GaussianBlur(img, kernel_size, sigma_x)
    else:
        raise ValueError(f"Unknown denoising method: {method}")


def threshold_image(
    gray_img: np.ndarray,
    mode: Optional[str] = 'binary',
    thresh_val: Optional[int] = None,
    max_value: int = 255,
    use_otsu: bool = True,
    invert: bool = False,
    block_size: int = 11,
    c: int = 2,
    method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
) -> np.ndarray:
    """
    Apply thresholding to a grayscale image.
    
    Args:
        gray_img: Input grayscale image
        mode: Thresholding mode - 'binary', 'adaptive', or None
        thresh_val: Threshold value for binary mode (None to use OTSU)
        max_value: Maximum value for thresholding (default: 255)
        use_otsu: Use OTSU algorithm for automatic threshold (default: True)
        invert: Invert the threshold (default: False)
        block_size: Block size for adaptive thresholding (default: 11)
        c: Constant subtracted from mean for adaptive thresholding (default: 2)
        method: Adaptive method - cv2.ADAPTIVE_THRESH_GAUSSIAN_C or MEAN_C
    
    Returns:
        Thresholded grayscale image
    
    Raises:
        ValueError: If input is not grayscale
    """
    # Validate input is grayscale
    if len(gray_img.shape) != 2:
        raise ValueError(f"Input must be grayscale (2D array), got shape: {gray_img.shape}")
    
    if mode is None:
        return gray_img
    
    if mode == 'binary':
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        
        if use_otsu and thresh_val is None:
            thresh_type |= cv2.THRESH_OTSU
            thresh_val = 0
        elif thresh_val is None:
            thresh_val = 127
        
        _, thresholded = cv2.threshold(gray_img, thresh_val, max_value, thresh_type)
        return thresholded
    
    elif mode == 'adaptive':
        thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        
        return cv2.adaptiveThreshold(
            gray_img,
            max_value,
            method,
            thresh_type,
            block_size,
            c
        )
    
    else:
        raise ValueError(f"Unknown thresholding mode: {mode}")
