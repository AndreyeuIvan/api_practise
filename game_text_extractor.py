"""
GameTextExtractor: A reusable OCR helper for game screenshots using EasyOCR and OpenCV.

This module provides a singleton class for extracting text from game screenshots with
configurable preprocessing options using OpenCV and EasyOCR.
"""

import threading
from typing import List, Optional, Union
import numpy as np

try:
    import cv2
except ImportError:
    raise ImportError("opencv-python is required. Install it with: pip install opencv-python")

try:
    import easyocr
except ImportError:
    raise ImportError("easyocr is required. Install it with: pip install easyocr")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert an image to grayscale.
    
    Args:
        img: Input image (BGR or grayscale).
        
    Returns:
        Single-channel grayscale image.
    """
    if len(img.shape) == 2:
        # Already grayscale
        return img
    elif len(img.shape) == 3 and img.shape[2] == 3:
        # Convert BGR to grayscale
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")


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
    Denoise an image using various methods.
    
    Args:
        img: Input image (grayscale or color).
        method: Denoising method ('fastnlmeans', 'gaussian', or None).
        h: Filter strength for fastNlMeans (higher removes more noise but also removes detail).
        templateWindowSize: Size in pixels of template patch for fastNlMeans.
        searchWindowSize: Size in pixels of search window for fastNlMeans.
        ksize: Kernel size for Gaussian blur.
        sigmaX: Gaussian kernel standard deviation in X direction.
        
    Returns:
        Denoised image.
    """
    if method is None:
        return img
    
    if method == 'fastnlmeans':
        if len(img.shape) == 2:
            # Grayscale image
            return cv2.fastNlMeansDenoising(
                img,
                h=h,
                templateWindowSize=templateWindowSize,
                searchWindowSize=searchWindowSize
            )
        else:
            # Color image
            return cv2.fastNlMeansDenoisingColored(
                img,
                h=h,
                hColor=h,
                templateWindowSize=templateWindowSize,
                searchWindowSize=searchWindowSize
            )
    elif method == 'gaussian':
        return cv2.GaussianBlur(img, ksize, sigmaX)
    else:
        raise ValueError(f"Unknown denoising method: {method}. Use 'fastnlmeans', 'gaussian', or None.")


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
        gray_img: Input grayscale image.
        mode: Thresholding mode ('binary', 'adaptive', or None).
        thresh_val: Threshold value for binary thresholding (ignored if use_otsu=True).
        max_value: Maximum value to use with THRESH_BINARY.
        use_otsu: Use Otsu's method for automatic threshold calculation (binary mode only).
        invert: Invert the threshold (use THRESH_BINARY_INV instead of THRESH_BINARY).
        blockSize: Size of neighborhood for adaptive thresholding (must be odd).
        C: Constant subtracted from mean/weighted mean for adaptive thresholding.
        method: Adaptive threshold method (cv2.ADAPTIVE_THRESH_MEAN_C or cv2.ADAPTIVE_THRESH_GAUSSIAN_C).
        
    Returns:
        Thresholded grayscale image.
        
    Raises:
        ValueError: If input is not grayscale or parameters are invalid.
    """
    if len(gray_img.shape) != 2:
        raise ValueError(f"Input must be grayscale (2D array). Got shape: {gray_img.shape}")
    
    if mode is None:
        return gray_img
    
    if mode == 'binary':
        threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        if use_otsu:
            threshold_type |= cv2.THRESH_OTSU
            thresh_val = 0  # Ignored when using Otsu
        elif thresh_val is None:
            thresh_val = 127  # Default threshold value
        
        _, thresholded = cv2.threshold(gray_img, thresh_val, max_value, threshold_type)
        return thresholded
    
    elif mode == 'adaptive':
        threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        return cv2.adaptiveThreshold(
            gray_img,
            max_value,
            method,
            threshold_type,
            blockSize,
            C
        )
    else:
        raise ValueError(f"Unknown threshold mode: {mode}. Use 'binary', 'adaptive', or None.")


class GameTextExtractor:
    """
    Singleton class for extracting text from game screenshots using EasyOCR.
    
    The EasyOCR model is loaded only once per process. Provides configurable
    preprocessing options using OpenCV before text extraction.
    
    Example:
        extractor = GameTextExtractor(langs=['en'], use_gpu=False)
        texts = extractor.extract_text('screenshot.png', preprocess=True)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, langs: Optional[List[str]] = None, use_gpu: Optional[bool] = None):
        """
        Ensure only one instance of GameTextExtractor exists.
        
        Args:
            langs: List of language codes (default: ['en']).
            use_gpu: Whether to use GPU. If None, auto-detect using torch.cuda.is_available().
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(GameTextExtractor, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, langs: Optional[List[str]] = None, use_gpu: Optional[bool] = None):
        """
        Initialize the GameTextExtractor (called only once due to singleton pattern).
        
        Args:
            langs: List of language codes (default: ['en']).
            use_gpu: Whether to use GPU. If None, auto-detect using torch.cuda.is_available().
        """
        if self._initialized:
            return
        
        self.langs = langs if langs is not None else ['en']
        
        # Determine GPU usage
        if use_gpu is None:
            if TORCH_AVAILABLE:
                self.use_gpu = torch.cuda.is_available()
            else:
                self.use_gpu = False
        else:
            self.use_gpu = use_gpu
        
        # Initialize EasyOCR reader
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
            image: File path (str) or image array (np.ndarray).
            preprocess: Whether to apply preprocessing.
            grayscale: Convert to grayscale during preprocessing.
            denoise: Denoising method ('fastnlmeans', 'gaussian', or None).
            thresholding: Thresholding mode ('binary', 'adaptive', or None).
            threshold_params: Additional parameters for threshold_image().
            
        Returns:
            List of recognized text strings.
            
        Raises:
            FileNotFoundError: If the image path cannot be read.
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
            # Convert to grayscale
            if grayscale:
                img = to_grayscale(img)
            
            # Denoise
            if denoise is not None:
                img = denoise_image(img, method=denoise)
            
            # Apply thresholding
            if thresholding is not None:
                # Ensure image is grayscale for thresholding
                if len(img.shape) == 3:
                    img = to_grayscale(img)
                
                thresh_params = threshold_params if threshold_params is not None else {}
                img = threshold_image(img, mode=thresholding, **thresh_params)
        
        # Perform OCR
        results = self.reader.readtext(img)
        
        # Extract text from results
        # EasyOCR returns list of tuples: (bbox, text, confidence)
        texts = [text for (_, text, _) in results]
        
        return texts
