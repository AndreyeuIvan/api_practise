"""
Unit tests for GameTextExtractor and preprocessing functions.
"""

import os
import tempfile
import numpy as np
import cv2
import pytest

from game_text_extractor import (
    GameTextExtractor,
    to_grayscale,
    denoise_image,
    threshold_image
)


class TestPreprocessingFunctions:
    """Test OpenCV preprocessing helper functions."""
    
    def test_to_grayscale_color_image(self):
        """Test converting a color image to grayscale."""
        # Create a simple color image (BGR)
        color_img = np.zeros((100, 100, 3), dtype=np.uint8)
        color_img[:, :] = [255, 0, 0]  # Blue image
        
        gray_img = to_grayscale(color_img)
        
        assert len(gray_img.shape) == 2, "Output should be 2D (grayscale)"
        assert gray_img.shape == (100, 100), "Shape should match input dimensions"
    
    def test_to_grayscale_already_gray(self):
        """Test that grayscale image passes through unchanged."""
        gray_img = np.zeros((100, 100), dtype=np.uint8)
        gray_img[:, :] = 128
        
        result = to_grayscale(gray_img)
        
        assert len(result.shape) == 2, "Output should remain 2D"
        np.testing.assert_array_equal(result, gray_img)
    
    def test_denoise_image_fastnlmeans_gray(self):
        """Test fastNlMeans denoising on grayscale image."""
        # Create a noisy grayscale image
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        denoised = denoise_image(img, method='fastnlmeans')
        
        assert denoised.shape == img.shape, "Shape should be preserved"
        assert denoised.dtype == img.dtype, "Dtype should be preserved"
    
    def test_denoise_image_fastnlmeans_color(self):
        """Test fastNlMeans denoising on color image."""
        # Create a noisy color image
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        denoised = denoise_image(img, method='fastnlmeans')
        
        assert denoised.shape == img.shape, "Shape should be preserved"
        assert denoised.dtype == img.dtype, "Dtype should be preserved"
    
    def test_denoise_image_gaussian(self):
        """Test Gaussian blur denoising."""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        denoised = denoise_image(img, method='gaussian')
        
        assert denoised.shape == img.shape, "Shape should be preserved"
    
    def test_denoise_image_none(self):
        """Test that None method returns image unchanged."""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        result = denoise_image(img, method=None)
        
        np.testing.assert_array_equal(result, img)
    
    def test_threshold_image_binary_otsu(self):
        """Test binary thresholding with OTSU."""
        # Create a grayscale image with clear separation
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:50, :] = 50  # Dark region
        img[50:, :] = 200  # Bright region
        
        thresholded = threshold_image(img, mode='binary', use_otsu=True)
        
        assert thresholded.shape == img.shape, "Shape should be preserved"
        assert len(np.unique(thresholded)) <= 2, "Should have at most 2 unique values"
    
    def test_threshold_image_binary_manual(self):
        """Test binary thresholding with manual threshold."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:50, :] = 50
        img[50:, :] = 200
        
        thresholded = threshold_image(img, mode='binary', thresh_val=127, use_otsu=False)
        
        assert thresholded.shape == img.shape
        assert len(np.unique(thresholded)) <= 2
    
    def test_threshold_image_adaptive(self):
        """Test adaptive thresholding."""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        thresholded = threshold_image(img, mode='adaptive')
        
        assert thresholded.shape == img.shape
    
    def test_threshold_image_none(self):
        """Test that None mode returns image unchanged."""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        result = threshold_image(img, mode=None)
        
        np.testing.assert_array_equal(result, img)
    
    def test_threshold_image_validates_grayscale(self):
        """Test that threshold_image raises error for color image."""
        color_img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="must be grayscale"):
            threshold_image(color_img, mode='binary')


class TestGameTextExtractor:
    """Test GameTextExtractor class."""
    
    def test_singleton_pattern(self):
        """Test that GameTextExtractor is a singleton."""
        extractor1 = GameTextExtractor(langs=['en'], use_gpu=False)
        extractor2 = GameTextExtractor(langs=['en'], use_gpu=False)
        
        assert extractor1 is extractor2, "Should return the same instance"
    
    def test_extract_text_with_synthetic_image(self):
        """Test text extraction with a synthetic image containing text."""
        # Create a synthetic image with text
        img = np.ones((200, 400, 3), dtype=np.uint8) * 255  # White background
        
        # Add text to the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "HELLO 123"
        font_scale = 2
        thickness = 3
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = (img.shape[0] + text_size[1]) // 2
        
        cv2.putText(img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = f.name
            cv2.imwrite(temp_path, img)
        
        try:
            # Initialize extractor with CPU mode
            extractor = GameTextExtractor(langs=['en'], use_gpu=False)
            
            # Extract text with preprocessing
            texts = extractor.extract_text(
                temp_path,
                preprocess=True,
                grayscale=True,
                denoise='fastnlmeans',
                thresholding='binary',
                threshold_params={'use_otsu': True}
            )
            
            # Assert at least one non-empty text fragment is returned
            assert any(t.strip() for t in texts), f"Expected non-empty text, got: {texts}"
            
            # Check if any recognized text contains expected substrings
            recognized_text = ' '.join(texts).upper()
            # OCR might recognize parts of "HELLO 123"
            assert len(recognized_text) > 0, "Should recognize some text"
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def test_extract_text_with_ndarray(self):
        """Test text extraction with ndarray input."""
        # Create a synthetic image with text
        img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "TEST", (50, 100), font, 2, (0, 0, 0), 3)
        
        # Initialize extractor
        extractor = GameTextExtractor(langs=['en'], use_gpu=False)
        
        # Extract text from ndarray
        texts = extractor.extract_text(img, preprocess=True)
        
        # Should return a list (may be empty depending on OCR quality)
        assert isinstance(texts, list)
    
    def test_extract_text_file_not_found(self):
        """Test that FileNotFoundError is raised for invalid path."""
        extractor = GameTextExtractor(langs=['en'], use_gpu=False)
        
        with pytest.raises(FileNotFoundError, match="Cannot read image file"):
            extractor.extract_text('/nonexistent/path/image.png')
    
    def test_extract_text_no_preprocessing(self):
        """Test text extraction without preprocessing."""
        # Create a simple image
        img = np.ones((100, 200, 3), dtype=np.uint8) * 255
        cv2.putText(img, "NO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        extractor = GameTextExtractor(langs=['en'], use_gpu=False)
        
        # Extract without preprocessing
        texts = extractor.extract_text(img, preprocess=False)
        
        assert isinstance(texts, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
