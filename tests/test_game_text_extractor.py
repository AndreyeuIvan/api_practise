"""
Unit tests for GameTextExtractor OCR helper.
"""

import os
import tempfile
import numpy as np
import cv2
import pytest
from game_text_extractor import GameTextExtractor, to_grayscale, denoise_image, threshold_image


class TestGameTextExtractor:
    """Test suite for GameTextExtractor class."""
    
    def test_singleton_pattern(self):
        """Test that GameTextExtractor follows singleton pattern."""
        instance1 = GameTextExtractor(use_gpu=False)
        instance2 = GameTextExtractor(use_gpu=False)
        assert instance1 is instance2, "GameTextExtractor should be a singleton"
    
    def test_extract_text_with_synthetic_image(self):
        """Test text extraction with a synthetic image containing text."""
        # Create a white image
        img = np.ones((200, 600, 3), dtype=np.uint8) * 255
        
        # Add text to the image
        text = "HELLO 123"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 3
        color = (0, 0, 0)  # Black text
        
        # Get text size and center it
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (img.shape[1] - text_size[0]) // 2
        text_y = (img.shape[0] + text_size[1]) // 2
        
        cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            cv2.imwrite(tmp_path, img)
        
        try:
            # Initialize GameTextExtractor with CPU mode
            extractor = GameTextExtractor(use_gpu=False)
            
            # Extract text with preprocessing
            texts = extractor.extract_text(
                tmp_path,
                preprocess=True,
                grayscale=True,
                denoise='fastnlmeans',
                thresholding='binary',
                threshold_params={'use_otsu': True}
            )
            
            # Assert at least one non-empty text fragment is returned
            assert any(t.strip() for t in texts), f"Expected non-empty text, got: {texts}"
            
            # Check if extracted text contains expected content
            extracted_text = ' '.join(texts).upper()
            assert 'HELLO' in extracted_text or '123' in extracted_text or 'HEL' in extracted_text, \
                f"Expected to find 'HELLO' or '123' in extracted text, got: {texts}"
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def test_extract_text_with_ndarray(self):
        """Test text extraction with numpy array input."""
        # Create a white image with text
        img = np.ones((200, 600, 3), dtype=np.uint8) * 255
        text = "TEST 456"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, (50, 100), font, 2, (0, 0, 0), 3)
        
        # Initialize extractor
        extractor = GameTextExtractor(use_gpu=False)
        
        # Extract text from numpy array
        texts = extractor.extract_text(
            img,
            preprocess=True,
            grayscale=True,
            denoise=None,
            thresholding='binary'
        )
        
        # Assert text was extracted
        assert any(t.strip() for t in texts), f"Expected non-empty text, got: {texts}"
    
    def test_extract_text_file_not_found(self):
        """Test that FileNotFoundError is raised for invalid file path."""
        extractor = GameTextExtractor(use_gpu=False)
        
        with pytest.raises(FileNotFoundError):
            extractor.extract_text("/nonexistent/path/to/image.png")
    
    def test_extract_text_without_preprocessing(self):
        """Test text extraction without preprocessing."""
        # Create a simple image
        img = np.ones((100, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img, "ABC", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            cv2.imwrite(tmp_path, img)
        
        try:
            extractor = GameTextExtractor(use_gpu=False)
            texts = extractor.extract_text(tmp_path, preprocess=False)
            
            # Should return a list (may be empty depending on OCR quality)
            assert isinstance(texts, list), "Expected list of texts"
        
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestPreprocessingFunctions:
    """Test suite for preprocessing helper functions."""
    
    def test_to_grayscale_color_image(self):
        """Test grayscale conversion for color image."""
        color_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        gray_img = to_grayscale(color_img)
        
        assert len(gray_img.shape) == 2, "Output should be single channel"
        assert gray_img.shape == (100, 100), "Output shape should match input dimensions"
    
    def test_to_grayscale_already_grayscale(self):
        """Test grayscale conversion for already grayscale image."""
        gray_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = to_grayscale(gray_img)
        
        assert len(result.shape) == 2, "Output should be single channel"
        np.testing.assert_array_equal(result, gray_img, "Should return same image")
    
    def test_denoise_fastnlmeans_grayscale(self):
        """Test fastNlMeans denoising on grayscale image."""
        gray_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        denoised = denoise_image(gray_img, method='fastnlmeans')
        
        assert denoised.shape == gray_img.shape, "Output shape should match input"
        assert len(denoised.shape) == 2, "Output should be grayscale"
    
    def test_denoise_fastnlmeans_color(self):
        """Test fastNlMeans denoising on color image."""
        color_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        denoised = denoise_image(color_img, method='fastnlmeans')
        
        assert denoised.shape == color_img.shape, "Output shape should match input"
        assert len(denoised.shape) == 3, "Output should be color"
    
    def test_denoise_gaussian(self):
        """Test Gaussian denoising."""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        denoised = denoise_image(img, method='gaussian')
        
        assert denoised.shape == img.shape, "Output shape should match input"
    
    def test_denoise_none(self):
        """Test that None method returns original image."""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = denoise_image(img, method=None)
        
        np.testing.assert_array_equal(result, img, "Should return original image")
    
    def test_denoise_invalid_method(self):
        """Test that invalid method raises ValueError."""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        with pytest.raises(ValueError):
            denoise_image(img, method='invalid')
    
    def test_threshold_binary_with_otsu(self):
        """Test binary thresholding with Otsu's method."""
        gray_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        thresholded = threshold_image(gray_img, mode='binary', use_otsu=True)
        
        assert thresholded.shape == gray_img.shape, "Output shape should match input"
        assert len(thresholded.shape) == 2, "Output should be grayscale"
        # Check that values are binary (0 or 255)
        unique_values = np.unique(thresholded)
        assert len(unique_values) <= 2, "Binary threshold should produce at most 2 unique values"
    
    def test_threshold_binary_manual(self):
        """Test binary thresholding with manual threshold."""
        gray_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        thresholded = threshold_image(gray_img, mode='binary', thresh_val=127, use_otsu=False)
        
        assert thresholded.shape == gray_img.shape, "Output shape should match input"
        unique_values = np.unique(thresholded)
        assert len(unique_values) <= 2, "Binary threshold should produce at most 2 unique values"
    
    def test_threshold_adaptive(self):
        """Test adaptive thresholding."""
        gray_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        thresholded = threshold_image(gray_img, mode='adaptive')
        
        assert thresholded.shape == gray_img.shape, "Output shape should match input"
    
    def test_threshold_none(self):
        """Test that None mode returns original image."""
        gray_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = threshold_image(gray_img, mode=None)
        
        np.testing.assert_array_equal(result, gray_img, "Should return original image")
    
    def test_threshold_invalid_input(self):
        """Test that color image raises ValueError."""
        color_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError):
            threshold_image(color_img, mode='binary')
    
    def test_threshold_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        gray_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        with pytest.raises(ValueError):
            threshold_image(gray_img, mode='invalid')
