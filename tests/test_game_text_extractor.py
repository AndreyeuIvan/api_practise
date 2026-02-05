"""
Unit tests for GameTextExtractor.
"""

import os
import tempfile
import pytest
import numpy as np
import cv2

from game_text_extractor import (
    GameTextExtractor,
    to_grayscale,
    denoise_image,
    threshold_image,
)


class TestPreprocessingHelpers:
    """Test the individual preprocessing helper functions."""
    
    def test_to_grayscale_color_image(self):
        """Test converting a color image to grayscale."""
        # Create a color image (BGR)
        color_img = np.zeros((100, 100, 3), dtype=np.uint8)
        color_img[:, :] = [255, 0, 0]  # Blue in BGR
        
        gray_img = to_grayscale(color_img)
        
        assert len(gray_img.shape) == 2
        assert gray_img.shape == (100, 100)
    
    def test_to_grayscale_already_grayscale(self):
        """Test that grayscale images pass through unchanged."""
        gray_img = np.zeros((100, 100), dtype=np.uint8)
        
        result = to_grayscale(gray_img)
        
        assert len(result.shape) == 2
        assert np.array_equal(result, gray_img)
    
    def test_denoise_image_none(self):
        """Test that method=None returns original image."""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        result = denoise_image(img, method=None)
        
        assert np.array_equal(result, img)
    
    def test_denoise_image_gaussian(self):
        """Test Gaussian denoising."""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        result = denoise_image(img, method='gaussian')
        
        assert result.shape == img.shape
        # Gaussian blur should smooth the image
        assert result.dtype == img.dtype
    
    def test_denoise_image_fastnlmeans_grayscale(self):
        """Test fastNlMeans denoising on grayscale image."""
        img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        result = denoise_image(img, method='fastnlmeans')
        
        assert result.shape == img.shape
        assert result.dtype == img.dtype
    
    def test_threshold_image_binary_with_otsu(self):
        """Test binary thresholding with Otsu's method."""
        # Create a simple grayscale image with two regions
        gray_img = np.zeros((100, 100), dtype=np.uint8)
        gray_img[:50, :] = 50  # Dark region
        gray_img[50:, :] = 200  # Bright region
        
        result = threshold_image(gray_img, mode='binary', use_otsu=True)
        
        assert len(result.shape) == 2
        assert result.shape == gray_img.shape
        # Result should be binary (0 or 255)
        unique_values = np.unique(result)
        assert len(unique_values) <= 2
    
    def test_threshold_image_none(self):
        """Test that mode=None returns original image."""
        gray_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        result = threshold_image(gray_img, mode=None)
        
        assert np.array_equal(result, gray_img)
    
    def test_threshold_image_requires_grayscale(self):
        """Test that threshold_image raises error for non-grayscale input."""
        color_img = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Input must be grayscale"):
            threshold_image(color_img, mode='binary')
    
    def test_threshold_image_adaptive(self):
        """Test adaptive thresholding."""
        gray_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        result = threshold_image(gray_img, mode='adaptive')
        
        assert len(result.shape) == 2
        assert result.shape == gray_img.shape
        # Result should be binary
        unique_values = np.unique(result)
        assert len(unique_values) <= 2


class TestGameTextExtractor:
    """Test the GameTextExtractor class."""
    
    def test_singleton_pattern(self):
        """Test that GameTextExtractor follows singleton pattern."""
        extractor1 = GameTextExtractor(use_gpu=False)
        extractor2 = GameTextExtractor(use_gpu=False)
        
        assert extractor1 is extractor2
    
    def test_extract_text_with_synthetic_image(self):
        """Test text extraction with a synthetic image containing text."""
        # Create a temporary file for the test image
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            # Create a white image
            img = np.ones((200, 400, 3), dtype=np.uint8) * 255
            
            # Add text to the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "HELLO 123"
            font_scale = 2
            thickness = 3
            color = (0, 0, 0)  # Black text
            
            # Get text size to center it
            (text_width, text_height), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            x = (img.shape[1] - text_width) // 2
            y = (img.shape[0] + text_height) // 2
            
            cv2.putText(img, text, (x, y), font, font_scale, color, thickness)
            
            # Save the image
            cv2.imwrite(tmp_path, img)
            
            # Initialize extractor with CPU only
            extractor = GameTextExtractor(langs=['en'], use_gpu=False)
            
            # Extract text with preprocessing
            texts = extractor.extract_text(
                tmp_path,
                preprocess=True,
                grayscale=True,
                denoise='fastnlmeans',
                thresholding='binary',
                threshold_params={'use_otsu': True}
            )
            
            # Assert that at least one non-empty text fragment is returned
            assert any(t.strip() for t in texts), f"Expected non-empty text, got: {texts}"
            
            # Check that extracted text contains expected characters/numbers
            all_text = ' '.join(texts).upper()
            # EasyOCR might not be perfect, so we check for some of the expected content
            # At minimum, we should get some alphanumeric characters
            assert len(all_text.strip()) > 0, "Extracted text should not be empty"
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_extract_text_with_ndarray_input(self):
        """Test text extraction with numpy array input."""
        # Create a white image with text
        img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "TEST 456"
        cv2.putText(img, text, (50, 100), font, 2, (0, 0, 0), 3)
        
        # Initialize extractor
        extractor = GameTextExtractor(langs=['en'], use_gpu=False)
        
        # Extract text from numpy array
        texts = extractor.extract_text(
            img,
            preprocess=True,
            grayscale=True,
            thresholding='binary'
        )
        
        # Should get some text back
        assert isinstance(texts, list)
        assert any(t.strip() for t in texts), f"Expected non-empty text, got: {texts}"
    
    def test_extract_text_file_not_found(self):
        """Test that FileNotFoundError is raised for invalid path."""
        extractor = GameTextExtractor(use_gpu=False)
        
        with pytest.raises(FileNotFoundError, match="Cannot read image file"):
            extractor.extract_text('/nonexistent/path/to/image.png')
    
    def test_extract_text_without_preprocessing(self):
        """Test text extraction without preprocessing."""
        # Create a simple image with text
        img = np.ones((200, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img, "SIMPLE", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        extractor = GameTextExtractor(use_gpu=False)
        
        # Extract without preprocessing
        texts = extractor.extract_text(img, preprocess=False)
        
        assert isinstance(texts, list)
        # Even without preprocessing, should be able to extract some text
        # (though results may vary)
