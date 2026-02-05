# API Practice - Game Text Extractor

This repository includes a Django REST API for game management and a reusable OCR helper for extracting text from game screenshots.

## GameTextExtractor

The `GameTextExtractor` is a Singleton class that provides text extraction from game screenshots using EasyOCR with configurable OpenCV preprocessing options. The EasyOCR model is loaded only once per process to optimize performance.

### Features

- **Singleton Pattern**: Ensures the EasyOCR model is loaded only once
- **Flexible Input**: Accepts both file paths and numpy arrays
- **Configurable Preprocessing**: Optional grayscale conversion, denoising (fastNlMeans/Gaussian), and thresholding (binary/adaptive)
- **GPU Support**: Automatically detects CUDA availability or can be explicitly configured

### Usage

```python
from game_text_extractor import GameTextExtractor

# Initialize the extractor (only happens once due to Singleton pattern)
extractor = GameTextExtractor(langs=['en'], use_gpu=False)

# Extract text from an image file with preprocessing
texts = extractor.extract_text(
    'screenshot.png',
    preprocess=True,
    grayscale=True,
    denoise='fastnlmeans',
    thresholding='binary'
)

print(texts)  # List of recognized text strings
```

### Preprocessing Functions

The module also provides standalone preprocessing functions:

- `to_grayscale(img)`: Convert image to grayscale
- `denoise_image(img, method='fastnlmeans'|'gaussian'|None)`: Apply denoising
- `threshold_image(gray_img, mode='binary'|'adaptive'|None)`: Apply thresholding

### Installation

Install the required dependencies:

```bash
pip install -r requirements-dev.txt
```

**Note**: EasyOCR will download language models on first run (approximately 100MB for English).

### Running Tests

The test suite includes unit tests for all preprocessing functions and the main GameTextExtractor class:

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=game_text_extractor
```

**Note**: Tests force CPU mode (`use_gpu=False`) to avoid GPU dependencies in CI environments.

### Example Test

The test suite creates synthetic images with `cv2.putText` and validates that:
1. Text can be extracted from images
2. Preprocessing functions work correctly
3. The Singleton pattern is enforced
4. Proper error handling for invalid file paths

## Django API

This project also includes a Django REST API for managing games. See the `games` and `gamesapi` directories for the API implementation.
