# API Practice

A Django-based REST API project for managing games.

## GameTextExtractor

This project includes a reusable OCR helper for extracting text from game screenshots using EasyOCR (PyTorch) and OpenCV preprocessing.

### Usage

The `GameTextExtractor` is a Singleton class that provides OCR capabilities with configurable preprocessing options:

```python
from game_text_extractor import GameTextExtractor

# Initialize the extractor (only once per process due to Singleton pattern)
extractor = GameTextExtractor(langs=['en'], use_gpu=False)

# Extract text from an image with preprocessing
texts = extractor.extract_text(
    'path/to/screenshot.png',
    preprocess=True,
    grayscale=True,
    denoise='fastnlmeans',
    thresholding='binary'
)

print(texts)  # List of recognized text strings
```

**Key Features:**
- **Singleton pattern**: EasyOCR model is loaded only once per process
- **Flexible preprocessing**: Configurable grayscale conversion, denoising (fastNlMeans or Gaussian), and thresholding (binary or adaptive)
- **Multiple input formats**: Accepts both file paths and numpy arrays
- **GPU support**: Automatically detects CUDA availability or can be manually configured

**Preprocessing Helper Functions:**

The module also provides standalone preprocessing functions that can be used independently:

```python
from game_text_extractor import to_grayscale, denoise_image, threshold_image
import cv2

# Load and preprocess an image
img = cv2.imread('screenshot.png')
gray = to_grayscale(img)
denoised = denoise_image(gray, method='fastnlmeans')
thresholded = threshold_image(denoised, mode='binary', use_otsu=True)
```

### Running Tests

To run the unit tests for GameTextExtractor:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests with pytest
pytest tests/test_game_text_extractor.py -v

# Run with coverage report
pytest tests/test_game_text_extractor.py --cov=game_text_extractor --cov-report=html
```

**Note:** EasyOCR will download language models on the first run. The tests force CPU mode to avoid GPU dependencies.

### Installation

Install the required dependencies:

```bash
pip install -r requirements-dev.txt
```

### Requirements

- Python 3.7+
- EasyOCR
- OpenCV (opencv-python)
- PyTorch
- NumPy
- pytest (for testing)
