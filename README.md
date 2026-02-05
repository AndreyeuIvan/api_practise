# API Practice

A Django REST API project for practicing API development.

## GameTextExtractor

This project includes a reusable OCR helper for extracting text from game screenshots using EasyOCR and OpenCV.

### Features

- **Singleton Pattern**: EasyOCR model loaded once per process for efficiency
- **Configurable Preprocessing**: Grayscale conversion, denoising (fastNlMeans/Gaussian), and thresholding (binary/adaptive)
- **Flexible Input**: Accept file paths or numpy arrays
- **GPU Support**: Auto-detect CUDA availability or force CPU-only mode

### Installation

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

**Note**: On first run, EasyOCR will download the required model files (~500MB for English).

### Usage

```python
from game_text_extractor import GameTextExtractor

# Initialize extractor (singleton - called once per process)
extractor = GameTextExtractor(langs=['en'], use_gpu=False)

# Extract text from a screenshot with preprocessing
texts = extractor.extract_text(
    'screenshot.png',
    preprocess=True,
    grayscale=True,
    denoise='fastnlmeans',
    thresholding='binary'
)

print(f"Extracted texts: {texts}")
```

#### ROI (Region of Interest) Cropping

To focus OCR on specific areas of the screenshot (e.g., title regions), use the `roi` parameter:

```python
from game_text_extractor import GameTextExtractor

extractor = GameTextExtractor(langs=['en'], use_gpu=False)

# Define ROI for title area (top 100 pixels of a 1920x1080 screenshot)
roi = {'x': 0, 'y': 0, 'width': 1920, 'height': 100}

# Extract text only from the title region
texts = extractor.extract_text(
    'screenshot.png',
    preprocess=True,
    grayscale=True,
    denoise='fastnlmeans',
    thresholding='binary',
    roi=roi
)

print(f"Title text: {texts}")
```

This avoids processing irrelevant parts of the screenshot and improves OCR accuracy and performance.

### Preprocessing Options

The `extract_text` method supports various preprocessing options:

- `grayscale` (bool): Convert to grayscale
- `denoise` (str|None): Denoising method - 'fastnlmeans', 'gaussian', or None
- `thresholding` (str|None): Thresholding mode - 'binary', 'adaptive', or None
- `threshold_params` (dict): Additional parameters for thresholding (e.g., `{'use_otsu': True}`)
- `roi` (dict|None): Region of Interest with keys 'x', 'y', 'width', 'height' to crop before processing

### Individual Preprocessing Functions

You can also use the preprocessing functions independently:

```python
from game_text_extractor import to_grayscale, denoise_image, threshold_image, crop_roi
import cv2

# Load image
img = cv2.imread('screenshot.png')

# Apply ROI cropping to focus on title area
cropped = crop_roi(img, x=0, y=0, width=1920, height=100)

# Apply preprocessing steps individually
gray = to_grayscale(cropped)
denoised = denoise_image(gray, method='fastnlmeans')
thresholded = threshold_image(denoised, mode='binary', use_otsu=True)
```

### Running Tests

Run the unit tests with pytest:

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest -v tests/test_game_text_extractor.py

# Run with coverage
pytest --cov=game_text_extractor tests/
```

**Note**: Tests use `use_gpu=False` to avoid GPU dependencies in CI/CD environments. EasyOCR will download model files on first test run.

### API Endpoints

This is a Django REST API project. More documentation coming soon.
