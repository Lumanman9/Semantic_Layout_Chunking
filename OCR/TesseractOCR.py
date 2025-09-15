# Install with: pip install pytesseract pillow
import pytesseract
from PIL import Image

# Path to Tesseract executable (Windows only)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Open image
#image = Image.open('/Users/manqin/PycharmProjects/layout_chunking/data/img_12.jpg')
image = Image.open('/Users/manqin/PycharmProjects/layout_chunking/OCR/image1.png')
# Simple text extraction
text = pytesseract.image_to_string(image)
print(text)

# Additional options
custom_config = r'--oem 3 --psm 6'
text = pytesseract.image_to_string(image, config=custom_config)