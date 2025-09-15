import pdfplumber
import json
from typing import List, Dict
from pdf2image import convert_from_path
import tempfile
import os
from paddleocr import PaddleOCR
import pytesseract
from PIL import Image
import numpy as np
import re


class PDFTextExtractor:
    def __init__(self, pdf_path: str, json_path: str, schema, ocr_engine: str = "none", separate_tokens: bool = False):
        """
        Initialize the PDF text extractor with paths to PDF and JSON files.

        Args:
            pdf_path (str): Path to the PDF file
            json_path (str): Path to the JSON file containing bounding boxes
            schema (list): List of labels to extract
            ocr_engine (str): OCR engine to use ("none", "paddle", or "tesseract")
            separate_tokens (bool): Whether to separate tokens in extracted text
        """
        self.pdf_path = pdf_path
        self.json_path = json_path
        self.bounding_boxes = []
        self.layout_data = []
        self.schema = schema
        self.ocr_engine = ocr_engine
        self.separate_tokens = separate_tokens
        if ocr_engine == "paddle":
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en')

    def load_bounding_boxes(self) -> None:
        """Load bounding box data from JSON file."""
        try:
            with open(self.json_path, "r", encoding="utf-8") as file:
                self.bounding_boxes = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found at {self.json_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {self.json_path}")

    def calculate_bbox(self, box: Dict, page) -> tuple:
        """
        Calculate the bounding box coordinates.

        Args:
            box (Dict): Bounding box information
            page: PDF page object

        Returns:
            tuple: Normalized coordinates (x1, y1, x2, y2)
        """
        return (
            box["x"] * page.width,
            box["y"] * page.height,
            (box["x"] + box["width"]) * page.width,
            (box["y"] + box["height"]) * page.height
        )

    def preprocess_text(self, text: str) -> str:
        """
        Preprocess extracted text to properly separate tokens and fix common issues.

        Args:
            text (str): Raw extracted text

        Returns:
            str: Cleaned and properly formatted text
        """
        if not text or not self.separate_tokens:
            return text.strip() if text else ""

        # Step 1: Fix basic spacing issues
        text = text.strip()

        # Step 2: Add space after periods if missing
        text = re.sub(r'\.(?=[A-Z])', '. ', text)

        # Step 3: Add space between concatenated words based on case changes
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)

        # Step 4: Fix spaces around punctuation
        text = re.sub(r'(?<=[.,;!?])(?=[^\s])', ' ', text)

        # Step 5: Fix multiple spaces
        text = re.sub(r'\s+', ' ', text)

        # Step 6: Fix specific patterns
        replacements = {
            r'(?<=\d)\.(?=\d)': '.',  # Keep decimal points
            r'(?<=\d),(?=\d)': ',',  # Keep thousand separators
            r'(?<=[a-z])(?=\d)': ' ',  # Add space between letters and numbers
            r'(?<=\d)(?=[a-z])': ' ',  # Add space between numbers and letters
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)

        return text.strip()

    def extract_text_pdfplumber(self) -> List[Dict]:
        """
        Extract text from PDF using pdfplumber.

        Returns:
            List[Dict]: List of dictionaries containing extracted text and IDs
        """
        self.layout_data = []
        try:
            with pdfplumber.open(self.pdf_path) as pdf:
                box_id = 0
                for box in self.bounding_boxes:
                    if box.get('label') in self.schema:
                        page = pdf.pages[box["pageNum"] - 1]
                        bbox = self.calculate_bbox(box, page)
                        raw_text = page.within_bbox(bbox).extract_text()

                        # Apply token separation if enabled
                        text = self.preprocess_text(raw_text)

                        self.layout_data.append({
                            'id': box_id,
                            'text': text,
                            'label': box.get('label'),
                            'page': box.get('pageNum'),
                            'bbox': bbox
                        })
                        box_id += 1
        except Exception as e:
            raise Exception(f"Error processing PDF with pdfplumber: {str(e)}")

        return self.layout_data

    def extract_text_paddle_ocr(self) -> List[Dict]:
        """
        Extract text from PDF using PaddleOCR.

        Returns:
            List[Dict]: List of dictionaries containing extracted text and IDs
        """
        self.layout_data = []
        try:
            # Convert PDF pages to images
            with tempfile.TemporaryDirectory() as temp_dir:
                # Convert PDF to images
                images = convert_from_path(self.pdf_path)

                box_id = 0
                for box in self.bounding_boxes:
                    if box.get('label') in self.schema:
                        page_num = box["pageNum"] - 1
                        if page_num < len(images):
                            page_image = images[page_num]
                            width, height = page_image.size

                            # Calculate bbox coordinates
                            x1 = int(box["x"] * width)
                            y1 = int(box["y"] * height)
                            x2 = int((box["x"] + box["width"]) * width)
                            y2 = int((box["y"] + box["height"]) * height)

                            # Crop image to bounding box
                            cropped_image = page_image.crop((x1, y1, x2, y2))

                            # Save temporary image
                            temp_image = os.path.join(temp_dir, f"temp_{box_id}.png")
                            cropped_image.save(temp_image)

                            # Perform OCR
                            result = self.ocr.ocr(temp_image, cls=False)

                            # Extract text from OCR result
                            raw_text = ""
                            if result and result[0]:
                                for line in result[0]:
                                    if isinstance(line, list) and len(line) >= 1:
                                        raw_text += line[1][0] + "\n"

                            # Apply token separation if enabled
                            text = self.preprocess_text(raw_text)

                            self.layout_data.append({
                                'id': box_id,
                                'text': text,
                                'label': box.get('label'),
                                'page': box.get('pageNum'),
                                'bbox': (x1, y1, x2, y2)
                            })
                            box_id += 1

        except Exception as e:
            raise Exception(f"Error processing PDF with PaddleOCR: {str(e)}")

        return self.layout_data

    def extract_text_tesseract_ocr(self) -> List[Dict]:
        """
        Extract text from PDF using Tesseract OCR.

        Returns:
            List[Dict]: List of dictionaries containing extracted text and IDs
        """
        self.layout_data = []
        try:
            # Convert PDF pages to images
            with tempfile.TemporaryDirectory() as temp_dir:
                # Convert PDF to images
                images = convert_from_path(self.pdf_path)

                box_id = 0
                for box in self.bounding_boxes:
                    if box.get('label') in self.schema:
                        page_num = box["pageNum"] - 1
                        if page_num < len(images):
                            page_image = images[page_num]
                            width, height = page_image.size

                            # Calculate bbox coordinates
                            x1 = int(box["x"] * width)
                            y1 = int(box["y"] * height)
                            x2 = int((box["x"] + box["width"]) * width)
                            y2 = int((box["y"] + box["height"]) * height)

                            # Crop image to bounding box
                            cropped_image = page_image.crop((x1, y1, x2, y2))

                            # Save temporary image
                            temp_image = os.path.join(temp_dir, f"temp_{box_id}.png")
                            cropped_image.save(temp_image)

                            # Perform OCR with Tesseract
                            raw_text = pytesseract.image_to_string(cropped_image, lang='eng')

                            # Apply token separation if enabled
                            text = self.preprocess_text(raw_text)

                            self.layout_data.append({
                                'id': box_id,
                                'text': text,
                                'label': box.get('label'),
                                'page': box.get('pageNum'),
                                'bbox': (x1, y1, x2, y2)
                            })
                            box_id += 1

        except Exception as e:
            raise Exception(f"Error processing PDF with Tesseract OCR: {str(e)}")

        return self.layout_data

    def process(self) -> List[Dict]:
        """
        Main method to process the PDF and extract text.

        Returns:
            List[Dict]: Extracted text data
        """
        self.load_bounding_boxes()

        if self.ocr_engine == "paddle":
            return self.extract_text_paddle_ocr()
        elif self.ocr_engine == "tesseract":
            return self.extract_text_tesseract_ocr()
        else:
            return self.extract_text_pdfplumber()

if __name__ == "__main__":
    pdf_path = '/Users/manqin/PycharmProjects/layout_chunking/Wikiperson_data/1/Gary Holton.pdf'
    json_path = '/Users/manqin/PycharmProjects/layout_chunking/Wikiperson_data/1/annotations_layout_733.json'
    schema = ['introduction', 'early_fife', 'career', 'contribution', 'award', 'legal_trouble', 'personal_experience','miscellaneous']
    extractor = PDFTextExtractor(
        pdf_path=pdf_path,
        json_path=json_path,
        schema=schema,
        ocr_engine="tesseract"
    )
    layout_data = extractor.process()

    chunks = [doc['text'] for doc in layout_data]

    print(chunks)