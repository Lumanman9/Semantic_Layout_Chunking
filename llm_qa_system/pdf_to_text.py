from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from typing import Union, List
import os
from pathlib import Path


class PDFConverter:
    """
    A utility class to convert PDF files to text using LangChain.
    Supports both single PDF files and directories containing multiple PDFs.
    """

    @staticmethod
    def convert_single_pdf(pdf_path: Union[str, Path]) -> str:
        """
        Convert a single PDF file to text.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            str: Extracted text from the PDF
        """
        try:
            # Load the PDF
            loader = PyPDFLoader(str(pdf_path))

            # Load and split the document into pages
            pages = loader.load()

            # Combine all pages into a single text
            text = "\n\n".join(page.page_content for page in pages)

            return text

        except Exception as e:
            raise Exception(f"Error processing PDF {pdf_path}: {str(e)}")

    @staticmethod
    def convert_pdf_directory(
            directory_path: Union[str, Path],
            recursive: bool = False
    ) -> dict[str, str]:
        """
        Convert all PDF files in a directory to text.

        Args:
            directory_path: Path to the directory containing PDF files
            recursive: Whether to search for PDFs in subdirectories

        Returns:
            dict: Dictionary mapping PDF filenames to their extracted text
        """
        try:
            directory_path = Path(directory_path)
            result = {}

            # Define the pattern for finding PDF files
            pattern = "**/*.pdf" if recursive else "*.pdf"

            # Find all PDF files in the directory
            for pdf_file in directory_path.glob(pattern):
                try:
                    # Get the PDF filename
                    pdf_name = pdf_file.name

                    # Convert the single PDF to text
                    pdf_text = PDFConverter.convert_single_pdf(pdf_file)

                    # Store the entire PDF text under the filename
                    result[pdf_name] = pdf_text

                except Exception as e:
                    print(f"Error processing {pdf_file}: {str(e)}")
                    continue

            return result

        except Exception as e:
            raise Exception(f"Error processing directory {directory_path}: {str(e)}")

    @staticmethod
    def save_text_output(
            text: Union[str, dict],
            output_path: Union[str, Path],
            filename: str = None
    ) -> None:
        """
        Save the extracted text to file(s).

        Args:
            text: Extracted text or dictionary of texts
            output_path: Directory to save the output
            filename: Filename for single PDF output (optional)
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        if isinstance(text, str):
            # Single PDF case
            output_file = output_path / (filename or "output.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
        else:
            # Multiple PDFs case
            for pdf_name, content in text.items():
                output_file = output_path / f"{os.path.splitext(pdf_name)[0]}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)


# Example usage functions
def convert_single_pdf_example():
    """Example of converting a single PDF file"""
    converter = PDFConverter()

    # Convert single PDF
    pdf_path = "path/to/your/document.pdf"
    text = converter.convert_single_pdf(pdf_path)

    # Save the output
    converter.save_text_output(
        text=text,
        output_path="path/to/output",
        filename="output.txt"
    )


def convert_multiple_pdfs_example():
    """Example of converting multiple PDF files from a directory"""
    converter = PDFConverter()

    # Convert all PDFs in directory
    pdf_dir = "path/to/pdf/directory"
    texts = converter.convert_pdf_directory(
        pdf_dir,
        recursive=True  # Set to True to include subdirectories
    )

    # Save all outputs
    converter.save_text_output(
        text=texts,
        output_path="path/to/output"
    )


