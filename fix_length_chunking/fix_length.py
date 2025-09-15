import spacy
from typing import List
from loguru import logger

'''
python -m spacy download en_core_web_sm
'''

class FixLengthChunker:
    def __init__(self, length: int, granularity: str):
        """
        Initialize the chunker with a fixed length and granularity.

        Args:
            length (int): Number of units (tokens/sentences/paragraphs) per chunk
            granularity (str): The unit type ('token', 'sentence', or 'paragraph')
        """
        self.nlp = spacy.load("en_core_web_sm")
        self.length = length
        self.granularity = granularity
        if granularity not in ['token', 'sentence', 'paragraph']:
            raise ValueError(f"Invalid granularity: {granularity}. Must be one of ['token', 'sentence', 'paragraph'].")

    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks of fixed length based on specified granularity.

        Args:
            text (str): Input text to be split

        Returns:
            List[str]: List of text chunks, each containing the specified number of units
        """
        doc = self.nlp(text)
        chunks = []

        if self.granularity == "token":
            # Get all tokens excluding whitespace
            tokens = [token.text_with_ws for token in doc if not token.is_space]
            logger.info(len(tokens))

            # Create chunks of specified length
            current_chunk = []
            token_count = 0

            for token in tokens:
                current_chunk.append(token)
                token_count += 1

                if token_count == self.length:
                    chunks.append(''.join(current_chunk).strip())
                    current_chunk = []
                    token_count = 0

            # Add any remaining tokens as the last chunk
            if current_chunk:
                chunks.append(''.join(current_chunk).strip())

        elif self.granularity == "sentence":
            # Get all sentences
            sentences = list(doc.sents)
            logger.info(len(sentences))

            # Create chunks of specified number of sentences
            current_chunk = []
            sentence_count = 0

            for sentence in sentences:
                current_chunk.append(str(sentence))
                sentence_count += 1

                if sentence_count == self.length:
                    chunks.append(' '.join(current_chunk).strip())
                    current_chunk = []
                    sentence_count = 0

            # Add any remaining sentences as the last chunk
            if current_chunk:
                chunks.append(' '.join(current_chunk).strip())

        elif self.granularity == "paragraph":
            # Split by double newlines to get paragraphs
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            logger.info(len(paragraphs))

            # Create chunks of specified number of paragraphs
            current_chunk = []
            paragraph_count = 0

            for paragraph in paragraphs:
                current_chunk.append(paragraph)
                paragraph_count += 1

                if paragraph_count == self.length:
                    chunks.append('\n\n'.join(current_chunk).strip())
                    current_chunk = []
                    paragraph_count = 0

            # Add any remaining paragraphs as the last chunk
            if current_chunk:
                chunks.append('\n\n'.join(current_chunk).strip())

        return chunks