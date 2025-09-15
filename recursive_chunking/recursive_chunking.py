import spacy
from typing import List, Dict, Any, Union
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ChunkMetadata:
    """Metadata for each chunk including its hierarchy and overlap information"""
    level: int  # Depth in the hierarchy
    parent_id: Union[str, None]  # ID of parent chunk
    chunk_id: str  # Unique ID for this chunk
    granularity: str  # Type of splitting used
    start_idx: int  # Start index in original text
    end_idx: int  # End index in original text
    overlap_prev: str = None  # ID of previous overlapping chunk
    overlap_next: str = None  # ID of next overlapping chunk


class RecursiveChunker:
    def __init__(self, model_name: str = "en_core_web_sm"):
        """
        Initialize the recursive chunker.

        Args:
            model_name: Name of the spaCy model to use
        """
        self.nlp = spacy.load(model_name)
        self.valid_granularities = ["document", "paragraph", "sentence", "token"]
        self.chunks_by_level = defaultdict(list)
        self.chunk_metadata = {}

    def _generate_chunk_id(self, level: int, index: int, parent_id: str = None) -> str:
        """Generate a unique ID for a chunk based on its position in the hierarchy"""
        if parent_id:
            return f"{parent_id}-{index}"
        return f"level_{level}_{index}"

    def _create_overlapping_chunks(
            self,
            chunks: List[str],
            overlap_size: Union[int, float],
            is_percentage: bool = False
    ) -> List[str]:
        """
        Create overlapping chunks from a list of chunks.

        Args:
            chunks: List of text chunks
            overlap_size: Size of overlap (either number of units or percentage)
            is_percentage: Whether overlap_size is a percentage

        Returns:
            List of overlapping chunks
        """
        if not chunks:
            return []

        overlapping_chunks = []

        for i, chunk in enumerate(chunks):
            if is_percentage:
                # Calculate overlap size as percentage of chunk length
                current_overlap = max(1, int(len(chunk) * overlap_size))
            else:
                current_overlap = overlap_size

            if i > 0:
                # Add overlap from previous chunk
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-current_overlap:] if len(prev_chunk) > current_overlap else prev_chunk
                chunk = overlap_text + chunk

            if i < len(chunks) - 1:
                # Add overlap to next chunk
                next_chunk = chunks[i + 1]
                overlap_text = next_chunk[:current_overlap] if len(next_chunk) > current_overlap else next_chunk
                chunk = chunk + overlap_text

            overlapping_chunks.append(chunk)

        return overlapping_chunks

    def _split_into_paragraphs(self, text: str, overlap_params: Dict = None) -> List[str]:
        """Split text into paragraphs with optional overlap"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if overlap_params and overlap_params.get('paragraph_overlap', 0) > 0:
            return self._create_overlapping_chunks(
                paragraphs,
                overlap_params['paragraph_overlap'],
                overlap_params.get('use_percentage', False)
            )
        return paragraphs

    def _split_into_sentences(self, text: str, overlap_params: Dict = None) -> List[str]:
        """Split text into sentences with optional overlap"""
        doc = self.nlp(text)
        sentences = [str(sent).strip() for sent in doc.sents]
        if overlap_params and overlap_params.get('sentence_overlap', 0) > 0:
            return self._create_overlapping_chunks(
                sentences,
                overlap_params['sentence_overlap'],
                overlap_params.get('use_percentage', False)
            )
        return sentences

    def _split_into_tokens(self, text: str, overlap_params: Dict = None) -> List[str]:
        """Split text into tokens with optional overlap"""
        doc = self.nlp(text)
        tokens = [token.text for token in doc if not token.is_space]
        if overlap_params and overlap_params.get('token_overlap', 0) > 0:
            return self._create_overlapping_chunks(
                tokens,
                overlap_params['token_overlap'],
                overlap_params.get('use_percentage', False)
            )
        return tokens

    def chunk_recursively(
            self,
            text: str,
            granularity_levels: List[str],
            overlap_params: Dict[str, Any] = None,
            min_chunk_size: int = 10,
            max_chunk_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Recursively chunk text based on specified granularity levels with optional overlap.

        Args:
            text: Input text to chunk
            granularity_levels: List of granularity levels in order
            overlap_params: Dictionary containing overlap parameters:
                {
                    'paragraph_overlap': int/float,  # Overlap size for paragraphs
                    'sentence_overlap': int/float,   # Overlap size for sentences
                    'token_overlap': int/float,      # Overlap size for tokens
                    'use_percentage': bool           # Whether overlap sizes are percentages
                }
            min_chunk_size: Minimum size of a chunk in characters
            max_chunk_size: Maximum size of a chunk in characters

        Returns:
            Dictionary containing chunks and their metadata
        """
        # Set default overlap parameters if none provided
        if overlap_params is None:
            overlap_params = {
                'paragraph_overlap': 0,
                'sentence_overlap': 0,
                'token_overlap': 0,
                'use_percentage': False
            }

        # Validate granularity levels
        for level in granularity_levels:
            if level not in self.valid_granularities:
                raise ValueError(f"Invalid granularity level: {level}")

        # Reset storage
        self.chunks_by_level.clear()
        self.chunk_metadata.clear()

        # Start with the entire document
        self.chunks_by_level[0] = [text]
        self.chunk_metadata[self._generate_chunk_id(0, 0)] = ChunkMetadata(
            level=0,
            parent_id=None,
            chunk_id=self._generate_chunk_id(0, 0),
            granularity="document",
            start_idx=0,
            end_idx=len(text)
        )

        # Process each granularity level
        for level, granularity in enumerate(granularity_levels, start=1):
            parent_chunks = self.chunks_by_level[level - 1]

            for parent_idx, parent_chunk in enumerate(parent_chunks):
                parent_id = self._generate_chunk_id(level - 1, parent_idx)

                if len(parent_chunk) < min_chunk_size:
                    continue

                # Split based on granularity with overlap
                if granularity == "paragraph":
                    sub_chunks = self._split_into_paragraphs(parent_chunk, overlap_params)
                elif granularity == "sentence":
                    sub_chunks = self._split_into_sentences(parent_chunk, overlap_params)
                elif granularity == "token":
                    sub_chunks = self._split_into_tokens(parent_chunk, overlap_params)
                else:
                    continue

                # Process sub-chunks
                current_position = 0
                prev_chunk_id = None

                for i, chunk in enumerate(sub_chunks):
                    chunk_id = self._generate_chunk_id(level, i, parent_id)

                    # Find chunk positions in original text
                    start_idx = parent_chunk.find(chunk, current_position)
                    end_idx = start_idx + len(chunk)
                    current_position = end_idx

                    # Create metadata with overlap information
                    metadata = ChunkMetadata(
                        level=level,
                        parent_id=parent_id,
                        chunk_id=chunk_id,
                        granularity=granularity,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        overlap_prev=prev_chunk_id,
                        overlap_next=self._generate_chunk_id(level, i + 1, parent_id) if i < len(
                            sub_chunks) - 1 else None
                    )

                    # Update previous chunk's next overlap
                    if prev_chunk_id:
                        self.chunk_metadata[prev_chunk_id].overlap_next = chunk_id

                    # Store chunk and metadata
                    self.chunks_by_level[level].append(chunk)
                    self.chunk_metadata[chunk_id] = metadata
                    prev_chunk_id = chunk_id

        return {
            "chunks": dict(self.chunks_by_level),
            "metadata": self.chunk_metadata
        }

    def get_overlapping_chunks(self, chunk_id: str) -> Dict[str, str]:
        """Get the overlapping chunks for a given chunk"""
        metadata = self.chunk_metadata.get(chunk_id)
        if not metadata:
            return {}

        result = {}
        if metadata.overlap_prev:
            result['previous'] = self.get_chunk_text(metadata.overlap_prev)
        if metadata.overlap_next:
            result['next'] = self.get_chunk_text(metadata.overlap_next)

        return result
