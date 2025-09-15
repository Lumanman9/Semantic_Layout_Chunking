import spacy
from typing import List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass
import openai
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class SentenceInfo:
    """Store information about each sentence"""
    index: int
    text: str
    start_char: int
    end_char: int
    embedding: np.ndarray = None


@dataclass
class ChunkInfo:
    """Store information about each chunk"""
    sentences: List[SentenceInfo]
    start_char: int
    end_char: int
    text: str
    avg_similarity: float


class SemanticChunker:
    def __init__(
            self,
            openai_api_key: str,
            spacy_model: str = "en_core_web_sm",
            embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize the semantic chunker.

        Args:
            openai_api_key: OpenAI API key for embeddings
            spacy_model: spaCy model name for sentence splitting
            embedding_model: OpenAI embedding model name
        """
        self.nlp = spacy.load(spacy_model)
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.embedding_model = embedding_model

    def _split_into_sentences(self, text: str) -> List[SentenceInfo]:
        """Split text into sentences and create SentenceInfo objects"""
        doc = self.nlp(text)
        sentences = []

        for i, sent in enumerate(doc.sents):
            sentences.append(SentenceInfo(
                index=i,
                text=sent.text.strip(),
                start_char=sent.start_char,
                end_char=sent.end_char,
                embedding=None
            ))

        return sentences

    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text using OpenAI API"""
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return np.array(response.data[0].embedding)

    def _get_embeddings_batch(self, sentences: List[SentenceInfo], batch_size: int = 100) -> None:
        """Get embeddings for all sentences in batches"""
        for i in tqdm(range(0, len(sentences), batch_size), desc="Getting embeddings"):
            batch = sentences[i:i + batch_size]
            texts = [sent.text for sent in batch]

            # Get embeddings for the batch
            response = self.client.embeddings.create(
                input=texts,
                model=self.embedding_model
            )

            # Assign embeddings to sentences
            for j, sent in enumerate(batch):
                sent.embedding = np.array(response.data[j].embedding)

    def _calculate_similarity_matrix(self, sentences: List[SentenceInfo]) -> np.ndarray:
        """Calculate similarity matrix between all sentences"""
        embeddings = np.array([sent.embedding for sent in sentences])
        return cosine_similarity(embeddings)

    def _find_similar_groups(
            self,
            sentences: List[SentenceInfo],
            similarity_matrix: np.ndarray,
            similarity_threshold: float,
            min_sentences_per_chunk: int = 1,
            max_sentences_per_chunk: int = 5
    ) -> List[List[int]]:
        """Find groups of similar sentences"""
        n_sentences = len(sentences)
        visited = set()
        groups = []

        for i in range(n_sentences):
            if i in visited:
                continue

            current_group = [i]
            visited.add(i)

            # Look for similar sentences
            for j in range(i + 1, n_sentences):
                if j in visited:
                    continue

                # Check similarity with all sentences in current group
                similarities = [similarity_matrix[k, j] for k in current_group]
                avg_similarity = np.mean(similarities)

                if avg_similarity >= similarity_threshold:
                    if len(current_group) < max_sentences_per_chunk:
                        current_group.append(j)
                        visited.add(j)
                else:
                    # Break if we find a dissimilar sentence
                    break

            if len(current_group) >= min_sentences_per_chunk:
                groups.append(current_group)

        return groups

    def _create_chunks(
            self,
            sentences: List[SentenceInfo],
            groups: List[List[int]],
            similarity_matrix: np.ndarray
    ) -> List[ChunkInfo]:
        """Create chunks from groups of similar sentences"""
        chunks = []

        for group in groups:
            group_sentences = [sentences[i] for i in group]

            # Calculate average similarity within the group
            group_similarities = []
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    group_similarities.append(similarity_matrix[group[i], group[j]])

            avg_similarity = np.mean(group_similarities) if group_similarities else 1.0

            chunk = ChunkInfo(
                sentences=group_sentences,
                start_char=min(sent.start_char for sent in group_sentences),
                end_char=max(sent.end_char for sent in group_sentences),
                text=' '.join(sent.text for sent in group_sentences),
                avg_similarity=avg_similarity
            )
            chunks.append(chunk)

        return chunks

    def chunk_text(
            self,
            text: str,
            similarity_threshold: float = 0.7,
            min_sentences_per_chunk: int = 1,
            max_sentences_per_chunk: int = 5,
            batch_size: int = 100
    ) -> List[ChunkInfo]:
        """
        Chunk text based on semantic similarity.

        Args:
            text: Input text to chunk
            similarity_threshold: Minimum similarity threshold for grouping sentences
            min_sentences_per_chunk: Minimum number of sentences per chunk
            max_sentences_per_chunk: Maximum number of sentences per chunk
            batch_size: Batch size for embedding API calls

        Returns:
            List of ChunkInfo objects containing the chunks
        """
        # Split text into sentences
        sentences = self._split_into_sentences(text)

        if not sentences:
            return []

        # Get embeddings for all sentences
        self._get_embeddings_batch(sentences, batch_size)

        # Calculate similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(sentences)

        # Find groups of similar sentences
        groups = self._find_similar_groups(
            sentences,
            similarity_matrix,
            similarity_threshold,
            min_sentences_per_chunk,
            max_sentences_per_chunk
        )

        # Create chunks from groups
        chunks = self._create_chunks(sentences, groups, similarity_matrix)

        return chunks

    def get_chunk_statistics(self, chunks: List[ChunkInfo]) -> Dict[str, Any]:
        """Get statistics about the chunks"""
        if not chunks:
            return {}

        n_chunks = len(chunks)
        sentences_per_chunk = [len(chunk.sentences) for chunk in chunks]
        similarities = [chunk.avg_similarity for chunk in chunks]

        return {
            "num_chunks": n_chunks,
            "avg_sentences_per_chunk": np.mean(sentences_per_chunk),
            "min_sentences_per_chunk": min(sentences_per_chunk),
            "max_sentences_per_chunk": max(sentences_per_chunk),
            "avg_similarity": np.mean(similarities),
            "min_similarity": min(similarities),
            "max_similarity": max(similarities)
        }