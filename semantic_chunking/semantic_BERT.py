import torch
import spacy
from transformers import BertTokenizer, BertForNextSentencePrediction
from typing import List, Tuple


class SemanticChunkerBert:
    def __init__(self):
        # Initialize BERT tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
        self.model.eval()  # Set model to evaluation mode

        # Load spaCy model for English
        # Using the small model for speed, can use 'en_core_web_md' or 'en_core_web_lg' for better accuracy
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer'])
        # Only enable sentence segmentation
        self.nlp.enable_pipe('senter')

    def split_into_sentences(self, text: str) -> List[Tuple[int, str]]:
        """
        Split text into sentences using spaCy and assign indices.

        Args:
            text (str): Input text to be split

        Returns:
            List[Tuple[int, str]]: List of (index, sentence) tuples
        """
        doc = self.nlp(text)
        return [(i, sent.text.strip()) for i, sent in enumerate(doc.sents)]

    def check_sentence_continuity(self, sent1: str, sent2: str) -> bool:
        """
        Check if sent2 is likely to follow sent1 using BERT's NSP task.

        Args:
            sent1 (str): First sentence
            sent2 (str): Second sentence

        Returns:
            bool: True if sent2 likely follows sent1, False otherwise
        """
        # Encode the sentence pair
        encoding = self.tokenizer(sent1, sent2, return_tensors='pt', padding=True, truncation=True)

        # Get model prediction
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)

            # Get probability that sent2 follows sent1
            # Index 0 corresponds to IsNext, 1 to NotNext
            is_next_prob = probs[0][0].item()

        return is_next_prob > 0.5

    def create_chunks(self, text: str) -> List[str]:
        """
        Create semantic chunks from input text.

        Args:
            text (str): Input text to be chunked

        Returns:
            List[str]: List of text chunks
        """
        # Split text into indexed sentences
        indexed_sentences = self.split_into_sentences(text)

        if not indexed_sentences:
            return []

        chunks = []
        current_chunk = [indexed_sentences[0][1]]  # Start with first sentence

        # Iterate through adjacent sentence pairs
        for i in range(len(indexed_sentences) - 1):
            sent1 = indexed_sentences[i][1]
            sent2 = indexed_sentences[i + 1][1]

            # Check if sentences should be in the same chunk
            if self.check_sentence_continuity(sent1, sent2):
                current_chunk.append(sent2)
            else:
                # End current chunk and start new one
                chunks.append(' '.join(current_chunk))
                current_chunk = [sent2]

        # Add last chunk if not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks


def main():
    # Example usage
    text = """
    The sun was setting over the horizon. The sky was painted in beautiful shades of orange and pink. 
    Meanwhile, in the city, traffic was building up. People were rushing home from work. 
    The local cafe was closing for the day. The chairs were stacked and floors were being mopped.
    """
    with open("/Users/manqin/PycharmProjects/layout_chunking/test/output/pdf_to_text_test.txt", "r",
              encoding="utf-8") as file:
        text = file.read()

    chunker = SemanticChunkerBert()
    chunks = chunker.create_chunks(text)

    print("Original text split into chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i}:")
        print(f"{chunk}\n")



if __name__ == "__main__":
    main()