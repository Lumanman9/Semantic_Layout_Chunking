from llm_qa_system.embedding import Embedding
from qdrant_client import QdrantClient
from typing import Dict, Any, List, Tuple
import os
from dotenv import load_dotenv
from loguru import logger
from qdrant_client.models import Filter, FieldCondition, MatchValue
import openai
import time
import json


load_dotenv()


class Retrieve:
    def __init__(self, collection_name, meta: List[str] = None, n_chunks: int = 3):
        self.collection_name = collection_name
        self.n_chunks = n_chunks
        self.qdrant = QdrantClient(os.getenv('qdrant_url'), port=os.getenv('qdrant_port'))
        self.meta = meta
        self.schema = ['author', 'title', 'introduction', 'methodology', 'related_work', 'experiment',
              'background', 'dataset', 'acknowledge', 'conclusion', 'result_discussion']
        #self.schema = ['company_overview', 'manager_discussion', 'performance_summary', 'strategy_outlook', 'segment_business_review', 'risk_factors',
        #  'governance_and_leadership', 'sustainability', 'financial_statement', 'financial_statement_note', 'auditors_report', 'shareholder_information',
        #  'legal_compliance', 'miscellaneous']

    def query(self, question: str, temperature: float = 0.7, max_tokens: int = 500) -> Tuple[str, float]:
        """
        Query the RAG system with a question.

        Args:
            question: User's question
            temperature: Temperature for OpenAI completion
            max_tokens: Maximum tokens for completion

        Returns:
            Tuple containing:
                - context: String containing concatenated relevant chunks
                - elapsed_time: Time taken for retrieval in seconds
        """
        start_time = time.perf_counter()
        filter_condition = None
        if self.meta:
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="name",  # Adjust this path based on where name is stored
                        match=MatchValue(value=self.meta[0])
                    )
                ]
            )
            logger.info(f"Filtering by document name: {self.meta[0]}")
        else:
            logger.info("No metadata filter applied, searching across all documents")

        # Generate embedding for the question
        embedder = Embedding(chunks=[question], collection_name=self.collection_name)
        question_embedding = embedder.generate_embedding(question)

        # Search for similar chunks
        search_result = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=question_embedding,
            limit=self.n_chunks,
            query_filter=filter_condition,
            with_payload=True
        ).points

        # Extract relevant chunks and their scores
        chunks = []
        for result in search_result:
            chunks.append({
                'text': result.payload['text'],
                'similarity': result.score,
                'metadata': {k: v for k, v in result.payload.items() if k != 'text'}
            })

        logger.info(chunks)
        # Prepare context for GPT
        context = "\n\n".join([chunk['text'] for chunk in chunks])

        elapsed_time = time.perf_counter() - start_time
        logger.info(f"Retrieval time: {elapsed_time:.3f} seconds")

        # Check if the JSON file exists, if not create it with an empty list
        json_file_path = f'retrieve_chunks{self.collection_name}_query.json'
        if not os.path.exists(json_file_path):
            with open(json_file_path, 'w') as f:
                json.dump([], f)

        # Read existing Literature_Paper
        with open(json_file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                # If file exists but is empty or invalid
                data = []

        # Append new entry
        new_entry = {
            "name": self.meta[0],
            "retrieve_time": round(elapsed_time, 3),
            "question": question,
            "context": context
        }
        data.append(new_entry)

        # Write updated Literature_Paper back to file
        with open(json_file_path, 'w') as f:
            json.dump(data, f, indent=2)

        return context, elapsed_time

    def parse_sections_from_response(self, response_text: str) -> list:
        """
        Parse section names from LLM response.

        Args:
            response_text (str): Raw response from LLM

        Returns:
            list: List of section names
        """
        cleaned_text = response_text.replace('- ', '').replace('* ', '').replace('1. ', '')
        sections = [s.strip() for s in cleaned_text.replace('\n', ',').split(',') if s.strip()]
        return sections

    def agentic_query(self, question: str, llm_client, model, labels=None,temperature: float = 0.7, max_tokens: int = 500) -> Tuple[
        str, float]:
        """
        Query function that uses LLM to determine relevant sections before searching vector database.

        Args:
            question (str): The query question
            llm_client: The LLM client instance
            model (str): The model name to use
            temperature (float): Temperature parameter for LLM
            max_tokens (int): Maximum tokens for LLM response

        Returns:
            Tuple containing:
                - context: String containing concatenated relevant chunks or None if failed
                - elapsed_time: Time taken for retrieval in seconds
        """
        start_time = time.perf_counter()

        if self.meta is None:
            logger.warning('Please provide meta Literature_Paper.')
            return None, 0.0


        if labels is None:
            # Ask LLM which sections are most relevant
            prompt = f'''We have a question: {question}, we want to find the answer from a literature paper which 
            has those sections:{self.schema}. Please tell me which three sections will best help to answer the question.
            Return the answer only as a comma-separated list of section names. No other words needed.'''
            try:
                response = llm_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens
                )

                relevant_sections = self.parse_sections_from_response(response.choices[0].message.content)
                logger.info(f"Selected sections: {relevant_sections}")

                embedder = Embedding(chunks=[question], collection_name=self.collection_name)
                question_embedding = embedder.generate_embedding(question)

                section_filter = Filter(
                    must=[
                        FieldCondition(key="name", match=MatchValue(value=self.meta[0])),
                        Filter(
                            should=[
                                FieldCondition(key="label", match=MatchValue(value=section))
                                for section in relevant_sections
                            ]
                        )
                    ]
                )

                search_results = self.qdrant.query_points(
                    collection_name=self.collection_name,
                    query=question_embedding,
                    query_filter=section_filter,
                    limit=self.n_chunks,
                    with_payload=True
                ).points

                if not search_results:
                    logger.warning("No relevant chunks found")
                    elapsed_time = time.perf_counter() - start_time
                    return None, elapsed_time

                chunks = []
                for result in search_results:
                    chunks.append({
                        'text': result.payload['text'],
                        'similarity': result.score,
                        'metadata': {k: v for k, v in result.payload.items() if k != 'text'}
                    })

                context = "\n\n".join([chunk['text'] for chunk in chunks])
                elapsed_time = time.perf_counter() - start_time
                logger.info(f"Retrieval time: {elapsed_time:.3f} seconds")

                # Check if the JSON file exists, if not create it with an empty list
                json_file_path = f'retrieve_chunks{self.collection_name}_agentic_without_label.json'
                if not os.path.exists(json_file_path):
                    with open(json_file_path, 'w') as f:
                        json.dump([], f)

                # Read existing Literature_Paper
                with open(json_file_path, 'r') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        # If file exists but is empty or invalid
                        data = []

                # Append new entry
                new_entry = {
                    "name": self.meta[0],
                    "selected_sections": relevant_sections,
                    "retrieve_time": round(elapsed_time, 3),
                    "question": question,
                    "context": context
                }
                data.append(new_entry)

                # Write updated Literature_Paper back to file
                with open(json_file_path, 'w') as f:
                    json.dump(data, f, indent=2)

                return context, elapsed_time

            except Exception as e:
                logger.error(f"Error in agentic_query: {str(e)}")
                elapsed_time = time.perf_counter() - start_time
                return None, elapsed_time

        if labels is not None:
            print(f'query:{labels}')
            section_filter = Filter(
                must=[
                    # Filter by name
                    FieldCondition(key="name", match=MatchValue(value=self.meta[0])),
                    # Filter by the provided labels
                    Filter(
                        should=
                            FieldCondition(key="label", match=MatchValue(value=labels[0]))
                    )
                ]
            )

            embedder = Embedding(chunks=[question], collection_name=self.collection_name)
            question_embedding = embedder.generate_embedding(question)

            search_results = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=question_embedding,
                query_filter=section_filter,
                limit=self.n_chunks,
                with_payload=True
            ).points

            if not search_results:
                logger.warning("No relevant chunks found")
                elapsed_time = time.perf_counter() - start_time
                return None, elapsed_time

            chunks = []
            for result in search_results:
                chunks.append({
                    'text': result.payload['text'],
                    'similarity': result.score,
                    'metadata': {k: v for k, v in result.payload.items() if k != 'text'}
                })
            print(f'chunks:{len(chunks)}',f'search results:{len(search_results)}')
            context = "\n\n".join([chunk['text'] for chunk in chunks])
            elapsed_time = time.perf_counter() - start_time
            logger.info(f"Retrieval time: {elapsed_time:.3f} seconds")

            # Check if the JSON file exists, if not create it with an empty list
            json_file_path = f'retrieve_chunks{self.collection_name}_agentic_with_label.json'
            if not os.path.exists(json_file_path):
                with open(json_file_path, 'w') as f:
                    json.dump([], f)

            # Read existing Literature_Paper
            with open(json_file_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    # If file exists but is empty or invalid
                    data = []

            # Append new entry
            new_entry = {
                "name": self.meta[0],
                "retrieve_time": round(elapsed_time, 3),
                "question": question,
                "context": context
            }
            data.append(new_entry)

            # Write updated Literature_Paper back to file
            with open(json_file_path, 'w') as f:
                json.dump(data, f, indent=2)

            return context, elapsed_time




if __name__ == '__main__':
    retrieve = Retrieve(
        collection_name='semantic_layout',
        meta=['1706.09147.pdf'],
    )
    question = 'How good is your method compared with others?'
    llm_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    model = "gpt-4-turbo-preview"

    context, time_taken = retrieve.agentic_query(
        question=question,
        llm_client=llm_client,
        model=model,
        temperature=0.7,
        max_tokens=500
    )

    logger.info(f"Retrieved context in {time_taken:.3f} seconds:")
    logger.info(context)

    with open('chunks.txt', 'w') as f:
            f.write(f"{context}\n")
