from llm_qa_system.retreival import Retrieve
import openai
import os
from loguru import logger


class RAG_llm:
    def __init__(self, client, model, retrieve):
        self.client = client
        self.model = model
        self.temperature = 0.7
        self.max_tokens = 500
        self.retrieve = retrieve

    def response(self, question, retrieval_method='plain',labels=None):
        """
        Generate a response based on retrieved context.

        Args:
            question (str): The question to answer
            retrieval_method (str): Method to retrieve context - 'plain' or 'agentic'
                                   (default: 'plain')

        Returns:
            dict: Contains answer and relevant context chunks
        """
        if retrieval_method == 'agentic':
            if labels is not None:
                print(f'retrieval_method{labels}')
                context, time_takeen = self.retrieve.agentic_query(question=question, llm_client=self.client,
                                                                   model=self.model, labels=labels)
            if labels is None:
                context,time_takeen = self.retrieve.agentic_query(question=question, llm_client=self.client, model=self.model,labels=None)
                logger.info(context)

        else:  # Default to plain query
            context = self.retrieve.query(question=question)

        prompt = f"""Use the following context to answer the question. If you cannot find the answer in the context, say so.
                    Context: {context}
                    Question: {question}
                    Answer: 
                    """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens)

        return {
            'answer': response.choices[0].message.content,
            'relevant_chunks': context
        }


if __name__ == '__main__':
    retrieve = Retrieve(
        collection_name='semantic_layout',
        meta=['1706.09147.pdf']
    )
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    model = "gpt-4-turbo-preview"
    rag_llm = RAG_llm(client=client, model=model, retrieve=retrieve)

    question = 'How good is your method compared with others?'

    # Using the default method (plain query)
    answer_plain = rag_llm.response(question=question)
    logger.info("Plain Query Results:")
    logger.info(answer_plain)

    # Using the agentic method
    answer_agentic = rag_llm.response(question=question, retrieval_method='agentic')
    logger.info("Agentic Query Results:")
    logger.info(answer_agentic)