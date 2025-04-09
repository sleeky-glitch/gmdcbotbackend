from openai import OpenAI
from app.config import get_settings
from app.utils.prompts import SYSTEM_PROMPT, get_chat_prompt
from typing import List

settings = get_settings()

class OpenAIService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings for the given text using OpenAI's embedding model.
        """
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            # Extract the embedding vector from the response
            embedding_vector = response.data[0].embedding
            return embedding_vector
        except Exception as e:
            raise Exception(f"OpenAI Embedding API call failed: {str(e)}")

    async def generate_response(self, context: str, user_query: str) -> str:
        """
        Generate a response using OpenAI's chat completion API.
        """
        try:
            chat_prompt = get_chat_prompt(context, user_query)
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": chat_prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI Chat API call failed: {str(e)}")

    async def process_query(self, query: str, context: str = "") -> str:
        """
        Process a query with optional context and generate a response.
        """
        try:
            # If context is provided, use it to generate a more informed response
            if context:
                return await self.generate_response(context, query)
            else:
                # Generate a response without specific context
                return await self.generate_response("", query)
        except Exception as e:
            raise Exception(f"Query processing failed: {str(e)}")
