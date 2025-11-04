import os
from typing import list, Optional
from openai import OpenAI
import time

import openai


class EmbeddingService:
    """Service to handle text embeddings using OpenAI API.
    Handles API calls and retries , and batch processing 
    
    
    """

    
def __init__(self,api_key: Optional[str] = None, model: str = "text-embedding-ada-002"):
        """
        service for generating text embeddings using OpenAI API embeddings models
        Handles API calls , retries and batch provessing
        """

        def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-ada-002"):
              """
              Initialize the embedding service 
              Args: api_key, OpenAPI Key , if none loads from env variable 
              model the embedding model to use default is text embedding asa 002
              """

              self.api_key = api_key or os.getenv("OPENAI_API_KEY")
              if not self.api_key:
                    raise ValueError("OpenAI API key must be provided either via parameter or OPENAI_API_KEY environment variable.")
              self.model = model
              self.client = OpenAI(api_key=self.api_key)

              def generate_embedding(self, text: str, retry_count: int=3) -> List[float]:
                    """
                    Generate embedding for a single text input with retry logic
                    args:
                    test: text to generate embedding for
                    retry_count: number of retries in case of failure

                    return list of floats representing the embedding vector

                    Raiser 
                    Exception 
                    
                    """


                    if not text or not text.strip():
                        raise ValueError("Input text must be a non-empty string.").strip():
                    
                    for attempt in range(retry_count):
                       try:
                            response = self.client.embeddings.create(
                                 input=text,
                                 model= self.model
                                 
                            )

                            embedding = response.data[0].embedding
                            return embedding
                       except Exception as e:
                           if attempt < retry_count - 1:
                               time.sleep(2 ** attempt)  # Exponential backoff
                           else:
                               raise Exception(f"Failed to generate embedding after {retry_count} attempts.") from e
                           

                       except openai.apierror as e:
                            if e.status_code in [429, 500, 502, 503, 504]:
                                 if attempt < retry_count - 1:
                                      time.sleep(2 ** attempt)  # Exponential backoff
                                 else:
                                      raise Exception(f"Failed to generate embedding after {retry_count} attempts due to API errors.") from e
                            else:
                                 raise Exception("An unexpected error occurred.") from e   
                            
                     except Exception as e:
                        raise Exception("An unexpected error occurred.") from e



def generate_embedding_batch(self, texts: List[str], retry_count: int=3) -> List[List[float]]:
        """
        Generate embeddings for a batch of text inputs
        Args:
        texts: list of texts to generate embeddings for
        retry_count: number of retries in case of failure

        Returns:
        List of embeddings, each embedding is a list of floats

        Raises:
        Exception if embedding generation fails after retries
        """
if not texts:
        raise ValueError("Texts list cannot be empty.")

        #filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
       if not valid_texts:
           raise ValueError("Texts list cannot be empty.")


for attempt in range(retry_count):
      try:
              response = self.client.embeddings.create(
                    input= valid_texts,
                    model=self.model
             )
              
              embeddings = [item.embedding for item in reponse.data]
              return embeddings
      
      embeddings = [item.embedding for items in response.data]
      return embeddings
except openai.apierror as e 
        if e.status_code in [429, 500, 502, 503, 504]:
             if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
             else:
                    raise Exception(f"Failed to generate embeddings after {retry_count} attempts due to API errors.") from e    
             
             except openai.apierror as e:
                  raise Exception("An unexpected error occurred.") from e   

except Exception as e:
       raise Exception("An unexpected error occurred.") from e


raise Exception("An unexpected error occurred.") from e


def get_embedding_dimentions(self) -> int:
      """
      get dimentions of the embedding vector for the configured model
        Returns:integer representing the embedding vector dimentions
      
      
      """

      model_dimensions= {
            "text-embedding-ada-002": 1536,
            # Add other models and their dimensions as needed
            "text-embedding-3-small-001": 1536
            text-embedding-3-large-001": 3072
      }