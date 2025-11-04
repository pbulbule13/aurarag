from embedding_service import EmbeddingService
import os
from dotenv import load_dotenv




load_dotenv()  # Load environment variables from .env file

def test_single_embedding():
    """
    test generating embedding for a single text input
    
    """

    try:

        service = EmbeddingService():
        """Initialize the embedding service"""
        print("===testing single embedding generation===")

    embedding = serivce.generate_embedding(text)

    except Exception as e:
        print("Error during embedding generation:", str(e))
        return  


def test_batch_embedding():
    """
    test generating embeddings for a batch of text inputs
    """
    try:
        service = EmbeddingService()
        print("===testing batch embedding generation===")

        texts = [
            "Hello world!",
            "Testing embedding generation.",
            "OpenAI provides powerful models."
        ]

        embeddings = [service.generate_embedding(text) for text in texts]

        for i, emb in enumerate(embeddings):
            print(f"Embedding for text {i}: {emb[:5]}...")  # Print first 5 dimensions

    except Exception as e:
        print("Error during batch embedding generation:", str(e))
        return
    

    def test_similarity():
        """
        test similarity between two embeddings
        """
        try:
            service = EmbeddingService()
            print("===testing embedding similarity===")

            text1 = "The quick brown fox jumps over the lazy dog."
            text2 = "A fast dark-colored fox leaps above a sleepy canine."

            emb1 = service.generate_embedding(text1)
            emb2 = service.generate_embedding(text2)

            # Compute cosine similarity
            dot_product = sum(a * b for a, b in zip(emb1, emb2))
            magnitude1 = sum(a * a for a in emb1) ** 0.5
            magnitude2 = sum(b * b for b in emb2) ** 0.5

            similarity = dot_product / (magnitude1 * magnitude2)
            print(f"Cosine similarity between embeddings: {similarity}")

        except Exception as e:
            print("Error during embedding similarity test:", str(e))
            return