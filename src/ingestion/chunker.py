class TextChunker:
    """split text on to overalapping chunks
     Splits text into overlapping chunks.
      
      Overlap helps preserve context across chunk boundaries.
      For example, if a sentence is split across two chunks,
      the overlap ensures that both chunks retain some context"""

    def __init__(self, chunk_size: int = 1000, overlap: int = 50):
        #store the configuration
        pass

    def chunk_text(self, text: str) -> list[str]:
        """chunk the text into overlapping chunks"""
        #split the text into chunks based on the configuration
        Return format:
        [
            {
                "text": "Python is great..."
                "chunk_index": 0,
                "start_char": 0,
                "end_char": 1000

            },
            {
                "text": "great for data science..."
                "chunk_index": 1,
                "start_char": 450,
                "end_char": 950 

            }
        ]