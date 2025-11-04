class TextChunker:
    """split text on to overalapping chunks
     Splits text into overlapping chunks.
      
      Overlap helps preserve context across chunk boundaries.
      For example, if a sentence is split across two chunks,
      the overlap ensures that both chunks retain some context"""

    def __init__(self, chunk_size: int = 1000, overlap: int = 50):
        #store the configuration
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> list[str]:
        """chunk the text into overlapping chunks"""
        chunks = []
        start = 0 
        chunk_index = 0 

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
         
            chunk_dict = {
                text  = chunk_text,
                chunk_index = chunk_index,
                start_char = start
                end_char = end
            }
        

        chunks.append(chunk_dict)

        start += chunk_size - overlap
        chunk_index += 1

        return chunks
      
  