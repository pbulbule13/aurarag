from src.ingestion.chunker import Chunker

test_text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua." "


chunker = textChunker(chunk_size=100, overlap=10)
chunks = chunker.chunk_text(test_text)

print(f"Total chunks: {len(chunks)}")
print(f"First Chunk: {chunks[0]}")
print(f"Second chunk: {chunks[1]}")


