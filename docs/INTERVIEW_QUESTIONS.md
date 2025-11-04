# RAG System Interview Questions & Answers Guide

This document contains comprehensive interview questions and cross-questions that might be asked when discussing the AuraRAG project in technical interviews.

---

## Table of Contents
1. [Architecture & Design Questions](#architecture--design-questions)
2. [Technical Implementation Questions](#technical-implementation-questions)
3. [Advanced/Tricky Questions](#advancedtricky-questions)
4. [Behavioral Questions](#behavioral-questions)

---

## Architecture & Design Questions

### Q1: "Explain your RAG architecture. How does data flow through the system?"

**Expected Answer Coverage:**
- **Ingestion Pipeline:**
  - Document connectors (PDF, web scraper, APIs) fetch raw data
  - Text preprocessor cleans and normalizes the text
  - Chunker splits documents into overlapping segments
  - Embedding service converts chunks to vector representations
  - Vector database (ChromaDB) stores embeddings with metadata

- **Query Pipeline:**
  - User query received via FastAPI endpoint
  - Query gets embedded using the same embedding model
  - Vector similarity search retrieves top-K relevant chunks
  - Retrieved chunks become context for LLM
  - OpenAI GPT generates answer based on context
  - Response returned to user

- **Storage Layer:**
  - ChromaDB for vector storage
  - Metadata stored alongside vectors (source, timestamp, chunk_id)
  - Persistent storage with backup strategy

**Cross-Questions You Should Be Ready For:**
- **"Why did you choose ChromaDB over Pinecone/Weaviate/FAISS?"**
  - Answer: ChromaDB is lightweight, open-source, runs locally for development, easy to set up, supports metadata filtering, and has Python-first API. For production, might consider Pinecone for scale or Weaviate for GraphQL support.

- **"How do you handle concurrent ingestion requests?"**
  - Answer: FastAPI handles requests asynchronously, background tasks for heavy processing, queue-based system (Celery) for large-scale ingestion, locking mechanisms to prevent duplicate processing.

- **"What happens if ChromaDB goes down?"**
  - Answer: Graceful degradation with error responses, health checks to detect failures, backup/restore procedures, consider running ChromaDB in HA mode or using managed alternatives.

- **"How do you ensure data consistency?"**
  - Answer: Transactional operations where possible, idempotent ingestion (same document = same ID), versioning of documents, audit logs for tracking changes.

---

### Q2: "What is chunking and why is it important in RAG systems?"

**Expected Answer Coverage:**
- **Definition:** Breaking large documents into smaller, semantically meaningful pieces (chunks)
- **Why Important:**
  - Embedding models have token limits (8K for ada-002)
  - Smaller chunks = more precise retrieval
  - Enables granular similarity matching
  - Better context relevance for LLM

- **Chunking Strategy:**
  - Fixed-size chunks with character/token limits
  - Overlap between chunks (e.g., 10-20%) to preserve context
  - Paragraph or sentence-based boundaries preferred

- **Trade-offs:**
  - **Small chunks (200-300 tokens):** Precise but may lack context
  - **Large chunks (800-1000 tokens):** More context but noisy retrieval
  - **Overlap:** Helps continuity but increases storage

**Cross-Questions:**
- **"What chunk size did you choose and why?"**
  - Answer: Started with 1000 characters (~250 tokens) with 50-character overlap. Chosen based on balancing context (need enough for meaning) and precision (not too much noise). Will A/B test different sizes.

- **"How did you determine the overlap percentage?"**
  - Answer: 5-10% overlap (50 out of 1000 chars) provides context continuity without excessive duplication. Ensures sentences split across boundaries appear in both chunks.

- **"Have you experimented with semantic chunking vs fixed-size chunking?"**
  - Answer: Currently using fixed-size for simplicity. Semantic chunking (by paragraphs, sections) is planned enhancement. Would use NLP libraries like spaCy for sentence detection.

- **"What happens if a sentence is split across chunks?"**
  - Answer: Overlap helps—the sentence appears partially in both chunks. For better handling, would implement sentence-boundary detection to avoid mid-sentence splits.

- **"How do you handle code snippets vs prose differently?"**
  - Answer: Code requires different chunking—preserve function/class boundaries. Would detect code blocks and use language-specific parsers. Separate chunk size for code (larger to keep functions intact).

---

### Q3: "Explain text embeddings. What embedding model are you using?"

**Expected Answer Coverage:**
- **What are Embeddings:**
  - Dense vector representations of text (e.g., 1536 dimensions)
  - Capture semantic meaning, not just keywords
  - Similar meanings = similar vectors (measured by cosine similarity)
  - Positions in high-dimensional space reflect relationships

- **Model Used:**
  - OpenAI's `text-embedding-ada-002` or `text-embedding-3-small`
  - 1536 dimensions for ada-002
  - Cost: $0.0001 per 1K tokens
  - State-of-the-art performance on MTEB benchmarks

- **How They Work:**
  - Trained on massive text corpora
  - Learn contextual relationships
  - "King - Man + Woman ≈ Queen" (semantic algebra)

**Cross-Questions:**
- **"Why not use open-source models like sentence-transformers?"**
  - Answer: Could use `all-MiniLM-L6-v2` or `all-mpnet-base-v2` for cost savings and offline operation. OpenAI embeddings offer superior quality but at cost. For production, would benchmark both.

- **"How do embeddings capture semantic similarity?"**
  - Answer: Through training on tasks like predicting context words, next sentence prediction, contrastive learning. Similar contexts push vectors closer in space.

- **"What's the cost consideration for OpenAI embeddings?"**
  - Answer: $0.0001/1K tokens. For 1M chunks (avg 250 tokens each), cost ~$25. Need to budget for re-embedding on model updates. Consider caching and batch processing.

- **"Have you considered fine-tuning embeddings for your domain?"**
  - Answer: Yes, for specialized domains (medical, legal), fine-tuning on domain data improves relevance. Would use sentence-transformers with domain-specific pairs or OpenAI fine-tuning API.

- **"How do you handle embedding API rate limits?"**
  - Answer: Batch requests (up to 100 per call), implement exponential backoff, queue system for large jobs, consider self-hosted models for high volume.

---

### Q4: "How does vector similarity search work in your system?"

**Expected Answer Coverage:**
- **Process:**
  1. User query gets embedded (same model as documents)
  2. ChromaDB performs nearest neighbor search
  3. Computes cosine similarity between query vector and all document vectors
  4. Returns top-K most similar chunks (typically K=3-5)
  5. Optionally applies metadata filters

- **Distance Metrics:**
  - **Cosine Similarity:** Measures angle between vectors (0-1, higher = more similar)
  - **Euclidean (L2):** Straight-line distance in space
  - **Dot Product:** Combines magnitude and direction

- **Optimization:**
  - Approximate Nearest Neighbor (ANN) algorithms for scale (HNSW, IVF)
  - Indexing structures for faster search
  - Trade-off between accuracy and speed

**Cross-Questions:**
- **"What's the difference between cosine similarity and L2 distance?"**
  - Answer: Cosine measures direction (angle), ignoring magnitude—good for text where length varies. L2 measures absolute distance—sensitive to magnitude. For normalized embeddings, they're equivalent.

- **"How do you determine the optimal K value?"**
  - Answer: Experiment with different K values (3, 5, 10). Too low = miss relevant context. Too high = introduce noise. Depends on chunk size and document diversity. Would use retrieval metrics (Recall@K).

- **"What if the retrieved chunks are irrelevant?"**
  - Answer: Implement similarity threshold (e.g., reject if cosine < 0.7), add re-ranking model, use hybrid search (vector + keyword), provide "confidence score" to user, allow feedback loop.

- **"How do you handle multi-query scenarios?"**
  - Answer: Generate multiple query variations (query expansion), embed each and merge results, use weighted averaging, or train a query encoder specifically for retrieval.

- **"What about hybrid search (vector + keyword)?"**
  - Answer: Combine vector search with BM25 (keyword ranking). ChromaDB supports this. Use weighted combination (e.g., 70% vector, 30% keyword) for best of both worlds.

---

### Q5: "Walk me through your ingestion pipeline."

**Expected Answer Coverage:**
- **Step-by-Step Flow:**
  1. **Document Acquisition:** Connectors fetch documents (upload, scrape, API)
  2. **Text Extraction:** Parse file formats (PDF via PyPDF2, DOCX, HTML)
  3. **Preprocessing:** Clean text, normalize whitespace, handle encoding
  4. **Chunking:** Split into overlapping segments with metadata
  5. **Embedding:** Generate vectors for each chunk
  6. **Storage:** Save to ChromaDB with metadata (source, timestamp, chunk_index)
  7. **Indexing:** Build search indices for fast retrieval

- **Error Handling:**
  - Validation at each step
  - Logging failures
  - Partial success handling (some chunks succeed)
  - Retry failed chunks

**Cross-Questions:**
- **"How do you handle different file formats?"**
  - Answer: Strategy pattern with connectors per format. PDF: PyPDF2/pdfplumber, DOCX: python-docx, HTML: BeautifulSoup, TXT: direct read. Factory pattern to select connector based on MIME type.

- **"What's your strategy for incremental updates?"**
  - Answer: Document versioning with unique IDs, detect changes via hash comparison, delete old chunks and re-ingest updated documents, timestamp-based querying for recent updates.

- **"How do you de-duplicate documents?"**
  - Answer: Content hash (SHA-256 of text), check if document already exists before ingestion, for near-duplicates use MinHash or SimHash, maintain document registry.

- **"What metadata do you store with each chunk?"**
  - Answer:
    - `document_id`, `chunk_index`, `source_url/file_path`
    - `timestamp`, `document_title`, `author`
    - `start_char`, `end_char` (position in original doc)
    - `chunk_size`, `overlap_size`
    - Custom tags/categories for filtering

- **"How do you handle failed ingestion jobs?"**
  - Answer: Dead letter queue for failed chunks, retry with exponential backoff, alert on persistent failures, manual review interface, job status tracking.

---

## Technical Implementation Questions

### Q6: "How do you handle API failures (OpenAI, ChromaDB)?"

**Expected Answer Coverage:**
- **Retry Logic:**
  - Exponential backoff (1s, 2s, 4s, 8s...)
  - Max retry attempts (3-5)
  - Retry only on transient errors (429, 500, 503)
  - Don't retry on 400, 401, 403

- **Circuit Breaker Pattern:**
  - After N consecutive failures, open circuit
  - Fail fast without calling API
  - Periodically test if service recovered

- **Fallback Mechanisms:**
  - Cache previous results
  - Degraded mode (return partial results)
  - Alternative providers

- **Monitoring:**
  - Log all errors with context
  - Alert on high error rates
  - Track API health metrics

**Cross-Questions:**
- **"What's your retry strategy?"**
  - Answer: Implement with `tenacity` library. Exponential backoff with jitter to prevent thundering herd. Max 3 retries for embeddings, 5 for vector store.

- **"How many retries before giving up?"**
  - Answer: 3-5 depending on operation cost. Embeddings: 3 (can requeue). Vector insert: 5 (critical operation). Query: 2 (user waiting).

- **"Do you queue failed requests?"**
  - Answer: Yes, failed ingestion goes to Celery queue for later retry. Failed queries return error to user. Use Redis or RabbitMQ as message broker.

- **"How do you monitor API health?"**
  - Answer: Health check endpoints, probe ChromaDB/OpenAI periodically, track response times and error rates, integrate with Prometheus + Grafana, alerting via PagerDuty/Slack.

---

### Q7: "Explain your FastAPI endpoint design."

**Expected Answer Coverage:**
- **RESTful Principles:**
  - `/ingest` (POST) - create ingestion job
  - `/query` (POST) - search and answer
  - `/documents/{id}` (GET, DELETE) - manage documents
  - `/health` (GET) - system status

- **Request/Response Schemas:**
  - Pydantic models for validation
  - Type hints throughout
  - Automatic OpenAPI generation
  - Error response standards

- **Dependency Injection:**
  - Services injected via `Depends()`
  - Database connections, auth
  - Configuration access

- **Async/Await:**
  - All I/O operations async
  - Non-blocking API calls
  - Better concurrency

**Cross-Questions:**
- **"Why FastAPI over Flask/Django?"**
  - Answer: FastAPI has async support (critical for I/O-heavy RAG), automatic validation with Pydantic, built-in OpenAPI docs, high performance (comparable to Node/Go), modern Python (type hints).

- **"How do you handle authentication?"**
  - Answer: JWT tokens or API keys, OAuth2 with password flow, dependency injection for auth check, scoped permissions (read vs write), rate limiting per user.

- **"What about rate limiting?"**
  - Answer: Use `slowapi` library, limits per IP/user (e.g., 100 requests/hour), different limits for different endpoints (query more limited than health), return 429 with Retry-After header.

- **"How do you validate incoming requests?"**
  - Answer: Pydantic schemas automatically validate, custom validators for complex logic, reject invalid data with 422 status, return clear error messages indicating which fields failed.

- **"What's your error response structure?"**
  - Answer: Consistent format:
    ```json
    {
      "error": "InvalidQuery",
      "message": "Query text cannot be empty",
      "details": { "field": "query_text" },
      "timestamp": "2024-11-02T12:00:00Z"
    }
    ```

---

### Q8: "How would you scale this system?"

**Expected Answer Coverage:**
- **Horizontal Scaling:**
  - Multiple FastAPI instances behind load balancer
  - Stateless design (no session storage in app)
  - Docker containers + Kubernetes orchestration

- **Database Scaling:**
  - ChromaDB in distributed mode or migrate to Pinecone/Weaviate
  - Read replicas for query load
  - Sharding by document category

- **Queue System:**
  - Celery workers for ingestion
  - RabbitMQ/Redis as broker
  - Scale workers independently

- **Caching:**
  - Redis for frequently accessed chunks
  - Cache embeddings for repeated queries
  - CDN for static assets

- **Performance:**
  - Database connection pooling
  - Batch API requests
  - Async I/O throughout
  - Index optimization

**Cross-Questions:**
- **"What's your current bottleneck?"**
  - Answer: Likely embedding API (rate limits, latency). Would batch requests, cache embeddings, consider self-hosted models for high volume.

- **"How would you handle millions of documents?"**
  - Answer: Partition by category/time, use distributed vector DB (Milvus, Weaviate cluster), implement multi-stage retrieval (coarse then fine), archive old documents.

- **"What about geo-distributed deployments?"**
  - Answer: Deploy in multiple regions, replicate vector DB across regions, use geo-routing for low latency, handle eventual consistency, CDN for static content.

- **"How do you ensure consistency across replicas?"**
  - Answer: Primary-replica pattern with async replication, eventual consistency acceptable for RAG (queries can hit any replica), critical updates go to primary, use distributed consensus (Raft) if needed.

---

### Q9: "What about testing strategy?"

**Expected Answer Coverage:**
- **Unit Tests:**
  - Test chunker logic with various inputs
  - Test embedder with mocked API
  - Test vector store operations
  - Pydantic schema validation

- **Integration Tests:**
  - Full ingestion pipeline
  - End-to-end query flow
  - API endpoint tests
  - Database integration

- **Mocking:**
  - Mock OpenAI API responses
  - Mock ChromaDB operations
  - Fixture data for tests
  - Test databases

- **Test Coverage:**
  - Aim for 80%+ coverage
  - Critical paths 100%
  - Use pytest + coverage.py

**Cross-Questions:**
- **"How do you test embedding quality?"**
  - Answer: Similarity tests (known similar texts should have high cosine similarity), benchmark against labeled pairs, compare with baseline models, qualitative review of edge cases.

- **"What's your test coverage goal?"**
  - Answer: 80% overall, 100% for critical paths (ingestion, query, embedding). Use `pytest-cov` to track. Exclude simple getters/setters.

- **"How do you test retrieval accuracy?"**
  - Answer: Create test dataset with questions and expected chunks, measure Recall@K and MRR, use BEIR benchmark datasets, human evaluation for edge cases.

- **"Do you have performance benchmarks?"**
  - Answer: Benchmark ingestion throughput (docs/second), query latency (p50, p95, p99), embedding generation time, vector search time. Use `pytest-benchmark` and track over time.

---

### Q10: "Security considerations in your RAG system?"

**Expected Answer Coverage:**
- **API Security:**
  - Keys stored in environment variables, never in code
  - Rotate keys regularly
  - Use secrets manager (AWS Secrets, Azure Key Vault)

- **Input Validation:**
  - Sanitize all user inputs
  - Prevent injection attacks
  - Limit input sizes
  - Content filtering

- **Access Control:**
  - Authentication on all endpoints
  - Role-based access (admin vs user)
  - Document-level permissions
  - Audit logging

- **Data Security:**
  - Encryption at rest (database)
  - Encryption in transit (HTTPS/TLS)
  - PII detection and masking
  - Compliance (GDPR, HIPAA)

**Cross-Questions:**
- **"How do you prevent prompt injection?"**
  - Answer: Sanitize user queries, use system prompts that can't be overridden, separate user input from instructions, content filtering, limit query length, monitor for suspicious patterns.

- **"What about PII in documents?"**
  - Answer: Run PII detection before ingestion (using libraries like `presidio`), mask/redact sensitive data, document retention policies, allow users to delete their data, compliance with privacy laws.

- **"How do you handle multi-tenant scenarios?"**
  - Answer: Separate collections per tenant in ChromaDB, tenant ID in all queries, data isolation at DB level, separate API keys per tenant, resource quotas per tenant.

- **"Do you log user queries? Privacy concerns?"**
  - Answer: Log for debugging/improvement but with caution. Anonymize logs, short retention period, encrypt logs, allow users to opt out, be transparent in privacy policy, comply with regulations.

---

## Advanced/Tricky Questions

### Q11: "What is the 'Lost in the Middle' problem in RAG?"

**Expected Answer:**
LLMs tend to focus on information at the **beginning** and **end** of the provided context, often ignoring or giving less weight to information in the **middle**. This was demonstrated in research showing recall drops significantly for facts in the middle of long contexts.

**Impact on RAG:**
If you retrieve 10 chunks and concatenate them, the LLM might miss crucial information in chunks 4-7.

**Cross-Questions:**
- **"How would you mitigate this?"**
  - Answer:
    - Reorder chunks: place most relevant at beginning and end
    - Limit context to 3-5 chunks max
    - Use multiple queries with subset of chunks
    - Summarize middle chunks
    - Use advanced prompting (instruct model to read all carefully)

- **"Should you reorder retrieved chunks?"**
  - Answer: Yes, by relevance score (highest score first and last), by recency (recent docs at start), or by diversity (alternate topics). Experiment to find what works best.

---

### Q12: "How do you evaluate RAG system performance?"

**Expected Answer Coverage:**

**Retrieval Metrics:**
- **Recall@K:** % of relevant chunks in top-K results (higher is better)
- **Precision@K:** % of retrieved chunks that are relevant
- **MRR (Mean Reciprocal Rank):** 1 / rank of first relevant chunk
- **NDCG:** Normalized discounted cumulative gain (considers ranking quality)

**Generation Metrics:**
- **BLEU/ROUGE:** Compare generated answer to reference (n-gram overlap)
- **BERTScore:** Semantic similarity using embeddings
- **Human evaluation:** Relevance, correctness, fluency
- **Factual accuracy:** Check if answer contradicts context

**End-to-End Metrics:**
- **Answer Correctness:** Is the answer right?
- **Answer Relevance:** Does it address the question?
- **Context Relevance:** Are retrieved chunks pertinent?
- **Latency:** Time to answer (p50, p95, p99)

**Cross-Questions:**
- **"What's more important: retrieval quality or generation quality?"**
  - Answer: Retrieval is MORE critical. "Garbage in, garbage out"—even the best LLM can't answer correctly with wrong context. Focus on optimizing retrieval first, then generation.

- **"How do you A/B test changes?"**
  - Answer: Split traffic between versions, use consistent test set of queries, measure delta in metrics, statistical significance testing, gradual rollout (10% → 50% → 100%), monitor user feedback.

---

### Q13: "What's the context window limitation problem?"

**Expected Answer:**
LLMs have **token limits**:
- GPT-3.5: 4K tokens
- GPT-4: 8K or 32K tokens
- GPT-4 Turbo: 128K tokens

If you retrieve 10 chunks of 500 tokens each = 5000 tokens. Add system prompt, user query, and there's not much room left, or you exceed the limit.

**Challenges:**
- Can't fit all relevant context
- Costs scale with context length
- Latency increases with longer context

**Cross-Questions:**
- **"How do you handle this?"**
  - Answer:
    - Retrieve top-3 chunks instead of 10
    - Compress/summarize chunks before sending
    - Use re-ranking to select most relevant subset
    - Truncate chunks to fit limit
    - Use models with larger context (GPT-4 Turbo, Claude)

- **"Do you summarize chunks?"**
  - Answer: Yes, can use extractive summarization (pull key sentences) or abstractive (LLM generates summary). Trade-off: lose detail but fit more context. Test if performance improves.

- **"Re-ranking strategies?"**
  - Answer: After initial retrieval, use a cross-encoder model (BERT-based) to re-score query-chunk pairs. More computationally expensive but more accurate. Select top-N after re-ranking.

---

### Q14: "Explain the concept of retrieval vs generation in RAG."

**Expected Answer:**
RAG has two distinct phases:

**Retrieval Phase:**
- Find relevant information from knowledge base
- Vector similarity search
- Goal: High recall (find all relevant info)
- Controlled by: chunk size, top-K, similarity threshold

**Generation Phase:**
- LLM generates answer using retrieved context
- Given context + query, produce coherent response
- Goal: Accurate, fluent, grounded answer
- Controlled by: prompt engineering, model choice, temperature

**Why Separate?**
- Different models: retrieval uses embeddings, generation uses LLM
- Different objectives: retrieval maximizes relevance, generation maximizes quality
- Can optimize each independently

**Cross-Questions:**
- **"Can you use the same model for both?"**
  - Answer: Yes, some models like RAG (the model) or Atlas do joint retrieval-generation. But typically separate models perform better—specialized for their task.

- **"What if retrieval fails?"**
  - Answer: Generation will hallucinate or return "I don't know". Better to detect low-confidence retrieval (low similarity scores) and reject the query or ask for clarification.

---

### Q15: "What are hallucinations in RAG and how do you prevent them?"

**Expected Answer:**
**Hallucinations:** LLM generates information not present in the retrieved context (makes things up).

**Causes in RAG:**
- Retrieved context doesn't answer the question
- LLM relies on training data instead of provided context
- Ambiguous or contradictory context
- Poor prompt engineering

**Prevention Strategies:**
- **Prompt Engineering:** Explicitly instruct "Only use provided context, don't use outside knowledge"
- **Retrieval Quality:** Ensure high-quality, relevant chunks
- **Confidence Thresholds:** Reject queries with low retrieval scores
- **Post-Processing:** Check if answer facts appear in context (entailment checking)
- **Few-Shot Examples:** Show model how to stick to context
- **Model Choice:** Some models hallucinate less (GPT-4 > GPT-3.5)

**Detection:**
- Use NLI (Natural Language Inference) models to check if answer is entailed by context
- Compare answer against retrieved chunks for factual consistency
- Human review for critical applications

**Cross-Questions:**
- **"How do you measure hallucination rate?"**
  - Answer: Manual evaluation (humans check if answer is supported), automated NLI scoring, compare to ground truth answers, user feedback (upvote/downvote).

- **"Can you eliminate hallucinations entirely?"**
  - Answer: No, LLMs are probabilistic and can always make errors. Can minimize with good practices but not eliminate. Critical applications need human-in-the-loop.

---

## Behavioral Questions

### Q16: "Walk me through a challenging problem you faced in this project."

**Example Answer:**
"One challenge was optimizing retrieval latency. Initially, embedding each query and searching ChromaDB was taking 2-3 seconds, which is too slow for user experience.

**What I did:**
1. Profiled the code and found embedding API call was the bottleneck
2. Implemented caching for repeated queries using Redis (30% of queries were repeats)
3. Switched to batch processing for multiple queries
4. Optimized ChromaDB indexing parameters
5. Reduced latency to ~500ms p95

**Result:** 5x improvement in latency, better user satisfaction."

---

### Q17: "How do you stay updated with RAG and LLM developments?"

**Good Answer:**
- Read research papers (arXiv, especially NeurIPS, ACL)
- Follow key researchers on Twitter (e.g., @karpathy, @omarsar0)
- Read blogs (OpenAI, Anthropic, LangChain)
- Participate in communities (r/MachineLearning, Discord groups)
- Experiment with new models/techniques
- Attend conferences/webinars

---

### Q18: "If you had unlimited resources, how would you improve this system?"

**Ideas to Mention:**
- Fine-tune embedding models on domain data
- Build custom re-ranker models
- Implement multi-modal RAG (images, tables, charts)
- Advanced query understanding (entity recognition, intent detection)
- Conversational memory (multi-turn dialogues)
- Active learning (learn from user feedback)
- Explainability (show which chunks led to answer)
- Multi-language support

---

## Quick Reference: Key Metrics & Numbers

| Metric | Typical Value | Your Target |
|--------|---------------|-------------|
| Chunk Size | 500-1000 tokens | 1000 chars (~250 tokens) |
| Chunk Overlap | 10-20% | 5% (50 chars) |
| Top-K Retrieval | 3-5 chunks | 5 chunks |
| Embedding Dimension | 1536 (ada-002) | 1536 |
| Query Latency (p95) | <1 second | 500ms |
| Ingestion Throughput | 100+ docs/min | TBD |
| Similarity Threshold | 0.7-0.8 | 0.75 |
| Context Window | 4K-128K tokens | Depends on model |

---

## Common Mistakes to Avoid in Interviews

1. **Don't oversimplify:** "RAG just retrieves and generates"—go deeper
2. **Don't ignore trade-offs:** Every choice has pros/cons, acknowledge them
3. **Don't claim perfection:** "My system never hallucinates"—be realistic
4. **Don't skip details:** Explain HOW things work, not just WHAT
5. **Don't forget about production:** Talk about monitoring, scaling, errors
6. **Don't ignore costs:** OpenAI API isn't free, show awareness
7. **Don't memorize:** Understand concepts, explain in your own words

---

## Resources for Further Study

### Papers:
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- "Lost in the Middle" (Liu et al., 2023)
- "REALM: Retrieval-Augmented Language Model Pre-Training" (Guu et al., 2020)

### Tools/Libraries:
- LangChain - RAG framework
- LlamaIndex - Data framework for LLMs
- Haystack - End-to-end NLP framework
- ChromaDB, Pinecone, Weaviate - Vector databases

### Benchmarks:
- BEIR - Retrieval benchmark
- MTEB - Embedding benchmark
- MS MARCO - Q&A benchmark

---

**Good luck with your interviews! Remember: Confidence + Clarity + Depth = Success**
