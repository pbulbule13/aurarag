# Nexus Project – Extracted Tech Stack

Extracted from `docs/1stseptnexusproject.txt`.

## Channels and Apps
- Web app, Mobile app
- Slack, Microsoft Teams, WhatsApp, Telegram
- Admin Dashboard with RBAC

## Cloud and Infrastructure
- AWS (EKS for Kubernetes, S3 for object storage, AWS cost calculator)
- Azure (alternative platform mentioned)
- Kubernetes (managed via EKS)

## Storage and Databases
- Object store: Amazon S3
- Vector DB: Pinecone, Chroma DB, MongoDB (vector features)
- Structured data sources: Azure SQL, generic SQL/NoSQL

## Data Sources / Connectors
- SharePoint, Google Drive, OneDrive, S3

## Streaming and Scheduling
- Kafka or AWS Kinesis (either option)
- Apache Airflow (scheduler/orchestration for pipelines)

## LLMs and Model Serving
- LLM APIs: OpenAI (GPT), Google Gemini
- Open-source models: Meta Llama, Mistral
- Serving: vLLM server (self-hosted option)

## Embeddings and Re-ranking
- Cohere (embeddings, re-ranking)
- Mistral (mentioned in class context for embeddings)

## Languages
- Python (primary), also mentions Java, Scala, R in streaming context

## Observability
- Prometheus, Grafana

## BI / Analytics (adjacent integrations)
- Power BI, Tableau

Notes
- The transcript mentions Weaviate/Qdrant implicitly ("Waviate", "Quadrant"). The chosen vector DBs above are Pinecone/Chroma/MongoDB per explicit references.

