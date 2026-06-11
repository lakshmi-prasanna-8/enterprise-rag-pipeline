# Responsible AI & Data Governance

This document describes the data handling, privacy, and responsible AI
principles applied in this RAG pipeline — reflecting practices implemented
across regulated enterprise environments (healthcare, financial services, aviation).

## Data handling

- No personally identifiable information (PII) or protected health information (PHI)
  is stored in the vector index. All sample data is synthetic or publicly licensed.
- In production deployments, documents are pre-processed through a PII redaction
  step before chunking (e.g., using AWS Comprehend or Azure Cognitive Services).
- The FAISS index stores only dense vector representations — source text is not
  reconstructable from embeddings alone.

## GDPR compliance

- The `/ingest` endpoint does not accept user-uploaded content in this demo.
- In production, a data retention policy governs how long document chunks
  remain in the index (default: 90 days, configurable).
- Users have the right to request deletion of their data ("right to erasure"),
  implemented via document-level metadata filters on the vector store.

## HIPAA considerations (healthcare deployments)

- PHI must never enter the vector store without de-identification.
- LLM API calls must route through a HIPAA-eligible endpoint
  (Azure OpenAI with BAA, or AWS Bedrock with BAA).
- All API calls are logged with audit trails for compliance review.

## Model transparency

- The system prompt instructs the LLM to answer only from retrieved context
  and to declare when it cannot find an answer — reducing hallucination risk.
- Retrieved source documents are returned with every answer, enabling
  human review and audit.
- RAGAS metrics (faithfulness, answer relevancy) are tracked in MLflow
  for ongoing quality monitoring.

## Bias and fairness

- Retrieval quality is evaluated across document types and topics in the
  benchmark dataset to detect systematic gaps.
- AI Fairness 360 tooling can be integrated for structured fairness audits
  in regulated domains.

## Incident response

- Embedding drift is monitored continuously; alerts fire when KS statistic
  exceeds 0.15 (see `monitoring/drift_detector.py`).
- Production incidents follow a documented runbook stored in the ops wiki.
