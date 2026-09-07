# Changelog

## 0.3.0 - 2026-09-07

- Rebuilt the repository around the supplied container-template style.
- Removed the Makefile.
- Removed all RAG, Chroma, embedding, ingestion, and retrieval functionality.
- Added a FastAPI sycophancy benchmark service.
- Added direct local-model execution through Ollama.
- Added an Expo/React Native benchmark dashboard.
- Tightened the event classifier so ordinary errors are not counted as sycophancy.
- Preserved correct-correction controls and the experimental UCTC metric.
