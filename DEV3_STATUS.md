# Developer 3 Status

## Completed

- Gateway routing
- Service health checks
- Streamlit UI
- Demo evaluation
- Real qrels-based dataset evaluation
- Qrels-based MAP, Recall@K, Precision@K, and nDCG@K
- AI Agent smart model routing

## Depends on Developer 1

- Preprocessing
- Indexing
- Datasets
- BM25, TF-IDF, and embedding indexes
- Shared models and constants

## Depends on Developer 2

- Retrieval `/search`
- BM25, TF-IDF, embedding, and hybrid retrieval
- Query refinement

## Known Limitation

- The old 10K TREC-COVID index was local testing only and must not be treated as final evaluation.

## Instructor Notes Compliance

- Real evaluation uses qrels from the selected dataset, not custom queries.
- Demo evaluation is only for formula verification.
- Auto Agent is integrated into Gateway and UI.
- Auto Agent is not used for real evaluation because evaluation needs a fixed model.
- The old 10K trec-covid index was local testing only.
- The final dataset is `quora`, with 522,931 documents, 10,000 queries, and 15,675 qrels.
- `quora` has full BM25, TF-IDF, and Embedding indexes.
- Dev3 depends on Dev1 full indexing and Dev2 retrieval endpoints.

## Recommended Next Step

- Run final qrels-based evaluation on `quora` across all fixed retrieval models.
- Document results in the final report.
