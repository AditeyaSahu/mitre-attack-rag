# MITRE ATT\&CK RAG: Cybersecurity Guideline Generation



A Retrieval-Augmented Generation (RAG) system that retrieves knowledge from

the MITRE ATT\&CK Enterprise framework and generates actionable guidelines for

cybersecurity practitioners.



*Course:* Advanced Topics in Artificial Intelligence and Machine Learning  

*Assignment:* 3 — Design and Evaluate a RAG Model  

*Author:* Aditeya Kumar Sahu







## Research Questions



This project investigates three research questions:



RQ1: How do different retrieval architectures — naive dense retrieval, hybrid sparse-dense retrieval with re-ranking, and agentic retrieval with query decomposition and self-verification — compare on context relevance, answer relevance, and faithfulness when generating cybersecurity guidelines grounded in the MITRE ATT&CK Enterprise knowledge base?



RQ2: To what extent does an agentic retrieval pipeline incorporating query decomposition and self-verification mitigate hallucination compared to single-pass dense retrieval, as measured by RAGAS faithfulness scores?



RQ3: What categories of failure (irrelevant context retrieval, partial answers, unsupported claims) dominate each retrieval architecture, and what are the implications for trustworthy deployment in real cybersecurity operations?



## System Architecture



Three retrieval variants share a common generator and knowledge base:



1 Naive RAG — dense embeddings (MiniLM-L6) + ChromaDB top-k

2 Hybrid RAG— BM25 + dense fusion + cross-encoder re-ranking

3 Agentic RAG — query decomposition → multi-step retrieval → answer generation → self-verification → revision



Generator: Llama 3.3 70B (open-source, via Groq API)  

Evaluator: RAGAS framework (Context Relevance, Answer Relevance, Faithfulness)



## Knowledge Base



Approximately 1,500–3,000 question–answer pairs derived from the MITRE ATT&CK

Enterprise framework, covering all tactics, techniques, sub-techniques,

detections, mitigations, and procedure examples.



## Repository Structure



\`\`\`

mitre-attack-rag/

├── data/                   # Raw scraped data, processed QA pairs, eval set

├── src/                    # Reusable modules

│   ├── scraper.py

│   ├── preprocessing.py

│   ├── embeddings.py

│   ├── retrievers/         # Naive, hybrid, agentic

│   ├── generator.py

│   └── evaluation.py

├── notebooks/              # Demo notebooks (submission)

├── reports/                # PDF report

├── slides/                 # Presentation

└── tests/

\`\`\`



## Reproducing the Results



(Will be filled in as the project develops.)



## Ethical Considerations



The MITRE ATT\&CK framework is enterprise-oriented; this system inherits any bias toward large-organisation contexts. Trustworthiness, source citation, and human-in-the-loop deployment are discussed in the final report.

