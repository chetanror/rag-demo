# Workshop RAG Demo

This is my small custom RAG demo for the AI workshop exercise. I wanted to keep the idea simple enough to understand from one codebase: documents are split into chunks, relevant chunks are found with TF-IDF, and the answer shows the evidence used.

## What it does

The app starts with a short built-in product handbook. You can also upload your own `.md` or `.txt` files and ask questions about them. Everything runs locally, so an API key is not needed. If the documents do not contain an answer, the app says that it could not find one.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. Try the built-in example first, then upload one or more `.md` or `.txt` files from the sidebar.

## Test

```bash
python -m unittest discover -s tests -v
python -m py_compile app.py rag_engine.py
```

## How the pipeline works

1. `split_into_chunks` breaks each uploaded file into paragraph-sized chunks.
2. `TfidfVectorizer` creates sparse vectors for the chunks and the question.
3. Cosine similarity ranks the passages by relevance.
4. The answer method selects relevant sentences and adds numbered source citations.
5. The UI exposes both the answer and the retrieved passages so the result can be inspected.

## Project link

Created by [Chetan Ror](https://github.com/chetanror).

After creating the GitHub repository, add its URL here and submit that link:

`https://github.com/chetanror/rag-demo`

## Limitations and next steps

This version uses TF-IDF and extractive synthesis rather than a hosted LLM. A possible next step would be to add embeddings and an optional LLM provider while keeping the citations and no-answer behavior.