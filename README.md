# Workshop RAG Demo

This is my small custom RAG demo for the AI workshop exercise. I wanted to keep the idea simple enough to understand from one codebase: documents are split into chunks, relevant chunks are found with TF-IDF, and the answer shows the evidence used.

## Demo and source

- **Live demo:** [rag-demo-chetanror.streamlit.app](https://rag-demo-chetanror.streamlit.app/)
- **GitHub repository:** [github.com/chetanror/rag-demo](https://github.com/chetanror/rag-demo)

## Tech stack

- **Language:** Python
- **User interface:** Streamlit
- **Retrieval:** scikit-learn `TfidfVectorizer`
- **Ranking:** cosine similarity
- **Answer generation:** extractive sentence selection with source citations
- **Testing:** Python `unittest`
- **Deployment:** Streamlit Community Cloud

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

## Deploy a test version

The easiest public test deployment is [Streamlit Community Cloud](https://share.streamlit.io/).

1. Sign in with GitHub and choose **Create app**.
2. Select the repository `chetanror/rag-demo`.
3. Select the `main` branch.
4. Set the main file to `app.py`.
5. Click **Deploy**.

The app does not need secrets or environment variables. After deployment, Streamlit will give you a public URL that you can share for testing.

## Project

Created by [chetanror](https://github.com/chetanror).

[View the repository on GitHub](https://github.com/chetanror/rag-demo)

## Limitations and next steps

This version uses TF-IDF and extractive synthesis rather than a hosted LLM. A possible next step would be to add embeddings and an optional LLM provider while keeping the citations and no-answer behavior.