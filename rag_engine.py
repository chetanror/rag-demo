"""Small, explainable retrieval-augmented generation engine."""

from dataclasses import dataclass
import re
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class DocumentChunk:
    title: str
    text: str
    source: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: DocumentChunk
    score: float


def split_into_chunks(text: str, source: str, chunk_size: int = 900) -> list[DocumentChunk]:
    """Split text on paragraphs while keeping chunks small enough for retrieval."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[DocumentChunk] = []
    for index, paragraph in enumerate(paragraphs, start=1):
        for start in range(0, len(paragraph), chunk_size):
            chunks.append(
                DocumentChunk(
                    title=f"{source} · section {index}",
                    text=paragraph[start : start + chunk_size],
                    source=source,
                )
            )
    return chunks


class RagEngine:
    """TF-IDF retriever plus a cited, extractive answer synthesizer."""

    def __init__(self, documents: Iterable[DocumentChunk]):
        self.documents = list(documents)
        if not self.documents:
            raise ValueError("At least one document chunk is required")
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(chunk.text for chunk in self.documents)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        if not query.strip():
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked_indexes = scores.argsort()[::-1]
        return [
            RetrievedChunk(self.documents[index], float(scores[index]))
            for index in ranked_indexes[:top_k]
            if scores[index] > 0
        ]

    def answer(self, query: str, top_k: int = 3) -> tuple[str, list[RetrievedChunk]]:
        retrieved = self.retrieve(query, top_k=top_k)
        if not retrieved:
            return (
                "I could not find an answer in the uploaded knowledge base. "
                "Try asking about a topic covered by the sources.",
                [],
            )

        query_terms = {
            term.lower()
            for term in re.findall(r"[a-zA-Z0-9]+", query)
            if len(term) > 2
        }
        candidate_sentences: list[tuple[float, int, str]] = []
        for source_index, result in enumerate(retrieved, start=1):
            sentences = re.split(r"(?<=[.!?])\s+", result.chunk.text)
            for sentence in sentences:
                terms = set(re.findall(r"[a-zA-Z0-9]+", sentence.lower()))
                overlap = len(query_terms & terms)
                if sentence.strip() and overlap:
                    candidate_sentences.append(
                        (overlap + result.score, source_index, sentence.strip())
                    )

        candidate_sentences.sort(key=lambda item: item[0], reverse=True)
        selected: list[str] = []
        used_sources: set[int] = set()
        for _, source_index, sentence in candidate_sentences:
            if sentence not in selected:
                selected.append(f"{sentence} [{source_index}]")
                used_sources.add(source_index)
            if len(selected) == 3:
                break

        if not selected:
            selected = [f"{retrieved[0].chunk.text.strip()} [1]"]
            used_sources.add(1)

        answer = " ".join(selected)
        citations = " ".join(
            f"[{index}] {retrieved[index - 1].chunk.title}"
            for index in sorted(used_sources)
        )
        return f"{answer}\n\nSources: {citations}", retrieved


def default_documents() -> list[DocumentChunk]:
    """Return the demo corpus used when no files have been uploaded."""
    source = "workshop-notes.md"
    text = (
        "This demo is a small local knowledge assistant for the RAG exercise. "
        "It answers questions from a local knowledge base and shows the sources used.\n\n"
        "The demo supports markdown and plain-text files. Uploaded files are split into paragraph-sized "
        "chunks and indexed with TF-IDF. The demo does not send documents to an external service.\n\n"
        "For better answers, ask a specific question about the available documents. "
        "When the retriever finds no relevant passage, the app says it does not know instead of inventing an answer.\n\n"
        "The retrieval score is based on cosine similarity between the question and each chunk. "
        "The answer panel shows the top matching passages and attaches numbered citations to the response."
    )
    return split_into_chunks(text, source)