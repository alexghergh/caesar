import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.tools import tool, ToolRuntime



@dataclass
class RagIndex:
    vectorstore: FAISS
    index_dir: str
    manifest_path: str

    def save(self) -> None:
        self.vectorstore.save_local(self.index_dir)

    def retrieve(
        self,
        query: str,
        k: int = 4,
        scope: str = "global",
        problem_id: int | None = None,
    ) -> str:
        docs = self.vectorstore.similarity_search(query, k=3*k)

        if scope == "problem" and problem_id is not None:
            docs = [
                doc for doc in docs
                if doc.metadata.get("problem_id") == int(problem_id)
            ]

        docs = docs[:k]
        if not docs:
            return ""

        return "\n\n".join(_format_doc(doc) for doc in docs)


def build_or_load_rag_index(
    docs_dir: str,
    index_dir: str,
    manifest_path: str,
    embeddings_model: str = "text-embedding-3-small",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> RagIndex:
    docs_dir_path = Path(docs_dir)
    index_dir_path = Path(index_dir)
    manifest = _build_manifest(docs_dir_path)

    if index_dir_path.exists() and _manifest_matches(manifest, manifest_path):
        vectorstore = FAISS.load_local(
            index_dir,
            OpenAIEmbeddings(model=embeddings_model),
            allow_dangerous_deserialization=True,
        )
        return RagIndex(vectorstore=vectorstore,
                        index_dir=index_dir,
                        manifest_path=manifest_path)

    docs = _load_documents(docs_dir_path)
    if not docs:
        vectorstore = FAISS.from_documents(
            [Document(page_content="", metadata={})],
            OpenAIEmbeddings(model=embeddings_model),
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        chunks = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(
            chunks,
            OpenAIEmbeddings(model=embeddings_model),
        )

    index_dir_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(index_dir)
    _write_manifest(manifest, manifest_path)

    return RagIndex(vectorstore=vectorstore,
                    index_dir=index_dir,
                    manifest_path=manifest_path)


@tool("rag_retrieve")
def rag_retrieve(query: str,
                 runtime: ToolRuntime = None # ToolRuntime[RefinedPromptContext]
                 ) -> str:
    """
    Retrieve relevant references.
    """
    if runtime is None:
        return ""

    ctx = runtime.context
    rag_index: RagIndex | None = getattr(ctx, "rag_index", None)
    conv_info = getattr(ctx, "conv_info", None)
    scope = getattr(ctx, "rag_scope", "global")
    problem_id = getattr(ctx, "problem_id", None)
    turn = getattr(ctx, "turn", None)
    k = int(getattr(ctx, "rag_top_k", 4))

    if rag_index is None:
        return ""

    context = rag_index.retrieve(
        query=query,
        k=k,
        scope=scope,
        problem_id=problem_id,
    )
    if conv_info is not None and turn is not None:
        conv_info.rag_query[turn] = query
        conv_info.rag_context[turn] = context
        conv_info.rag_scope[turn] = scope
    return context


def _load_documents(docs_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in _iter_doc_paths(docs_dir):
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), autodetect_encoding=True)
        try:
            loaded = loader.load()
        except Exception:
            loaded = []
        for doc in loaded:
            doc.metadata["source"] = str(path)
            doc.metadata["source_type"] = "doc"
        docs.extend(loaded)
    return docs


def _iter_doc_paths(docs_dir: Path) -> Iterable[Path]:
    if not docs_dir.exists():
        return []
    exts = {".pdf", ".txt", ".md"}
    return [p for p in docs_dir.rglob("*") if p.suffix.lower() in exts]


def _build_manifest(docs_dir: Path) -> dict:
    entries = []
    for path in _iter_doc_paths(docs_dir):
        stat = path.stat()
        entries.append({
            "path": str(path),
            "mtime": stat.st_mtime,
            "size": stat.st_size,
        })
    entries.sort(key=lambda x: x["path"])
    return {"docs": entries}


def _manifest_matches(manifest: dict, manifest_path: str) -> bool:
    if not os.path.exists(manifest_path):
        return False
    try:
        with open(manifest_path, "r") as f:
            existing = json.load(f)
    except Exception:
        return False
    return existing == manifest


def _write_manifest(manifest: dict, manifest_path: str) -> None:
    Path(os.path.dirname(manifest_path)).mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def _format_doc(doc: Document) -> str:
    source = doc.metadata.get("source", "")
    source_type = doc.metadata.get("source_type", "doc")
    header = f"[source_type: {source_type}]"
    if source:
        header += f" [source: {source}]"
    return f"{header}\n{doc.page_content.strip()}"
