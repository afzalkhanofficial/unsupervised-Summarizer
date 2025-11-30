import io
import re
import math
import os
from typing import List, Tuple

from flask import Flask, request, render_template_string, abort, send_file
import numpy as np
import networkx as nx

# sentence-transformers and its dependencies
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# PDF extraction
import fitz  # PyMuPDF

# For robust sentence tokenization with fallback
try:
    import nltk

    _nltk_available = True
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        # try to download punkt quietly; if this fails we'll fall back to regex
        try:
            nltk.download('punkt', quiet=True)
        except Exception:
            pass
except Exception:
    _nltk_available = False

app = Flask(__name__)

# -------------------------
# Configuration / Hyperparams
# -------------------------
SENT_EMBED_MODEL_NAME = os.getenv("SBERT_MODEL", "all-mpnet-base-v2")
# When processing very long documents, we'll only score top_k sentences by PageRank before MMR
CANDIDATE_SENTENCE_CAP = 300
SIM_THRESHOLD = 0.1  # threshold to consider edges in graph (avoid fully connected trivial graphs)
MMR_LAMBDA = 0.7  # balance between relevance and diversity (1.0 -> full relevance)
DEFAULT_SUMMARY_SENTENCES = 5

# Load model once at start
print("Loading SBERT model:", SENT_EMBED_MODEL_NAME)
sbert = SentenceTransformer(SENT_EMBED_MODEL_NAME)
print("Model loaded.")

# -------------------------
# Utilities: text extraction and cleaning
# -------------------------
def extract_text_from_pdf_stream(stream: io.BytesIO) -> str:
    """
    Use PyMuPDF (fitz) to extract text from a PDF stream.
    """
    try:
        doc = fitz.open(stream=stream, filetype="pdf")
        texts = []
        for page in doc:
            texts.append(page.get_text("text"))
        return "\n".join(texts)
    except Exception as e:
        raise RuntimeError(f"Could not extract text from PDF: {e}")

def clean_text(text: str) -> str:
    # Remove excessive whitespace, repeated page headers like "Page 1", common artifacts
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{2,}", "\n\n", text)
    # Remove 'Page X of Y' and 'Page X' footers/headers heuristically
    text = re.sub(r"Page\s*\d+\s*(of\s*\d+)?", "", text, flags=re.IGNORECASE)
    # Remove common multiple dashes/separators
    text = re.sub(r"-{3,}", " ", text)
    # Trim spaces
    text = text.strip()
    return text

def split_into_sentences(text: str) -> List[str]:
    """
    Prefer nltk.sent_tokenize if available; otherwise fallback to a regex-based split.
    """
    text = text.strip()
    if not text:
        return []
    if _nltk_available:
        try:
            from nltk.tokenize import sent_tokenize
            sents = sent_tokenize(text)
            # Filter out very short lines that are likely headings or artifacts (keep > 3 chars)
            sents = [s.strip() for s in sents if len(s.strip()) > 3]
            return sents
        except Exception:
            pass

    # Fallback (simple): split on punctuation followed by whitespace and uppercase (heuristic)
    pattern = re.compile(r'(?<=[\.\?\!])\s+(?=[A-Z0-9\"\'\(\[])')
    sents = pattern.split(text)
    sents = [s.strip() for s in sents if len(s.strip()) > 3]
    return sents

# -------------------------
# Semantic TextRank (PageRank over SBERT similarity graph)
# -------------------------
def build_sentence_embeddings(sentences: List[str]) -> np.ndarray:
    if len(sentences) == 0:
        return np.zeros((0, sbert.get_sentence_embedding_dimension()))
    embeddings = sbert.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
    return embeddings

def build_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.shape[0] == 0:
        return np.array([[]])
    sim = cosine_similarity(embeddings)
    # clip numerical noise
    np.fill_diagonal(sim, 0.0)
    return sim

def semantic_textrank(sentences: List[str], embeddings: np.ndarray, sim_threshold: float = SIM_THRESHOLD) -> np.ndarray:
    """
    Build a graph where nodes are sentences and edge weights are SBERT cosine similarity (if > threshold).
    Run PageRank to score sentences.
    Returns PageRank scores aligned to sentences array.
    """
    n = len(sentences)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    sim = build_similarity_matrix(embeddings)
    # Add edges for pairs above threshold
    for i in range(n):
        for j in range(i + 1, n):
            w = float(sim[i, j])
            if w > sim_threshold:
                G.add_edge(i, j, weight=w)
    # If graph has no edges (all similarities below threshold), add a weak fully connected graph
    if G.number_of_edges() == 0 and n > 0:
        for i in range(n):
            for j in range(i + 1, n):
                G.add_edge(i, j, weight=float(sim[i, j]) + 1e-6)

    # PageRank with weights
    try:
        pr = nx.pagerank(G, weight='weight')
        scores = np.array([pr[i] for i in range(n)])
    except Exception:
        # fallback: average similarity score
        scores = sim.sum(axis=1)
        if scores.sum() > 0:
            scores = scores / scores.sum()
    return scores

# -------------------------
# MMR selection
# -------------------------
def mmr_selection(sentences: List[str],
                  sent_embeddings: np.ndarray,
                  doc_embedding: np.ndarray,
                  top_k: int = 5,
                  lambda_param: float = MMR_LAMBDA,
                  use_initial_scores: np.ndarray = None) -> List[int]:
    """
    Return indices of selected sentences using MMR.
    If use_initial_scores is provided, it's used as the relevance scores (e.g., PageRank).
    Otherwise relevance is cosine similarity to doc embedding.
    """
    if len(sentences) == 0:
        return []
    n = len(sentences)
    # Compute relevance scores
    if use_initial_scores is not None:
        rel = np.array(use_initial_scores, dtype=float)
        # normalize
        if rel.max() - rel.min() > 0:
            rel = (rel - rel.min()) / (rel.max() - rel.min())
    else:
        rel = cosine_similarity(sent_embeddings, doc_embedding.reshape(1, -1)).reshape(-1)

    # Candidate set indices
    candidate_idxs = list(range(n))
    selected = []

    # Start by picking the highest relevance sentence
    first = int(np.argmax(rel))
    selected.append(first)
    candidate_idxs.remove(first)

    # Precompute cosine similarities between sentences
    sim_matrix = cosine_similarity(sent_embeddings)

    while len(selected) < top_k and candidate_idxs:
        mmr_scores = []
        for idx in candidate_idxs:
            relevance = rel[idx]
            diversity = max(sim_matrix[idx, sel] for sel in selected) if selected else 0.0
            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * diversity
            mmr_scores.append((mmr_score, idx))
        # pick max mmr score
        mmr_scores.sort(reverse=True)
        chosen_idx = mmr_scores[0][1]
        selected.append(chosen_idx)
        candidate_idxs.remove(chosen_idx)

    return selected

# -------------------------
# Top-level summarization pipeline
# -------------------------
def summarize_text(text: str, max_sentences: int = DEFAULT_SUMMARY_SENTENCES, ratio: float = None) -> Tuple[str, List[Tuple[int, str]]]:
    """
    Returns tuple: (summary_text, list_of_selected_sentences_as (index, sentence) in original order)
    If ratio is provided (0 < ratio <= 1), it overrides max_sentences and determines number of sentences.
    """
    text = clean_text(text)
    sentences = split_into_sentences(text)
    n = len(sentences)
    if n == 0:
        return "", []

    # decide number of sentences
    if ratio is not None:
        if ratio <= 0 or ratio > 1:
            raise ValueError("ratio must be in (0, 1].")
        num_sentences = max(1, math.ceil(n * ratio))
    else:
        num_sentences = min(max_sentences, n)

    # If document is very short, return original
    if n <= num_sentences:
        return "\n\n".join(sentences), list(enumerate(sentences))

    # Build embeddings
    embeddings = build_sentence_embeddings(sentences)

    # Compute PageRank scores (Semantic TextRank)
    pr_scores = semantic_textrank(sentences, embeddings)

    # Limit candidate set for MMR to top_k by PageRank (helps with performance for long docs)
    top_k_candidates = min(max(num_sentences * 5, 50), CANDIDATE_SENTENCE_CAP, n)
    cand_idxs_sorted = np.argsort(-pr_scores)[:top_k_candidates]
    cand_idxs_sorted = list(map(int, cand_idxs_sorted))

    cand_sentences = [sentences[i] for i in cand_idxs_sorted]
    cand_embeddings = embeddings[cand_idxs_sorted, :]

    # Document embedding (centroid)
    doc_embedding = np.mean(embeddings, axis=0)

    # Use MMR to pick final indices from candidate set
    # We'll supply PageRank scores (on candidate set) as initial relevance
    pr_on_candidates = pr_scores[cand_idxs_sorted]
    selected_in_cand = mmr_selection(cand_sentences, cand_embeddings, doc_embedding, top_k=num_sentences,
                                     lambda_param=MMR_LAMBDA, use_initial_scores=pr_on_candidates)

    # Map selected indices back to original sentence indices
    selected_orig_idxs = [cand_idxs_sorted[i] for i in selected_in_cand]
    selected_orig_idxs_sorted = sorted(selected_orig_idxs)  # keep original document order for readability

    selected_sentences = [(i, sentences[i]) for i in selected_orig_idxs_sorted]
    summary = "\n\n".join([s for (_, s) in selected_sentences])

    return summary, selected_sentences

# -------------------------
# Flask routes and UI
# -------------------------
INDEX_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Policy Brief Summarizer — Extractive (SBERT + TextRank + MMR)</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      body {font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; background:#f6f8fa; color:#111; margin:0; padding:2rem;}
      .container {max-width:980px; margin:0 auto; background:white; border-radius:12px; padding:1.6rem; box-shadow:0 6px 24px rgba(17, 24, 39, 0.06);}
      h1 {margin:0 0 0.5rem 0; font-size:1.5rem;}
      p.lead {margin-top:0; color:#4b5563;}
      label {display:block; margin-top:1rem; font-weight:600;}
      textarea {width:100%; min-height:160px; padding:0.6rem; font-size:0.95rem; border-radius:8px; border:1px solid #e5e7eb;}
      input[type=file] {margin-top:0.5rem;}
      .row {display:flex; gap:1rem; flex-wrap:wrap; align-items:center;}
      .small {font-size:0.9rem; color:#6b7280;}
      .btn {background:#0b74ff; color:white; padding:0.6rem 1rem; border-radius:8px; border:none; cursor:pointer; font-weight:600;}
      .result {margin-top:1rem; white-space:pre-wrap; background:#f8fafc; padding:1rem; border-radius:8px; border:1px solid #e6eef8;}
      .meta {margin-top:0.5rem; color:#6b7280;}
      .footer {margin-top:1rem; font-size:0.85rem; color:#6b7280;}
      input[type=number] {width:120px; padding:0.45rem; border-radius:8px; border:1px solid #e5e7eb;}
      input[type=range] {width:200px;}
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Policy Brief Summarizer — Extractive</h1>
      <p class="lead">Upload a PDF or paste text. This service uses <strong>SBERT embeddings</strong>, <strong>Semantic TextRank</strong>, and <strong>MMR</strong> to produce concise extractive summaries — tuned for policy / healthcare briefs.</p>

      <form action="/summarize" method="post" enctype="multipart/form-data">
        <label>Upload PDF (policy brief) or plain text file (.txt)</label>
        <input type="file" name="file" accept=".pdf,.txt" />

        <label>Or paste the policy brief text here</label>
        <textarea name="pasted_text" placeholder="Paste full text of policy brief or healthcare document..."></textarea>

        <div style="margin-top:0.6rem;" class="row">
          <div>
            <label>Summary length (sentences)</label>
            <input type="number" name="num_sentences" min="1" max="80" value="5" />
            <div class="small meta">If you prefer ratio, use the next field instead.</div>
          </div>

          <div>
            <label>Or summary ratio (0-1)</label>
            <input type="number" name="ratio" step="0.05" min="0.05" max="1.0" placeholder="e.g., 0.15" />
            <div class="small meta">Examples: 0.1 → 10% of sentences. If set, this overrides sentences value.</div>
          </div>

          <div style="align-self:flex-end;">
            <button class="btn" type="submit">Generate Summary</button>
          </div>
        </div>
      </form>

      <div class="footer">
        <p>Notes: For best results, upload a clean policy brief PDF (no scanned images). Scanned PDFs with no selectable text won't be summarized.</p>
      </div>
    </div>
  </body>
</html>
"""

RESULT_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>Summary Result</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      body {font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial; background:#f6f8fa; color:#111; margin:0; padding:2rem;}
      .container {max-width:980px; margin:0 auto; background:white; border-radius:12px; padding:1.6rem; box-shadow:0 6px 24px rgba(17, 24, 39, 0.06);}
      a.btn {background:#0b74ff; color:white; padding:0.6rem 1rem; border-radius:8px; text-decoration:none; font-weight:600;}
      .meta {color:#6b7280; margin-top:0.25rem;}
      pre.summary {background:#f8fafc; padding:1rem; border-radius:8px; border:1px solid #e6eef8; white-space:pre-wrap;}
      ol {padding-left:1.2rem;}
      li {margin-bottom:0.6rem;}
    </style>
  </head>
  <body>
    <div class="container">
      <a class="btn" href="/">← New summary</a>
      <h2>Extractive Summary</h2>
      <div class="meta">Selected {{count}} sentence(s) from document.</div>
      <pre class="summary">{{summary}}</pre>

      {% if selected_sentences %}
      <h3>Selected sentences (in original order)</h3>
      <ol>
        {% for idx, sent in selected_sentences %}
          <li><strong>[{{idx}}]</strong> {{sent}}</li>
        {% endfor %}
      </ol>
      {% endif %}
    </div>
  </body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/summarize", methods=["POST"])
def summarize():
    # Get uploaded file or pasted text
    uploaded = request.files.get("file")
    pasted_text = (request.form.get("pasted_text") or "").strip()

    if uploaded and uploaded.filename != "":
        filename = uploaded.filename.lower()
        file_bytes = uploaded.read()
        if filename.endswith(".pdf"):
            try:
                text = extract_text_from_pdf_stream(io.BytesIO(file_bytes))
            except Exception as e:
                return f"<h3>Error extracting PDF text:</h3><pre>{e}</pre>", 400
        elif filename.endswith(".txt"):
            try:
                text = file_bytes.decode("utf-8", errors="ignore")
            except Exception as e:
                text = str(file_bytes)
        else:
            return "Unsupported file type. Please upload PDF or TXT.", 400
    elif pasted_text:
        text = pasted_text
    else:
        return "No input provided. Upload a PDF/TXT or paste text and try again.", 400

    # get parameters
    ratio_val = request.form.get("ratio", "").strip()
    num_sentences = request.form.get("num_sentences", "").strip()
    ratio = None
    try:
        if ratio_val:
            ratio = float(ratio_val)
    except Exception:
        ratio = None

    try:
        if num_sentences:
            n_sent = int(num_sentences)
        else:
            n_sent = DEFAULT_SUMMARY_SENTENCES
    except Exception:
        n_sent = DEFAULT_SUMMARY_SENTENCES

    # Generate summary
    try:
        summary_text, selected = summarize_text(text, max_sentences=n_sent, ratio=ratio)
    except Exception as e:
        return f"<h3>Summarization Failed</h3><pre>{e}</pre>", 500

    return render_template_string(RESULT_HTML, summary=summary_text, selected_sentences=selected, count=len(selected))

# simple health check
@app.route("/_health")
def health():
    return {"status": "ok", "model": SENT_EMBED_MODEL_NAME}

if __name__ == "__main__":
    # Bind to 0.0.0.0 for Render / container hosting
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
