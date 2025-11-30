import os
import io
import math
import fitz  # PyMuPDF
import numpy as np
from flask import Flask, request, redirect, url_for, render_template_string, send_file, abort
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import umap
import hdbscan
import networkx as nx
import nltk
import tempfile
import traceback

# Ensure punkt is available (downloads if needed)
try:
    nltk.data.find("tokenizers/punkt")
except Exception:
    nltk.download("punkt", quiet=True)

from nltk.tokenize import sent_tokenize

# -------------------------
# Configuration
# -------------------------
ALLOWED_EXTENSIONS = {"pdf", "txt"}
MAX_FILE_MB = 25
DEFAULT_SUMMARY_SENTENCES = 5
MODEL_NAME = os.environ.get("SBERT_MODEL", "all-MiniLM-L6-v2")  # lightweight, good tradeoff

# UMAP / HDBSCAN hyperparams tuned for short-to-medium doc summarization
UMAP_N_COMPONENTS = 5
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42

HDBSCAN_MIN_CLUSTER_SIZE = 2
HDBSCAN_METRIC = "euclidean"

MMR_LAMBDA = 0.7  # higher -> more relevance, lower -> more diversity

# -------------------------
# Initialize heavy resources (will download model on first run if necessary)
# -------------------------
print("Loading sentence transformer model:", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)
print("Model loaded.")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_MB * 1024 * 1024

# -------------------------
# Utility functions
# -------------------------


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_pdf(file_stream) -> str:
    """
    Extract text from a file-like object containing a PDF using PyMuPDF.
    Returns concatenated text from all pages.
    """
    try:
        file_stream.seek(0)
        doc = fitz.open(stream=file_stream.read(), filetype="pdf")
        text_parts = []
        for page in doc:
            text = page.get_text("text")
            if text:
                text_parts.append(text)
        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        print("PDF extraction error:", e)
        return ""


def read_text_file(file_stream) -> str:
    try:
        file_stream.seek(0)
        raw = file_stream.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        return raw
    except Exception as e:
        print("Text read error:", e)
        return ""


def split_into_sentences(text: str) -> list:
    """
    Splits text into sentences using nltk punkt.
    Performs simple cleaning and filters out very short sentences.
    """
    raw_sentences = []
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sents = sent_tokenize(paragraph)
        raw_sentences.extend([s.strip() for s in sents if s.strip()])

    # Filter tiny sentences that are unlikely to be useful
    sentences = [s for s in raw_sentences if len(s) >= 20]
    # If everything is filtered out, fall back to raw_sentences
    if not sentences:
        sentences = [s for s in raw_sentences if s]
    return sentences


def build_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    sim = cosine_similarity(embeddings)
    # clip small negatives due to numeric noise
    sim[sim < 0] = 0.0
    np.fill_diagonal(sim, 0.0)
    return sim


def compute_pagerank_scores(sim_matrix: np.ndarray) -> np.ndarray:
    """
    Build a weighted graph from similarity matrix and compute PageRank.
    Returns a score per node (sentence).
    """
    try:
        G = nx.from_numpy_array(sim_matrix)
        # convert 'weight' attribute to numeric if needed
        pr = nx.pagerank_numpy(G, weight="weight")
        # pr is dict {node: score}
        scores = np.array([pr[i] for i in range(len(pr))])
        return scores
    except Exception as e:
        print("PageRank error:", e)
        # fallback: degree or row sums
        fallback = sim_matrix.sum(axis=1)
        if fallback.sum() == 0:
            fallback = np.ones(sim_matrix.shape[0]) / sim_matrix.shape[0]
        return fallback


def mmr_selection(sentences, embeddings, relevance_scores, top_n=5, lambda_param=0.7):
    """
    Maximal Marginal Relevance (MMR)
    - sentences: list of sentence strings (only used for length)
    - embeddings: numpy array (n_sentences x dim)
    - relevance_scores: numpy array of relevance scores (e.g., PageRank)
    - top_n: desired number of items
    - lambda_param: trade-off between relevance and diversity
    Returns list of selected indices in original order of selection (not sorted).
    """
    n = len(sentences)
    if top_n >= n:
        return list(range(n))

    sim = cosine_similarity(embeddings)
    selected = []
    candidates = set(range(n))

    # Normalize relevance scores
    rel = relevance_scores.astype(float)
    if rel.max() > 0:
        rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)

    # pick the highest relevance as first
    first = int(np.argmax(rel))
    selected.append(first)
    candidates.remove(first)

    while len(selected) < top_n and candidates:
        mmr_scores = {}
        for c in candidates:
            # compute max similarity to already selected
            if selected:
                max_sim_to_selected = max(sim[c, s] for s in selected)
            else:
                max_sim_to_selected = 0.0
            mmr_score = lambda_param * rel[c] - (1 - lambda_param) * max_sim_to_selected
            mmr_scores[c] = mmr_score

        # pick argmax
        next_idx = max(mmr_scores.items(), key=lambda x: x[1])[0]
        selected.append(next_idx)
        candidates.remove(next_idx)

    return selected


def summarize_text(text: str, summary_sentences: int = DEFAULT_SUMMARY_SENTENCES, mmr_lambda=MMR_LAMBDA):
    """
    Full pipeline: split -> embed -> reduce -> cluster -> sim matrix -> pagerank -> MMR selection
    Returns:
      - summary_sentences_list: ordered list of sentence strings (in original doc order)
      - meta dict with internal info (cluster labels, scores, etc.)
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return [], {"warning": "No sentences extracted."}

    n_sent = len(sentences)
    # adjust desired summary length if user specified percentage
    if isinstance(summary_sentences, float) and 0 < summary_sentences <= 1:
        k = max(1, math.ceil(n_sent * summary_sentences))
    else:
        k = int(summary_sentences)

    if k >= n_sent:
        # nothing to do
        return sentences, {"note": "Requested summary length >= sentence count; returning full text."}

    # embed
    embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
    # UMAP reduction for HDBSCAN stability (and speed)
    try:
        reducer = umap.UMAP(
            n_components=UMAP_N_COMPONENTS,
            metric=UMAP_METRIC,
            random_state=UMAP_RANDOM_STATE,
        )
        emb_reduced = reducer.fit_transform(embeddings)
    except Exception as e:
        print("UMAP error, falling back to original embeddings:", e)
        emb_reduced = embeddings

    # HDBSCAN clustering (optional, we don't strictly require clusters to create summary)
    try:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE, metric=HDBSCAN_METRIC)
        cluster_labels = clusterer.fit_predict(emb_reduced)
    except Exception as e:
        print("HDBSCAN error, setting all to one cluster:", e)
        cluster_labels = np.zeros(len(sentences), dtype=int)

    # similarity matrix (use original embeddings for similarity)
    sim_matrix = build_similarity_matrix(embeddings)

    # PageRank scores
    pr_scores = compute_pagerank_scores(sim_matrix)

    # We now do MMR selection using PageRank as relevance and embeddings for redundancy penalty
    selected_indices = mmr_selection(sentences, embeddings, pr_scores, top_n=k, lambda_param=mmr_lambda)

    # Order selected sentences in their original order for an extractive readable summary
    selected_indices_sorted = sorted(selected_indices)
    summary_sentences_list = [sentences[i] for i in selected_indices_sorted]

    meta = {
        "n_sent": n_sent,
        "embeddings_shape": embeddings.shape,
        "reduced_shape": emb_reduced.shape,
        "cluster_labels": cluster_labels.tolist(),
        "pagerank_scores": pr_scores.tolist(),
        "selected_indices": selected_indices,
        "selected_indices_sorted": selected_indices_sorted,
    }
    return summary_sentences_list, meta


# -------------------------
# Web UI
# -------------------------
INDEX_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Policy Brief Summarizer (Extractive)</title>
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <style>
      body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; background: #f6f8fa; color: #222; padding: 24px; }
      .container { max-width: 900px; margin: 0 auto; background: white; padding: 24px; border-radius: 10px; box-shadow: 0 8px 30px rgba(10,10,10,0.08); }
      h1 { margin-top:0; }
      label { display:block; margin-top: 12px; font-weight:600; }
      textarea { width:100%; min-height:160px; padding:10px; font-size:14px; border-radius:6px; border:1px solid #ddd; }
      input[type=file] { margin-top:8px; }
      .row { display:flex; gap:12px; align-items:center; margin-top:12px; }
      .small { font-size:0.9rem; color:#666; }
      button { background:#0366d6; color:white; border:none; padding:10px 16px; border-radius:8px; cursor:pointer; }
      .result { margin-top:18px; padding:14px; border-radius:8px; background:#fbfdff; border:1px solid #e6eef8; }
      .meta { font-size:12px; color:#555; margin-top:8px; }
      .sentence { margin:6px 0; padding:6px; border-left:4px solid #e8eefc; background:#fff; border-radius:4px; }
      .highlight { background: #fff7cc; padding: 2px 4px; border-radius:3px; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Automatic Extractive Summarizer — Policy Briefs (Primary Healthcare)</h1>
      <p class="small">Upload a PDF or paste plain text. The pipeline uses SBERT → UMAP → HDBSCAN → PageRank → MMR.</p>

      <form method="POST" action="/summarize" enctype="multipart/form-data">
        <label>Upload PDF (policy brief) or TXT:</label>
        <input type="file" name="file" accept=".pdf,.txt" />

        <label>Or paste plain text:</label>
        <textarea name="text" placeholder="Paste policy brief or healthcare doc here..."></textarea>

        <div class="row">
          <label style="margin:0;">Summary length:</label>
          <input type="number" name="sentences" min="1" value="{{default_sentences}}" style="width:90px;" />
          <span class="small">Number of sentences. Use a small number for short briefs; default is {{default_sentences}}.</span>
        </div>

        <div style="margin-top:12px;">
          <label style="margin:0;">MMR lambda (0..1):</label>
          <input type="number" step="0.05" min="0" max="1" name="mmr_lambda" value="{{default_lambda}}" style="width:90px;" />
          <span class="small">Higher → more relevance, lower → more diversity.</span>
        </div>

        <div style="margin-top:16px;">
          <button type="submit">Summarize</button>
        </div>
      </form>

      <div style="margin-top:18px; color:#666; font-size:13px;">
        <strong>Notes:</strong> Short sentences (under ~20 chars) are removed during preprocessing. If result seems short, increase "summary length" or paste more of the document.
      </div>
    </div>
  </body>
</html>
"""

RESULT_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Summary Result</title>
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <style>
      body { font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; background: #f6f8fa; color: #222; padding: 24px; }
      .container { max-width: 1100px; margin: 0 auto; background: white; padding: 24px; border-radius: 10px; box-shadow: 0 8px 30px rgba(10,10,10,0.08); }
      h1 { margin-top:0; }
      .meta { color:#555; font-size:13px; margin-bottom:12px; }
      .pane { display:flex; gap:18px; align-items:flex-start; }
      .left { flex:1; }
      .right { flex:1; }
      .original { white-space:pre-wrap; padding:12px; border-radius:8px; background:#fcfdff; border:1px solid #eef6ff; max-height:520px; overflow:auto; }
      .summary { padding:12px; border-radius:8px; background:#fffdf6; border:1px solid #fff0c2; }
      .sentence { margin:8px 0; padding:8px; border-radius:6px; background:white; border-left:4px solid #e8eefc; }
      .small { font-size:12px; color:#666; }
      a.btn { display:inline-block; margin-top:12px; background:#0366d6; color:white; padding:8px 12px; border-radius:6px; text-decoration:none; }
      ol { padding-left: 18px; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Extractive Summary</h1>
      <div class="meta">
        <span class="small">Sentences in doc: {{n_sent}} &nbsp; • &nbsp; Selected: {{selected_count}} &nbsp; • &nbsp; PageRank+MMR used.</span>
      </div>

      <div class="pane">
        <div class="left">
          <h3>Summary (extractive)</h3>
          <div class="summary">
            <ol>
            {% for s in summary %}
              <li class="sentence">{{s}}</li>
            {% endfor %}
            </ol>
            <a class="btn" href="/download_summary" target="_blank">Download summary (.txt)</a>
          </div>
          <div style="margin-top:12px;" class="small">Selected indices (in-document order): {{selected_indices}}</div>
        </div>

        <div class="right">
          <h3>Original text (highlights are the selected sentences)</h3>
          <div class="original">
            {% for i, s in enumerate(original_sentences) %}
              {% if i in highlighted %}
                <div style="background:#fff7cc; padding:6px; border-radius:6px; margin-bottom:6px;"><strong>{{i+1}}.</strong> {{s}}</div>
              {% else %}
                <div style="margin-bottom:6px;"><strong>{{i+1}}.</strong> {{s}}</div>
              {% endif %}
            {% endfor %}
          </div>
        </div>
      </div>

      <div style="margin-top:18px;">
        <a href="/" class="btn">Summarize another document</a>
      </div>

    </div>
  </body>
</html>
"""

# temporary storage for download (simple, single-user friendly)
_last_summary_text = None


@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML, default_sentences=DEFAULT_SUMMARY_SENTENCES, default_lambda=MMR_LAMBDA)


@app.route("/summarize", methods=["POST"])
def summarize_route():
    global _last_summary_text
    try:
        # get form inputs
        mmr_lambda = request.form.get("mmr_lambda", type=float)
        if mmr_lambda is None:
            mmr_lambda = MMR_LAMBDA
        else:
            mmr_lambda = max(0.0, min(1.0, float(mmr_lambda)))

        sentences_param = request.form.get("sentences", type=float)
        if sentences_param is None:
            sentences_param = DEFAULT_SUMMARY_SENTENCES
        else:
            # if user typed a float in (0,1), treat as fraction
            if 0 < sentences_param <= 1:
                sentences_param = float(sentences_param)
            else:
                sentences_param = int(max(1, round(sentences_param)))

        # fetch text either from uploaded file or pasted text
        text_input = ""
        uploaded_file = request.files.get("file")
        if uploaded_file and uploaded_file.filename:
            filename = secure_filename(uploaded_file.filename)
            ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""
            if ext == "pdf":
                text_input = extract_text_from_pdf(uploaded_file.stream)
            elif ext == "txt":
                text_input = read_text_file(uploaded_file.stream)
            else:
                return "Unsupported file type. Upload PDF or TXT.", 400

        # if user pasted text, prefer that when provided
        pasted_text = request.form.get("text", "").strip()
        if pasted_text:
            text_input = pasted_text

        if not text_input or not text_input.strip():
            return "No text provided. Upload PDF/TXT or paste text.", 400

        # Run summarization
        summary_sentences, meta = summarize_text(text_input, summary_sentences=sentences_param, mmr_lambda=mmr_lambda)

        # Save last summary for download
        _last_summary_text = "\n".join(summary_sentences)

        # Prepare original sentences for highlighting
        original_sentences = split_into_sentences(text_input)
        highlighted = meta.get("selected_indices_sorted", [])

        return render_template_string(
            RESULT_HTML,
            summary=summary_sentences,
            original_sentences=original_sentences,
            highlighted=highlighted,
            n_sent=meta.get("n_sent", len(original_sentences)),
            selected_count=len(summary_sentences),
            selected_indices=meta.get("selected_indices_sorted", []),
        )

    except Exception as e:
        traceback.print_exc()
        return f"Internal server error: {e}", 500


@app.route("/download_summary", methods=["GET"])
def download_summary():
    global _last_summary_text
    if not _last_summary_text:
        return redirect(url_for("index"))
    # create temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt")
    tmp.write(_last_summary_text.encode("utf-8"))
    tmp.flush()
    tmp.close()
    return send_file(tmp.name, as_attachment=True, attachment_filename="summary.txt")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0"
    print(f"Starting app on {host}:{port}")
    app.run(host=host, port=port)
