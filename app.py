"""
app.py

Flask app for an unsupervised extractive summarizer using:
  - Sentence-BERT (sentence-transformers)
  - Semantic TextRank (PageRank on SBERT similarity graph)

Features:
  - Accepts PDF uploads (.pdf) or plain text (.txt or pasted text)
  - Extracts text from PDFs using pdfplumber
  - Splits into sentences (NLTK)
  - Builds SBERT sentence embeddings (lazy-loaded)
  - Builds similarity graph and runs PageRank -> sentence ranking
  - Returns extractive summary (keeps original order for coherence)
  - Provides simple single-file UI and download for the summary
"""

import os
import io
import math
import tempfile
from typing import List, Tuple, Dict

from flask import Flask, request, render_template_string, abort, jsonify, send_file
import pdfplumber
import nltk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import html as html_module

# Lazy import of SentenceTransformer to avoid heavy startup cost until needed.
# We'll import inside a getter.
SBERT_MODEL_NAME = os.environ.get("SBERT_MODEL", "all-MiniLM-L6-v2")

# ---------- Configuration ----------
MAX_CONTENT_LENGTH = 30 * 1024 * 1024  # 30 MB
ALLOWED_EXTENSIONS = {"pdf", "txt"}
DEFAULT_SUMMARY_RATIO = 0.25
MIN_SENTENCE_WORDS = 3
SIMILARITY_THRESHOLD = 0.1

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# ---------- Ensure NLTK punkt is available ----------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ---------- Lazy model holder ----------
_sbert_model = None


def get_sbert_model():
    """
    Load SentenceTransformer model on first use to reduce import-time memory.
    This keeps the module import light for Gunicorn/Render until the first request.
    """
    global _sbert_model
    if _sbert_model is None:
        try:
            # Import here to avoid heavy imports at module import time
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError(f"sentence-transformers not installed or failed to import: {e}")
        try:
            _sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
        except Exception as e:
            raise RuntimeError(f"Failed to load SBERT model '{SBERT_MODEL_NAME}': {e}")
    return _sbert_model


# ---------- Utilities ----------

def allowed_file(filename: str) -> bool:
    return bool(filename and "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS)


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    text_parts = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
    except Exception as e:
        # log warning and continue with whatever text was collected
        app.logger.warning(f"pdfplumber extraction warning: {e}")
    return "\n\n".join(text_parts).strip()


def split_into_sentences(text: str) -> List[str]:
    raw_sentences = nltk.tokenize.sent_tokenize(text)
    sentences = []
    for s in raw_sentences:
        s_clean = " ".join(s.strip().split())
        if len(s_clean.split()) >= MIN_SENTENCE_WORDS:
            sentences.append(s_clean)
    return sentences


def build_similarity_graph(embeddings: np.ndarray, threshold: float = SIMILARITY_THRESHOLD) -> nx.Graph:
    sim_matrix = cosine_similarity(embeddings)
    n = sim_matrix.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            score = float(sim_matrix[i, j])
            if score > threshold:
                G.add_edge(i, j, weight=score)
    # If graph has no edges (rare), connect weakly so PageRank can run
    if G.number_of_edges() == 0 and n > 1:
        for i in range(n):
            for j in range(i + 1, n):
                G.add_edge(i, j, weight=float(sim_matrix[i, j]) + 1e-6)
    return G


def run_textrank_on_sentences(sentences: List[str], ratio: float = DEFAULT_SUMMARY_RATIO) -> Tuple[List[str], List[int]]:
    if not sentences:
        return [], []
    model = get_sbert_model()
    # encode and normalize embeddings for cosine similarity
    embeddings = model.encode(sentences, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    G = build_similarity_graph(embeddings)
    try:
        scores = nx.pagerank(G, weight="weight")
    except Exception:
        scores = nx.pagerank_numpy(G, weight="weight")
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    n_sentences = len(sentences)
    k = max(1, int(math.ceil(n_sentences * ratio)))
    top_indices = [idx for idx, _ in ranked[:k]]
    top_indices_sorted = sorted(top_indices)
    summary_sentences = [sentences[i] for i in top_indices_sorted]
    return summary_sentences, top_indices_sorted


# ---------- Simple download registry ----------
DOWNLOAD_REGISTRY: Dict[str, str] = {}


# ---------- HTML UI ----------
INDEX_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width,initial-scale=1"/>
    <title>SBERT + Semantic TextRank — Policy Brief Summarizer</title>
    <style>
      body{font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial;padding:2rem;background:#f3f6fb;color:#111}
      .wrap{max-width:960px;margin:0 auto;background:white;border-radius:12px;padding:1.6rem;box-shadow:0 8px 30px rgba(35,40,50,0.07)}
      header{display:flex;align-items:center;gap:1rem}
      h1{margin:0;font-size:1.3rem}
      p.lead{margin:0;color:#556}
      form{margin-top:1rem;display:grid;gap:.75rem}
      .row{display:flex;gap:.5rem;flex-wrap:wrap}
      label{display:inline-block;padding:.6rem .8rem;background:#f1f5f9;border:1px dashed #e2e8f0;border-radius:8px;cursor:pointer}
      input[type=file]{display:none}
      button{padding:.6rem .9rem;border:0;background:#2563eb;color:white;border-radius:8px;cursor:pointer}
      textarea{width:100%;min-height:160px;padding:.6rem;border-radius:8px;border:1px solid #e6eef8}
      .small{font-size:.9rem;color:#445}
      .result{margin-top:1rem;padding:1rem;border-radius:8px;background:#fcfeff;border:1px solid #e6eef8}
      mark{background: #fff8b0}
      footer{margin-top:1rem;color:#657}
      .controls{display:flex;gap:.5rem;align-items:center}
      input[type=number]{width:80px;padding:.4rem;border-radius:8px;border:1px solid #e6eef8}
    </style>
  </head>
  <body>
    <div class="wrap">
      <header>
        <div>
          <h1>SBERT + Semantic TextRank</h1>
          <p class="lead">Extractive summarizer for policy briefs & healthcare documents (PDF or text)</p>
        </div>
      </header>

      <form id="upload-form" method="post" action="/summarize" enctype="multipart/form-data">
        <div class="row">
          <label for="file">Select PDF / TXT
            <input id="file" name="file" type="file" accept=".pdf,.txt">
          </label>

          <label for="text_input">Or paste text</label>
        </div>

        <textarea id="text_input" name="text" placeholder="Paste policy brief or healthcare document text here (optional)"></textarea>

        <div class="row controls">
          <label class="small">Summary ratio (fraction of sentences to keep):</label>
          <input type="number" name="ratio" step="0.05" min="0.05" max="1" value="0.25">
          <button type="submit">Summarize</button>
        </div>

        <p class="small">Tip: For PDFs, text extraction works best on digital PDFs (not scanned images).</p>
      </form>

      <div id="result"></div>

      <footer>
        <p class="small">Server model: <strong>{{ model_name }}</strong>.</p>
      </footer>
    </div>

    <script>
      const form = document.getElementById('upload-form');
      const resultDiv = document.getElementById('result');

      form.addEventListener('submit', async (e) => {
        e.preventDefault();
        resultDiv.innerHTML = "<div class='result small'>Working... this may take a few seconds for large documents.</div>";
        const formData = new FormData(form);
        try {
          const resp = await fetch('/summarize', { method: 'POST', body: formData });
          if (!resp.ok) {
            const txt = await resp.text();
            throw new Error(txt || resp.statusText);
          }
          const data = await resp.json();
          let html = `<div class='result'><h3>Extractive Summary</h3>`;
          html += `<p class='small'><strong>Selected ${data.selected_count} of ${data.total_sentences} sentences (ratio=${data.ratio})</strong></p>`;
          html += `<div>`;
          if (data.highlighted_html) {
            html += `<div style="line-height:1.6">${data.highlighted_html}</div>`;
          } else {
            html += `<pre style="white-space:pre-wrap">${data.summary_text}</pre>`;
          }
          html += `</div>`;
          if (data.download_url) {
            html += `<p><a href="${data.download_url}" download="summary.txt">Download summary.txt</a></p>`;
          }
          html += `</div>`;
          resultDiv.innerHTML = html;
        } catch (err) {
          resultDiv.innerHTML = `<div class="result small" style="color:maroon">Error: ${err.message}</div>`;
        }
      });
    </script>
  </body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML, model_name=SBERT_MODEL_NAME)


@app.route("/summarize", methods=["POST"])
def summarize_route():
    # Parse ratio
    try:
        ratio = float(request.form.get("ratio", DEFAULT_SUMMARY_RATIO))
        if ratio <= 0 or ratio > 1:
            ratio = DEFAULT_SUMMARY_RATIO
    except Exception:
        ratio = DEFAULT_SUMMARY_RATIO

    uploaded_file = request.files.get("file", None)
    raw_text = (request.form.get("text") or "").strip()

    text = ""
    if uploaded_file and uploaded_file.filename:
        filename = uploaded_file.filename
        if not allowed_file(filename):
            return abort(400, "Only PDF and TXT files are supported.")
        try:
            content = uploaded_file.read()
            ext = filename.rsplit(".", 1)[1].lower()
            if ext == "pdf":
                text = extract_text_from_pdf_bytes(content)
            elif ext == "txt":
                try:
                    text = content.decode("utf-8", errors="replace")
                except Exception:
                    text = content.decode("latin1", errors="replace")
        except Exception as e:
            app.logger.error(f"Error reading uploaded file: {e}")
            return abort(400, "Failed to read uploaded file.")
    elif raw_text:
        text = raw_text
    else:
        return abort(400, "No file or text provided. Please upload a PDF/TXT or paste text.")

    if not text.strip():
        return abort(400, "No text could be extracted from the provided input.")

    sentences = split_into_sentences(text)
    if not sentences:
        return abort(400, "Could not split the document into sentences. Try providing plain text or a digital PDF.")

    summary_sentences, selected_indices = run_textrank_on_sentences(sentences, ratio=ratio)

    # Build highlighted HTML (escape for safety)
    selected_set = set(selected_indices)
    highlighted_parts = []
    for idx, sent in enumerate(sentences):
        safe = html_module.escape(sent)
        if idx in selected_set:
            highlighted_parts.append(f"<mark>{safe}</mark>")
        else:
            highlighted_parts.append(safe)
    highlighted_html = " ".join(highlighted_parts)

    summary_text = "\n\n".join(summary_sentences)

    # Save temporary file for download
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", prefix="summary_")
    try:
        tmp.write(summary_text.encode("utf-8"))
        tmp.flush()
        tmp_path = tmp.name
    finally:
        tmp.close()

    DOWNLOAD_REGISTRY[os.path.basename(tmp_path)] = tmp_path
    download_url = f"/download_summary/{os.path.basename(tmp_path)}"

    response = {
        "summary_text": summary_text,
        "highlighted_html": highlighted_html,
        "selected_count": len(summary_sentences),
        "total_sentences": len(sentences),
        "ratio": ratio,
        "download_url": download_url,
    }
    return jsonify(response)


@app.route("/download_summary/<fname>", methods=["GET"])
def download_summary(fname: str):
    path = DOWNLOAD_REGISTRY.get(fname)
    if not path or not os.path.exists(path):
        return abort(404, "File not found or expired.")
    return send_file(path, as_attachment=True, download_name="summary.txt")


# WSGI entrypoint for gunicorn/render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
