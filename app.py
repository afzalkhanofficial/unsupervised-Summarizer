# app.py
import os
import io
import re
import json
import tempfile
from typing import List, Tuple, Dict
from collections import defaultdict

from flask import (
    Flask, request, render_template_string, send_from_directory,
    redirect, url_for, abort, jsonify, make_response
)
from werkzeug.utils import secure_filename

import numpy as np
import networkx as nx
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

# ---------- Configuration ----------
UPLOAD_DIR = os.environ.get("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_EXT = {"pdf", "txt", "png", "jpg", "jpeg"}

# Gemini API settings (replace with your details or set env var)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")  # set in Render env
GEMINI_ENDPOINT = os.environ.get("GEMINI_API_URL", "https://api.generative.example/v1/response")  # placeholder

app = Flask(__name__, static_folder=".", template_folder=".")

# ----------------- Utilities (text extraction / cleaning) -----------------

def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for pg in reader.pages:
        pages.append(pg.extract_text() or "")
    return "\n".join(pages)

def extract_text_from_image_bytes(img_bytes: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        app.logger.exception("OCR failed")
        return ""

def read_uploaded_file(file_storage) -> Tuple[str, str]:
    """
    Returns (text, saved_filepath)
    """
    filename = secure_filename(file_storage.filename or "upload")
    ext = filename.split(".")[-1].lower()
    saved_path = os.path.join(UPLOAD_DIR, filename)
    file_storage.save(saved_path)

    text = ""
    if ext == "pdf":
        with open(saved_path, "rb") as f:
            raw = f.read()
            text = extract_text_from_pdf_bytes(raw)
    elif ext == "txt":
        with open(saved_path, "rb") as f:
            try:
                text = f.read().decode("utf-8", errors="ignore")
            except:
                text = ""
    elif ext in ("png", "jpg", "jpeg"):
        with open(saved_path, "rb") as f:
            text = extract_text_from_image_bytes(f.read())
    else:
        text = ""
    return text, saved_path

# ----------------- Sentence splitting and TOC filtering -----------------

def strip_leading_numbering(s: str) -> str:
    return re.sub(r"^\s*\d+(\.\d+)*\s*[:\-\)]?\s*", "", s).strip()

def is_toc_like(s: str) -> bool:
    digits = sum(c.isdigit() for c in s)
    if digits >= 10 and len(s) > 80 and not re.search(r"\b(reduce|increase|improve|achieve|eliminate|raise|reach|decrease|enhance)\b", s.lower()):
        return True
    if re.search(r"\bcontents\b", s.lower()):
        return True
    return False

def sentence_split(text: str) -> List[str]:
    text = re.sub(r"\n+", " ", text)
    bulleted = re.split(r"\s+[•o]\s+", text)
    sentences = []
    for chunk in bulleted:
        parts = re.split(r'(?<=[\.\?\!])\s+(?=[A-Z0-9“\'"-])', chunk)
        for p in parts:
            p = p.strip()
            if not p:
                continue
            p = re.sub(r"^[\-\–\•\*]+\s*", "", p)
            p = strip_leading_numbering(p)
            if len(p) < 20:
                continue
            if is_toc_like(p):
                continue
            sentences.append(p)
    return sentences

# ----------------- Summarization core (TF-IDF + TextRank + MMR) -----------------

def build_tfidf(sentences: List[str]):
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2), max_df=0.9, min_df=1)
    mat = vec.fit_transform(sentences)
    return mat

def textrank_scores(sim_mat: np.ndarray, positional_boost: np.ndarray = None) -> Dict[int, float]:
    np.fill_diagonal(sim_mat, 0.0)
    G = nx.from_numpy_array(sim_mat)
    pr = nx.pagerank(G, max_iter=200, tol=1e-6)
    scores = np.array([pr.get(i, 0.0) for i in range(sim_mat.shape[0])], dtype=float)
    if positional_boost is not None:
        scores = scores * (1.0 + positional_boost)
    return {i: float(scores[i]) for i in range(len(scores))}

def mmr(scores_dict: Dict[int, float], sim_mat: np.ndarray, k: int, lambda_param: float = 0.7) -> List[int]:
    n = sim_mat.shape[0]
    indices = list(range(n))
    scores = np.array([scores_dict.get(i, 0.0) for i in indices], dtype=float)
    if scores.max() > 0:
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
    selected: List[int] = []
    candidates = set(indices)
    while len(selected) < k and candidates:
        best = None
        best_score = -1e9
        for i in list(candidates):
            if not selected:
                div = 0.0
            else:
                div = max(sim_mat[i][j] for j in selected)
            mmr_score = lambda_param * scores[i] - (1 - lambda_param) * div
            if mmr_score > best_score:
                best_score = mmr_score
                best = i
        if best is None:
            break
        selected.append(best)
        candidates.remove(best)
    return selected

# ----------------- Goal detection / categorization -----------------

GOAL_METRIC_WORDS = [
    "life expectancy", "mortality", "imr", "u5mr", "mmr",
    "coverage", "immunization", "immunisation", "incidence",
    "prevalence", "%", " per ", "gdp", "reduction", "rate"
]
GOAL_VERBS = [
    "reduce", "reducing", "reduction",
    "increase", "increasing",
    "improve", "improving",
    "achieve", "achieving",
    "eliminate", "eliminating",
    "raise", "raising",
    "reach", "reaching",
    "decrease", "decreasing",
    "enhance", "enhancing"
]

def is_goal_sentence(s: str) -> bool:
    s_lower = s.lower()
    has_digit = any(ch.isdigit() for ch in s_lower)
    if not has_digit:
        return False
    if not any(w in s_lower for w in GOAL_METRIC_WORDS):
        return False
    if not any(v in s_lower for v in GOAL_VERBS):
        return False
    return True

def categorize_sentence(s: str) -> str:
    s_lower = s.lower()
    if is_goal_sentence(s):
        return "key goals"
    if any(w in s_lower for w in ["principle", "values", "equity", "universal access", "right to health"]):
        return "policy principles"
    if any(w in s_lower for w in ["primary care", "secondary care", "tertiary care", "referral", "emergency services", "free drugs"]):
        return "service delivery"
    if any(w in s_lower for w in ["prevention", "preventive", "promotive", "nutrition", "tobacco", "alcohol"]):
        return "prevention & promotion"
    if any(w in s_lower for w in ["human resources for health", "health workforce", "doctors", "nurses", "medical college"]):
        return "human resources"
    if any(w in s_lower for w in ["financing", "financial protection", "insurance", "strategic purchasing", "gdp", "expenditure", "private sector"]):
        return "financing & private sector"
    if any(w in s_lower for w in ["digital health", "ehr", "electronic health", "telemedicine", "information system"]):
        return "digital health"
    if any(w in s_lower for w in ["ayush", "ayurveda", "yoga", "unani", "siddha"]):
        return "ayush integration"
    if any(w in s_lower for w in ["implementation", "way forward", "roadmap", "action plan", "strategy", "governance"]):
        return "implementation"
    return "other"

# ----------------- Summarizer function exposed to app -----------------

def summarize_extractive(raw_text: str, length_choice: str = "medium"):
    cleaned = normalize_whitespace(raw_text)
    sentences = sentence_split(cleaned)
    n = len(sentences)
    if n == 0:
        return [], {"original_sentences": 0, "summary_sentences": 0, "compression_ratio": 0, "original_chars": len(cleaned), "summary_chars": 0}
    # pick target length
    if length_choice == "short":
        ratio, max_s = 0.08, 6
    elif length_choice == "long":
        ratio, max_s = 0.30, 20
    else:
        ratio, max_s = 0.20, 12
    target = min(max(1, int(round(n * ratio))), max_s, n)

    tfidf = build_tfidf(sentences)
    sim = cosine_similarity(tfidf)
    pos_boost = np.zeros(n, dtype=float)
    tr_scores = textrank_scores(sim, positional_boost=pos_boost)

    # Select per-section or global MMR picks (here simplified: global MMR on top-k)
    ranked = sorted(range(n), key=lambda i: -tr_scores.get(i, 0.0))
    topK = min(max(2, target*4), n)
    candidates = ranked[:topK]
    cand_sim = sim[np.ix_(candidates, candidates)]
    cand_scores = {i: tr_scores.get(i, 0.0) for i in candidates}
    local_index_map = {g:i for i,g in enumerate(candidates)}
    local_picks = mmr({local_index_map[g]:cand_scores[g] for g in candidates}, cand_sim, k=target, lambda_param=0.75)
    selected_idxs = [candidates[idx] for idx in local_picks]

    # Force include goal sentences if present
    goal_indices = [i for i, s in enumerate(sentences) if is_goal_sentence(s)]
    goal_indices_sorted = sorted(goal_indices, key=lambda i: tr_scores.get(i, 0.0), reverse=True)
    if goal_indices_sorted:
        max_goal = max(1, min(3, int(0.25 * target)))
        forced_goal = goal_indices_sorted[:max_goal]
    else:
        forced_goal = []
    combined = set(selected_idxs) | set(forced_goal)
    # If too many, keep goals and top non-goals
    if len(combined) > target:
        goal_set = set(forced_goal)
        non_goal = [i for i in combined if i not in goal_set]
        non_goal_sorted = sorted(non_goal, key=lambda i: tr_scores.get(i, 0.0), reverse=True)
        keep_non_goal = non_goal_sorted[: max(0, target - len(goal_set))]
        combined = goal_set | set(keep_non_goal)
    selected = sorted(combined)

    summary_sentences = [sentences[i] for i in selected][:target]
    summary_text = " ".join(summary_sentences)
    stats = {
        "original_sentences": n,
        "summary_sentences": len(summary_sentences),
        "original_chars": len(cleaned),
        "summary_chars": len(summary_text),
        "compression_ratio": int(round(100.0 * len(summary_sentences) / max(1,n))),
    }
    return summary_sentences, stats

def build_structured_summary(summary_sentences: List[str], tone: str = "academic"):
    if tone == "easy":
        def simple(s): return re.sub(r"\([^)]{1,30}\)", "", s).strip()
        processed = [simple(s) for s in summary_sentences]
    else:
        processed = summary_sentences[:]
    abstract = " ".join(processed[:3])
    category_to_sentences = defaultdict(list)
    for s in processed:
        category_to_sentences[categorize_sentence(s)].append(s)
    section_order = [
        ("key goals", "Key Goals"),
        ("policy principles", "Policy Principles"),
        ("service delivery", "Strengthening Healthcare Delivery"),
        ("prevention & promotion", "Prevention & Health Promotion"),
        ("human resources", "Human Resources for Health"),
        ("financing & private sector", "Financing & Private Sector Engagement"),
        ("digital health", "Digital Health & Information Systems"),
        ("ayush integration", "AYUSH Integration"),
        ("implementation", "Implementation & Way Forward"),
        ("other", "Other Important Points"),
    ]
    sections = []
    for key, title in section_order:
        bullets = category_to_sentences.get(key, [])
        if bullets:
            unique = []
            seen = set()
            for b in bullets:
                if b not in seen:
                    seen.add(b)
                    unique.append(b)
            sections.append({"title": title, "bullets": unique})
    return {"abstract": abstract, "sections": sections, "category_counts": {k: len(v) for k, v in category_to_sentences.items()}, "implementation_points": category_to_sentences.get("implementation", [])}

# ----------------- PDF download generator for summary -----------------

def summary_to_pdf_bytes(title: str, abstract: str, sections: List[Dict]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4
    margin = 40
    y = height - margin
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y, title)
    y -= 28
    c.setFont("Helvetica", 10)
    textobj = c.beginText(margin, y)
    textobj.setLeading(14)
    textobj.textLines("Abstract:")
    textobj.textLines(abstract)
    textobj.textLine("")
    for sec in sections:
        textobj.textLine(sec["title"] + ":")
        for b in sec["bullets"]:
            textobj.textLine("- " + b)
        textobj.textLine("")
    c.drawText(textobj)
    c.showPage()
    c.save()
    buf.seek(0)
    return buf.read()

# ----------------- Routes -----------------

@app.route("/")
def index():
    # serve your uploaded index.html (the Med.AI site) if present
    index_path = os.path.join(app.static_folder, "index.html")
    if os.path.exists(index_path):
        return send_from_directory(app.static_folder, "index.html")
    return "<p>Upload an index.html into the project root to use the tailored UI.</p>"

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=False)

@app.route("/ocr_upload", methods=["POST"])
def ocr_upload():
    """
    Endpoint to receive image blobs (camera capture), run OCR, return extracted text and saved file path.
    Expects form field 'image_blob' (file).
    """
    if "image_blob" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["image_blob"]
    ext = f.filename.rsplit(".",1)[-1].lower() if "." in f.filename else "png"
    if ext not in ("png","jpg","jpeg"):
        ext = "png"
    filename = secure_filename(f.filename or f"cam.{ext}")
    saved = os.path.join(UPLOAD_DIR, filename)
    f.save(saved)
    with open(saved, "rb") as fh:
        text = extract_text_from_image_bytes(fh.read())
    return jsonify({"text": text, "saved_path": url_for("uploaded_file", filename=filename)})

@app.route("/summarize", methods=["POST"])
def summarize_route():
    """
    main route called by frontend. Accepts:
     - file (pdf/txt/png/jpg)
     - or image_blob (camera capture)
     and form fields:
       length: short/medium/long
       tone: academic/easy
       show_original: 1/0
       enable_chat: 1/0
    Returns a rendered HTML (summary + embed + chat widget)
    """
    length = request.form.get("length", "medium")
    tone = request.form.get("tone", "academic")
    show_original = request.form.get("show_original", "1") == "1"
    enable_chat = request.form.get("enable_chat", "1") == "1"

    if "file" in request.files and request.files["file"].filename:
        text, saved_path = read_uploaded_file(request.files["file"])
        saved_url = url_for("uploaded_file", filename=os.path.basename(saved_path))
    elif "image_blob" in request.files and request.files["image_blob"].filename:
        f = request.files["image_blob"]
        filename = secure_filename(f.filename or "camera.png")
        path = os.path.join(UPLOAD_DIR, filename)
        f.save(path)
        with open(path, "rb") as fh:
            text = extract_text_from_image_bytes(fh.read())
        saved_url = url_for("uploaded_file", filename=filename)
    else:
        return abort(400, "No file uploaded")

    if not text or len(text.strip()) < 50:
        # fallback: if pdf text extraction failed, attempt OCR on pdf pages (not implemented here)
        # but simply warn the user
        warning = "Could not extract meaningful text from the uploaded file. If you uploaded a scanned PDF use camera or image upload for OCR."
        return render_template_string("<p>{{warning}}</p>", warning=warning)

    summary_sentences, stats = summarize_extractive(text, length_choice=length)
    structured = build_structured_summary(summary_sentences, tone=tone)

    # Render a simple results page (Jinja template inline). This page also contains a small chat UI (JS) that posts to /api/chat
    result_html = """
    <!doctype html>
    <html>
      <head>
        <meta charset="utf-8">
        <title>Summary — {{title}}</title>
        <meta name="viewport" content="width=device-width,initial-scale=1">
        <style>
          body{font-family:Inter,system-ui;margin:0;background:#f8fafc;color:#0f172a}
          .wrap{max-width:1100px;margin:32px auto;padding:24px}
          .top{display:flex;gap:16px;align-items:center;justify-content:space-between}
          .meta{color:#475569;font-size:0.9rem}
          .card{background:white;padding:20px;border-radius:12px;box-shadow:0 8px 30px rgba(2,6,23,0.06);margin-top:18px}
          .grid{display:grid;grid-template-columns:1fr 420px;gap:18px}
          .bullets ul{padding-left:20px}
          iframe.pdf{width:100%;height:600px;border:0;border-radius:8px}
          pre.textpreview{white-space:pre-wrap;max-height:560px;overflow:auto;padding:12px;background:#0f172a;color:#e6f4f1;border-radius:8px}
          .chatbox{display:flex;flex-direction:column;height:600px}
          .chatlog{flex:1;overflow:auto;padding:12px;border-radius:8px;background:#fbfbfd;border:1px solid #eef2f7}
          .chatinput{display:flex;gap:8px;margin-top:10px}
          .chatinput input{flex:1;padding:10px;border-radius:8px;border:1px solid #e6e9ef}
          .btn{background:#0d9488;color:white;padding:8px 12px;border-radius:8px;border:0;cursor:pointer}
        </style>
      </head>
      <body>
        <div class="wrap">
          <div class="top">
            <div>
              <h1 style="margin:0">{{title}}</h1>
              <div class="meta">Automatic extractive summary — TF-IDF + TextRank + MMR | Summary sentences: {{stats.summary_sentences}} | Original sentences: {{stats.original_sentences}} | Compression: {{stats.compression_ratio}}%</div>
            </div>
            <div>
              <form method="post" action="{{url_for('download_summary')}}" style="display:inline">
                <input type="hidden" name="title" value="{{title}}">
                <input type="hidden" name="abstract" value="{{structured.abstract}}">
                <input type="hidden" name="sections_json" value='{{sections_json|tojson}}'>
                <button class="btn">Download PDF</button>
              </form>
            </div>
          </div>

          <div class="card grid">
            <div>
              <h3 style="margin-top:0">Abstract</h3>
              <p>{{ structured.abstract }}</p>

              {% for sec in structured.sections %}
                <div style="margin-top:14px">
                  <h4 style="margin:0">{{ sec.title }}</h4>
                  <div class="bullets">
                    <ul>
                      {% for b in sec.bullets %}
                        <li>{{ b }}</li>
                      {% endfor %}
                    </ul>
                  </div>
                </div>
              {% endfor %}
            </div>

            <div style="display:flex;flex-direction:column;gap:12px">
              {% if show_original %}
                <div class="card" style="padding:12px">
                  <h4 style="margin:0 0 8px 0">Original document</h4>
                  {% if original_ext == 'pdf' %}
                    <iframe class="pdf" src="{{ original_url }}"></iframe>
                  {% else %}
                    <pre class="textpreview">{{ original_text_preview }}</pre>
                  {% endif %}
                </div>
              {% endif %}

              {% if enable_chat %}
                <div class="card chatbox">
                  <h4 style="margin:0 0 8px 0">Ask the AI (Gemini)</h4>
                  <div id="chatlog" class="chatlog"></div>
                  <div class="chatinput">
                    <input id="msg" placeholder="Ask about the uploaded document..." />
                    <button id="send" class="btn">Send</button>
                  </div>
                </div>
              {% endif %}
            </div>
          </div>

          <div style="margin-top:18px" class="meta">If the AI assistant is enabled, queries are sent to the server which forwards them to the configured Gemini API. Please set GEMINI_API_KEY in the environment. </div>

        </div>

        <script>
          const enableChat = {{ 'true' if enable_chat else 'false' }};
          const originalUrl = "{{ original_url }}";
          const docText = `{{ original_text_escaped }}`;

          if (enableChat) {
            const log = document.getElementById("chatlog");
            const input = document.getElementById("msg");
            const btn = document.getElementById("send");

            function append(kind, text){
              const el = document.createElement("div");
              el.style.marginBottom = "8px";
              el.style.padding = "8px";
              el.style.borderRadius = "8px";
              if (kind==="user"){ el.style.background = "#eef2f7"; el.style.textAlign="right"; el.innerText = text; }
              else { el.style.background = "#fff"; el.innerText = text; }
              log.appendChild(el);
              log.scrollTop = log.scrollHeight;
            }

            btn.addEventListener("click", async ()=>{
              const q = input.value.trim();
              if(!q) return;
              append("user", q);
              input.value = "";
              append("ai", "Thinking...");
              const payload = {query: q, doc_text: docText};
              try {
                const r = await fetch("/api/chat", {
                  method:"POST",
                  headers: {"Content-Type":"application/json"},
                  body: JSON.stringify(payload)
                });
                const data = await r.json();
                log.lastChild.innerText = data.answer || "(no reply)";
              } catch(e){
                log.lastChild.innerText = "Error contacting assistant.";
              }
            });
          }
        </script>
      </body>
    </html>
    """
    original_ext = "pdf" if saved_url.lower().endswith(".pdf") else "txt"
    # provide a short preview for non-pdf
    preview = text[:3000] + ("..." if len(text) > 3000 else "")
    return render_template_string(result_html,
                                  title=os.path.basename(saved_path),
                                  stats=stats,
                                  structured=structured,
                                  original_url=saved_url,
                                  original_ext=original_ext,
                                  original_text_preview=preview,
                                  original_text_escaped=text.replace("`","'"),
                                  sections_json=structured["sections"],
                                  show_original=show_original,
                                  enable_chat=enable_chat)

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Proxy endpoint to talk to Gemini API. We send the uploaded document text
    as context + user query. GEMINI_API_KEY must be set as env var.
    This is a very lightweight example — when integrating Gemini, follow their
    official API docs, add token rotation, streaming, and safety checks.
    """
    payload = request.get_json() or {}
    query = payload.get("query", "")
    doc_text = payload.get("doc_text", "")

    if not GEMINI_API_KEY:
        # Fallback: simple local answer using keyword search
        snippet = doc_text[:600]
        fallback = f"(No Gemini key configured) I can search the document snippet:\n\n{snippet[:800]}"
        return jsonify({"answer": fallback})

    # Example request (placeholder). Replace with real Gemini request format.
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "prompt": f"Document context:\n{doc_text[:6000]}\n\nUser question: {query}",
        "max_output_tokens": 512
    }
    try:
        r = requests.post(GEMINI_ENDPOINT, headers=headers, json=body, timeout=30)
        r.raise_for_status()
        data = r.json()
        # adjust this parsing to Gemini response format
        answer = data.get("output_text") or data.get("answer") or json.dumps(data)[:2000]
        return jsonify({"answer": answer})
    except Exception as e:
        app.logger.exception("Gemini call failed")
        return jsonify({"answer": f"Assistant error: {str(e)}"})

@app.route("/download_summary", methods=["POST"])
def download_summary():
    title = request.form.get("title", "Summary")
    abstract = request.form.get("abstract", "")
    sections_json = request.form.get("sections_json", "[]")
    try:
        sections = json.loads(sections_json) if isinstance(sections_json, str) else sections_json
    except:
        sections = []
    pdf_bytes = summary_to_pdf_bytes(title, abstract, sections)
    resp = make_response(pdf_bytes)
    resp.headers.set("Content-Type", "application/pdf")
    resp.headers.set("Content-Disposition", "attachment", filename=f"{secure_filename(title)}.pdf")
    return resp

# ----------------- Run -----------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
