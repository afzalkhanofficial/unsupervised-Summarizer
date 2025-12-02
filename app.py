import io
import os
import re
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import networkx as nx
from flask import Flask, request, render_template_string, abort
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader

app = Flask(__name__)

# ---------------------- HTML TEMPLATES ---------------------- #

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Policy Brief Summarizer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f172a;
      color: #e5e7eb;
      margin: 0;
      padding: 0;
    }
    .wrapper {
      max-width: 960px;
      margin: 0 auto;
      padding: 1.5rem;
    }
    .card {
      background: #111827;
      border-radius: 1rem;
      padding: 1.5rem 1.75rem;
      box-shadow: 0 18px 45px rgba(0,0,0,0.6);
      margin-top: 2rem;
    }
    h1 {
      margin: 0 0 0.4rem 0;
      font-size: 1.8rem;
    }
    .subtitle {
      font-size: 0.95rem;
      color: #9ca3af;
      margin-bottom: 1rem;
    }
    .badge {
      display: inline-block;
      font-size: 0.7rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      padding: 0.25rem 0.6rem;
      border-radius: 999px;
      border: 1px solid #2563eb;
      color: #bfdbfe;
      margin-bottom: 0.5rem;
    }
    .section {
      margin-top: 1rem;
      padding-top: 1rem;
      border-top: 1px solid #1f2937;
    }
    label {
      font-size: 0.9rem;
    }
    input[type="file"] {
      width: 100%;
      margin: 0.75rem 0 0.25rem;
    }
    .row {
      display: flex;
      flex-wrap: wrap;
      gap: 1rem;
    }
    .col {
      flex: 1 1 230px;
    }
    .radio-group, .check-group {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
      font-size: 0.9rem;
      margin-top: 0.4rem;
    }
    .radio-item, .check-item {
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }
    .helper {
      font-size: 0.78rem;
      color: #9ca3af;
      margin-top: 0.4rem;
    }
    button[type="submit"] {
      margin-top: 1.4rem;
      background: linear-gradient(135deg,#2563eb,#1d4ed8);
      color: #f9fafb;
      border: none;
      border-radius: 999px;
      padding: 0.6rem 1.6rem;
      font-size: 0.95rem;
      font-weight: 600;
      cursor: pointer;
      box-shadow: 0 12px 30px rgba(37,99,235,0.5);
    }
    button[type="submit"]:hover {
      filter: brightness(1.07);
    }
    @media (max-width: 640px) {
      .card {
        padding: 1.25rem;
      }
    }
  </style>
</head>
<body>
  <div class="wrapper">
    <div class="card">
      <div class="badge">Unsupervised • TF-IDF • TextRank • MMR</div>
      <h1>Policy Brief Summarizer</h1>
      <p class="subtitle">
        Upload a policy brief (PDF / TXT). The system will generate an abstract and structured bullet summary
        tailored for primary healthcare policies and similar documents.
      </p>
      <form action="{{ url_for('summarize') }}" method="post" enctype="multipart/form-data">
        <div class="section">
          <label for="file">Policy Brief (PDF or .txt)</label><br>
          <input id="file" type="file" name="file" accept=".pdf,.txt" required>
          <p class="helper">Works best for structured policy documents, guidelines, and official reports.</p>
        </div>
        <div class="section row">
          <div class="col">
            <label>Summary Length</label>
            <div class="radio-group">
              <label class="radio-item">
                <input type="radio" name="length" value="short">
                <span>Short (highly compressed)</span>
              </label>
              <label class="radio-item">
                <input type="radio" name="length" value="medium" checked>
                <span>Medium (balanced)</span>
              </label>
              <label class="radio-item">
                <input type="radio" name="length" value="long">
                <span>Long (more detailed)</span>
              </label>
            </div>
          </div>
          <div class="col">
            <label>Tone</label>
            <div class="radio-group">
              <label class="radio-item">
                <input type="radio" name="tone" value="academic" checked>
                <span>Academic</span>
              </label>
              <label class="radio-item">
                <input type="radio" name="tone" value="easy">
                <span>Easy English</span>
              </label>
            </div>
          </div>
          <div class="col">
            <label>Extras</label>
            <div class="check-group">
              <label class="check-item">
                <input type="checkbox" name="extra_table" value="1" checked>
                <span>Key data table</span>
              </label>
              <label class="check-item">
                <input type="checkbox" name="extra_chart" value="1" checked>
                <span>Category distribution chart</span>
              </label>
              <label class="check-item">
                <input type="checkbox" name="extra_roadmap" value="1" checked>
                <span>Implementation roadmap</span>
              </label>
            </div>
          </div>
        </div>
        <button type="submit">Generate Summary</button>
      </form>
    </div>
  </div>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{{ title }}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0f172a;
      color: #e5e7eb;
      margin: 0;
      padding: 0;
    }
    .wrapper {
      max-width: 960px;
      margin: 0 auto;
      padding: 1.5rem;
    }
    .card {
      background: #111827;
      border-radius: 1rem;
      padding: 1.5rem 1.75rem;
      box-shadow: 0 18px 40px rgba(0,0,0,0.7);
      margin-top: 2rem;
    }
    h1 {
      margin: 0 0 0.5rem 0;
      font-size: 1.7rem;
    }
    h2 {
      margin-top: 1.2rem;
      margin-bottom: 0.4rem;
      font-size: 1.2rem;
    }
    h3 {
      margin-top: 0.8rem;
      margin-bottom: 0.3rem;
      font-size: 1.05rem;
    }
    .subtitle {
      font-size: 0.9rem;
      color: #9ca3af;
      margin-bottom: 0.5rem;
    }
    .stats {
      font-size: 0.85rem;
      color: #9ca3af;
      margin-bottom: 0.6rem;
    }
    .stats span {
      margin-right: 1rem;
    }
    .section {
      margin-top: 1rem;
      padding-top: 1rem;
      border-top: 1px solid #1f2937;
    }
    p {
      font-size: 0.94rem;
      line-height: 1.6;
    }
    ul {
      padding-left: 1.2rem;
      margin-top: 0.2rem;
      margin-bottom: 0.4rem;
    }
    li {
      font-size: 0.9rem;
      line-height: 1.6;
      margin-bottom: 0.25rem;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 0.5rem;
      font-size: 0.85rem;
    }
    th, td {
      border: 1px solid #1f2937;
      padding: 0.4rem 0.5rem;
      text-align: left;
    }
    th {
      background: #1f2937;
    }
    .note {
      font-size: 0.8rem;
      color: #9ca3af;
      margin-top: 0.4rem;
    }
    .back-link {
      display: inline-block;
      margin-top: 1.3rem;
      font-size: 0.9rem;
      color: #93c5fd;
      text-decoration: none;
    }
    .back-link:hover {
      text-decoration: underline;
    }
    canvas {
      max-width: 100%;
      background: #020617;
      margin-top: 0.6rem;
      border-radius: 0.3rem;
    }
  </style>
</head>
<body>
  <div class="wrapper">
    <div class="card">
      <h1>{{ title }}</h1>
      <p class="subtitle">{{ subtitle }}</p>
      <div class="stats">
        <span><strong>Summary sentences:</strong> {{ stats.summary_sentences }}</span>
        <span><strong>Original sentences:</strong> {{ stats.original_sentences }}</span>
        <span><strong>Compression:</strong> {{ stats.compression_ratio }}%</span>
      </div>

      <div class="section">
        <h2>Abstract</h2>
        <p>{{ abstract }}</p>
      </div>

      {% if sections %}
      <div class="section">
        <h2>Structured Summary</h2>
        {% for sec in sections %}
          <h3>{{ sec.title }}</h3>
          <ul>
            {% for bullet in sec.bullets %}
              <li>{{ bullet }}</li>
            {% endfor %}
          </ul>
        {% endfor %}
      </div>
      {% endif %}

      {% if extras.key_table %}
      <div class="section">
        <h2>Key Data Table</h2>
        <table>
          <thead>
            <tr>
              <th>Category</th>
              <th>Number of Summary Points</th>
            </tr>
          </thead>
          <tbody>
            {% for cat, count in category_counts.items() %}
            <tr>
              <td>{{ cat }}</td>
              <td>{{ count }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
        <p class="note">Categories are derived automatically from the extracted sentences using simple keyword-based grouping (goals, principles, delivery, prevention, HR, finance, digital, AYUSH, implementation, other).</p>
      </div>
      {% endif %}

      {% if extras.goals_chart %}
      <div class="section">
        <h2>Category Distribution Chart</h2>
        <canvas id="catChart" height="160"></canvas>
        <p class="note">This chart shows how the summarized content is distributed across conceptual categories (e.g., key goals, principles, service delivery, financing).</p>
      </div>
      {% endif %}

      {% if extras.roadmap and implementation_points %}
      <div class="section">
        <h2>Implementation Roadmap (Extractive)</h2>
        <ul>
          {% for s in implementation_points %}
            <li>{{ s }}</li>
          {% endfor %}
        </ul>
        <p class="note">These points are automatically selected sentences related to implementation, strategy, governance or “way forward” from the document.</p>
      </div>
      {% endif %}

      <a href="{{ url_for('index') }}" class="back-link">← Summarize another document</a>
    </div>
  </div>

  {% if extras.goals_chart %}
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script>
    const ctx = document.getElementById('catChart').getContext('2d');
    const labels = {{ category_labels|tojson }};
    const data = {{ category_values|tojson }};
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Number of summary points',
          data: data,
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: { ticks: { color: '#e5e7eb' }, grid: { display: false } },
          y: { ticks: { color: '#e5e7eb' }, grid: { color: '#1f2937' } }
        }
      }
    });
  </script>
  {% endif %}
</body>
</html>
"""

# ---------------------- TEXT UTILITIES ---------------------- #

def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def strip_leading_numbering(s: str) -> str:
    return re.sub(r"^\s*\d+(\.\d+)*\s*[:\-\)]?\s*", "", s).strip()

def is_toc_like(s: str) -> bool:
    digits = sum(c.isdigit() for c in s)
    if digits > 6 and len(s) > 80:
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

def extract_text_from_pdf(file_storage) -> str:
    raw = file_storage.read()
    reader = PdfReader(io.BytesIO(raw))
    pages = []
    for pg in reader.pages:
        pages.append(pg.extract_text() or "")
    return "\n".join(pages)

def extract_sections(raw_text: str) -> List[Tuple[str, str]]:
    lines = raw_text.splitlines()
    sections: List[Tuple[str,str]] = []
    current_title = "Front"
    buffer: List[str] = []

    heading_re = re.compile(r"^\s*\d+(\.\d+)*\s+[A-Za-z].{0,120}$")
    short_upper_re = re.compile(r"^[A-Z][A-Z\s\-]{4,}$")

    for ln in lines:
        s = ln.strip()
        if not s:
            buffer.append("")
            continue
        if heading_re.match(s) or (short_upper_re.match(s) and len(s.split()) < 12):
            if buffer:
                sections.append((current_title, " ".join(buffer).strip()))
            title = strip_leading_numbering(s)
            current_title = title[:120]
            buffer = []
        else:
            buffer.append(s)
    if buffer:
        sections.append((current_title, " ".join(buffer).strip()))

    cleaned = [(t, normalize_whitespace(b)) for t, b in sections if b.strip()]
    return cleaned

def detect_title(raw_text: str) -> str:
    for line in raw_text.splitlines():
        s = line.strip()
        if len(s) < 5:
            continue
        if "content" in s.lower():
            break
        # Do NOT strip year; just return the first meaningful line
        return s
    return "Policy Document"


# ---------------------- CORE SUMMARIZATION (TF-IDF + TextRank + MMR) ---------------------- #

def build_tfidf(sentences: List[str]):
    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=1
    )
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

def summarize_extractive(raw_text: str, length_choice: str = "medium"):
    cleaned = normalize_whitespace(raw_text)
    sections = extract_sections(cleaned)

    sentences: List[str] = []
    sent_to_section: List[int] = []
    for si, (title, body) in enumerate(sections):
        sents = sentence_split(body)
        for s in sents:
            sentences.append(s)
            sent_to_section.append(si)

    if not sentences:
        sentences = sentence_split(cleaned)
        sent_to_section = [0] * len(sentences)
        sections = [("Document", cleaned)]

    n = len(sentences)
    if n <= 3:
        summary_sentences = sentences
        summary_text = " ".join(summary_sentences)
        stats = {
            "original_sentences": n,
            "summary_sentences": len(summary_sentences),
            "original_chars": len(cleaned),
            "summary_chars": len(summary_text),
            "compression_ratio": 100,
        }
        return summary_sentences, stats

    if length_choice == "short":
        ratio, max_s = 0.10, 6
    elif length_choice == "long":
        ratio, max_s = 0.30, 20
    else:
        ratio, max_s = 0.20, 12

    target = min(max(1, int(round(n * ratio))), max_s, n)

    tfidf = build_tfidf(sentences)
    sim = cosine_similarity(tfidf)

    pos_boost = np.zeros(n, dtype=float)
    sec_first_idx: Dict[int, int] = {}
    for idx, sec_idx in enumerate(sent_to_section):
        sec_first_idx.setdefault(sec_idx, None)
        if sec_first_idx[sec_idx] is None:
            sec_first_idx[sec_idx] = idx
    num_sections = max(sent_to_section) + 1 if sent_to_section else 1
    for sec_idx, first_idx in sec_first_idx.items():
        if first_idx is not None:
            if num_sections > 1:
                weight = 0.06 * (1.0 - (sec_idx / (num_sections - 1)))
            else:
                weight = 0.06
            pos_boost[first_idx] += weight

    tr_scores = textrank_scores(sim, positional_boost=pos_boost)

    # Section importance with extra boost for goals/principles sections
    sec_scores = defaultdict(float)
    for i, sec_idx in enumerate(sent_to_section):
        row = tfidf[i]
        sec_scores[sec_idx] += float(np.linalg.norm(row.toarray()))

    for sec_idx, (title, _) in enumerate(sections):
        t = title.lower()
        boost = 1.0
        if any(w in t for w in ["goal", "objective"]):
            boost *= 1.5
        if "principle" in t:
            boost *= 1.2
        sec_scores[sec_idx] *= boost

    sorted_secs = sorted(sec_scores.items(), key=lambda x: -x[1])
    num_secs = len(sorted_secs)
    per_section_quota = [0] * num_secs

    if target >= num_secs:
        for i in range(min(target, num_secs)):
            per_section_quota[i] = 1
        remaining = target - sum(per_section_quota)
        idx = 0
        while remaining > 0 and num_secs > 0:
            per_section_quota[idx % num_secs] += 1
            idx += 1
            remaining -= 1
    else:
        for i in range(target):
            per_section_quota[i] = 1

    selected_idxs: List[int] = []
    sec_to_global = defaultdict(list)
    sec_order = [s for s, _ in sorted_secs]
    for g_idx, s_idx in enumerate(sent_to_section):
        sec_to_global[s_idx].append(g_idx)

    for rank_pos, sec_idx in enumerate(sec_order):
        quota = per_section_quota[rank_pos] if rank_pos < len(per_section_quota) else 0
        if quota <= 0:
            continue
        candidates = sec_to_global.get(sec_idx, [])
        if not candidates:
            continue
        local_index_map = {g: i for i, g in enumerate(candidates)}
        local_sim = sim[np.ix_(candidates, candidates)]
        local_scores = {i: tr_scores[g] for g, i in local_index_map.items()}
        local_picks = mmr(local_scores, local_sim, min(quota, len(candidates)), lambda_param=0.75)
        for lp in local_picks:
            selected_idxs.append(candidates[lp])

    if len(selected_idxs) < target:
        remaining = target - len(selected_idxs)
        already = set(selected_idxs)
        ranked_global = sorted(range(n), key=lambda i: -tr_scores.get(i, 0.0))
        cand = [i for i in ranked_global if i not in already]
        local_scores = {idx: tr_scores.get(idx, 0.0) for idx in cand}
        local_sim = sim[np.ix_(cand, cand)]
        global_picks = mmr(local_scores, local_sim, min(remaining, len(cand)), lambda_param=0.7)
        for p in global_picks:
            selected_idxs.append(cand[p])

    selected_idxs = sorted(set(selected_idxs))
    summary_sentences = [sentences[i].strip() for i in selected_idxs][:target]
    summary_text = " ".join(summary_sentences)
    stats = {
        "original_sentences": n,
        "summary_sentences": len(summary_sentences),
        "original_chars": len(cleaned),
        "summary_chars": len(summary_text),
        "compression_ratio": int(round(100.0 * len(summary_sentences) / max(1, n))),
    }
    return summary_sentences, stats


# ---------------------- STRUCTURED SUMMARY BUILDING ---------------------- #

def simplify_for_easy_english(s: str) -> str:
    s = re.sub(r"\([^)]{1,30}\)", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def categorize_sentence(s: str) -> str:
    s_lower = s.lower()
    has_digit = any(ch.isdigit() for ch in s_lower)

    # key goals: need numbers + goal/target/health outcome words
    if has_digit and any(w in s_lower for w in [
        "life expectancy", "mortality", "imr", "u5mr", "mmr",
        "coverage", "immunization", "immunisation", "incidence",
        "prevalence", "%", " per ", "by 20", "gdp", "reduction"
    ]):
        return "key goals"

    if any(w in s_lower for w in [
        "principle", "values", "equity", "universal access",
        "universal health", "right to health", "accountability",
        "integrity", "patient-centred", "patient-centered"
    ]):
        return "policy principles"

    if any(w in s_lower for w in [
        "primary care", "secondary care", "tertiary care",
        "health and wellness centre", "health & wellness centre",
        "health & wellness center", "hospital", "service delivery",
        "referral", "emergency services", "free drugs", "free diagnostics"
    ]):
        return "service delivery"

    if any(w in s_lower for w in [
        "prevention", "preventive", "promotive", "promotion",
        "sanitation", "nutrition", "tobacco", "alcohol",
        "air pollution", "road safety", "lifestyle", "behaviour change",
        "behavior change", "swachh", "clean water"
    ]):
        return "prevention & promotion"

    if any(w in s_lower for w in [
        "human resources for health", "hrh", "health workforce",
        "doctors", "nurses", "mid-level", "medical college",
        "nursing college", "public health management cadre",
        "training", "capacity building"
    ]):
        return "human resources"

    if any(w in s_lower for w in [
        "financing", "financial protection", "insurance",
        "strategic purchasing", "public spending", "gdp",
        "expenditure", "catastrophic", "private sector", "ppp",
        "reimbursement", "fees", "empanelment"
    ]):
        return "financing & private sector"

    if any(w in s_lower for w in [
        "digital health", "health information", "ehr",
        "electronic health record", "telemedicine",
        "information system", "surveillance", "ndha", "health data"
    ]):
        return "digital health"

    if any(w in s_lower for w in [
        "ayush", "ayurveda", "yoga", "unani", "siddha", "homeopathy"
    ]):
        return "ayush integration"

    if any(w in s_lower for w in [
        "implementation", "way forward", "roadmap",
        "action plan", "strategy", "governance",
        "monitoring", "evaluation", "framework"
    ]):
        return "implementation"

    return "other"

def build_structured_summary(summary_sentences: List[str], tone: str):
    if tone == "easy":
        processed = [simplify_for_easy_english(s) for s in summary_sentences]
    else:
        processed = summary_sentences[:]

    abstract_sents = processed[:3] if len(processed) >= 3 else processed
    abstract = " ".join(abstract_sents)

    category_to_sentences: Dict[str, List[str]] = defaultdict(list)
    for s in processed:
        cat = categorize_sentence(s)
        category_to_sentences[cat].append(s)

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
            seen = set()
            unique_bullets = []
            for b in bullets:
                if b not in seen:
                    seen.add(b)
                    unique_bullets.append(b)
            sections.append({"title": title, "bullets": unique_bullets})

    category_counts = {k: len(v) for k, v in category_to_sentences.items()}
    implementation_points = category_to_sentences.get("implementation", [])

    return {
        "abstract": abstract,
        "sections": sections,
        "category_counts": category_counts,
        "implementation_points": implementation_points,
    }


# ---------------------- FLASK ROUTES ---------------------- #

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)

@app.route("/summarize", methods=["POST"])
def summarize():
    if "file" not in request.files:
        abort(400, "No file uploaded.")
    f = request.files["file"]
    filename = (f.filename or "").lower()

    if filename.endswith(".pdf"):
        raw_text = extract_text_from_pdf(f)
    elif filename.endswith(".txt"):
        try:
            raw_text = f.read().decode("utf-8", errors="ignore")
        except Exception as e:
            abort(400, f"Error reading text file: {e}")
    else:
        abort(400, "Unsupported format. Please upload a PDF or .txt file.")

    if not raw_text or len(raw_text.strip()) < 50:
        abort(400, "Uploaded document appears to be empty or too short.")

    length_choice = request.form.get("length", "medium").lower()
    tone = request.form.get("tone", "academic").lower()

    extra_table = bool(request.form.get("extra_table"))
    extra_chart = bool(request.form.get("extra_chart"))
    extra_roadmap = bool(request.form.get("extra_roadmap"))

    try:
        summary_sentences, stats = summarize_extractive(raw_text, length_choice=length_choice)
        structured = build_structured_summary(summary_sentences, tone=tone)
    except Exception as e:
        abort(500, f"Error during summarization: {e}")

    doc_title_raw = detect_title(raw_text)
    doc_title = doc_title_raw.strip()
    if doc_title:
        title = f"{doc_title} — Summary"
    else:
        title = "Policy Brief Summary"

    subtitle = "Automatic extractive summary organized as abstract and bullet points (unsupervised: TF-IDF + TextRank + MMR)."

    extras = {
        "key_table": extra_table,
        "goals_chart": extra_chart,
        "roadmap": extra_roadmap,
    }

    category_counts = structured["category_counts"]
    if category_counts:
        labels = list(category_counts.keys())
        values = [category_counts[k] for k in labels]
    else:
        labels, values = [], []

    return render_template_string(
        RESULT_HTML,
        title=title,
        subtitle=subtitle,
        abstract=structured["abstract"],
        sections=structured["sections"],
        stats=stats,
        extras=extras,
        implementation_points=structured["implementation_points"],
        category_counts=category_counts,
        category_labels=labels,
        category_values=values,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
