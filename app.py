import io
import os
import re
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import networkx as nx
from flask import (
    Flask,
    request,
    render_template_string,
    abort,
    make_response,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

app = Flask(__name__)

# ---------------------- HTML TEMPLATES (Tailwind / Med.AI style) ---------------------- #

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Med.AI | Policy Summarizer (Unsupervised)</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ['Inter', 'system-ui', 'sans-serif'],
          },
          colors: {
            teal: {
              50: '#f0fdfa',
              100: '#ccfbf1',
              200: '#99f6e4',
              300: '#5eead4',
              400: '#2dd4bf',
              500: '#14b8a6',
              600: '#0d9488',
              700: '#0f766e',
              800: '#115e59',
              900: '#134e4a',
            }
          }
        }
      }
    }
  </script>
  <link rel="stylesheet"
        href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body class="bg-slate-50 text-slate-800 min-h-screen flex items-center justify-center py-10">

  <div class="max-w-3xl w-full px-4">
    <div class="bg-white/90 shadow-xl shadow-slate-200/80 rounded-2xl border border-slate-200/70 p-8 relative overflow-hidden">
      <div class="absolute -top-24 -right-24 w-64 h-64 bg-teal-100 rounded-full blur-3xl opacity-70 pointer-events-none"></div>
      <div class="absolute -bottom-24 -left-24 w-64 h-64 bg-cyan-100 rounded-full blur-3xl opacity-60 pointer-events-none"></div>

      <div class="relative z-10">
        <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-50 border border-teal-200 text-teal-800 text-xs font-bold uppercase tracking-wide mb-3">
          <span class="w-2 h-2 rounded-full bg-teal-500 animate-pulse"></span>
          Unsupervised • TF-IDF • TextRank • MMR
        </div>
        <h1 class="text-2xl sm:text-3xl font-extrabold text-slate-900 mb-1">
          Automatic Policy Brief Summarization
        </h1>
        <p class="text-sm text-slate-600 mb-6 max-w-xl">
          Upload a primary healthcare policy brief (PDF / TXT). The engine will extract an abstract and
          structured bullet summary optimized for research, viva, and reporting.
        </p>

        <form action="{{ url_for('summarize') }}" method="post" enctype="multipart/form-data" class="space-y-6">
          <div>
            <label for="file" class="block text-sm font-semibold text-slate-800 mb-1">
              Policy Brief File
            </label>
            <div class="flex items-center justify-between gap-3 border border-slate-200 rounded-xl px-4 py-3 bg-slate-50/80 hover:bg-slate-50 transition">
              <div class="flex items-center gap-3">
                <div class="w-9 h-9 rounded-xl bg-teal-600 flex items-center justify-center text-white shadow-md shadow-teal-500/40">
                  <i class="fa-solid fa-file-pdf text-lg"></i>
                </div>
                <div>
                  <p class="text-xs font-medium text-slate-800">Upload PDF or .txt</p>
                  <p class="text-[11px] text-slate-500">Works best for structured policy / guideline documents.</p>
                </div>
              </div>
              <label class="inline-flex items-center px-4 py-2 rounded-full text-xs font-semibold bg-teal-600 text-white cursor-pointer shadow shadow-teal-500/40 hover:bg-teal-700 transition">
                <i class="fa-solid fa-upload mr-2 text-xs"></i> Choose file
                <input id="file" type="file" name="file" accept=".pdf,.txt" class="hidden" required>
              </label>
            </div>
          </div>

          <div class="grid sm:grid-cols-3 gap-4 pt-1">
            <div>
              <p class="text-xs font-semibold text-slate-700 mb-1">Summary Length</p>
              <div class="space-y-1 text-xs">
                <label class="flex items-center gap-2 cursor-pointer">
                  <input type="radio" name="length" value="short" class="text-teal-600" />
                  <span>Short (very compressed)</span>
                </label>
                <label class="flex items-center gap-2 cursor-pointer">
                  <input type="radio" name="length" value="medium" checked class="text-teal-600" />
                  <span>Medium (balanced)</span>
                </label>
                <label class="flex items-center gap-2 cursor-pointer">
                  <input type="radio" name="length" value="long" class="text-teal-600" />
                  <span>Long (more detailed)</span>
                </label>
              </div>
            </div>

            <div>
              <p class="text-xs font-semibold text-slate-700 mb-1">Tone</p>
              <div class="space-y-1 text-xs">
                <label class="flex items-center gap-2 cursor-pointer">
                  <input type="radio" name="tone" value="academic" checked class="text-teal-600" />
                  <span>Academic</span>
                </label>
                <label class="flex items-center gap-2 cursor-pointer">
                  <input type="radio" name="tone" value="easy" class="text-teal-600" />
                  <span>Easy English</span>
                </label>
              </div>
            </div>

            <div>
              <p class="text-xs font-semibold text-slate-700 mb-1">Summary View</p>
              <div class="space-y-1 text-xs">
                <label class="flex items-center gap-2 cursor-pointer">
                  <input type="radio" name="view" value="standard" checked class="text-teal-600" />
                  <span>Standard summary</span>
                </label>
                <label class="flex items-center gap-2 cursor-pointer">
                  <input type="radio" name="view" value="actions" class="text-teal-600" />
                  <span>Key action points</span>
                </label>
                <label class="flex items-center gap-2 cursor-pointer">
                  <input type="radio" name="view" value="research" class="text-teal-600" />
                  <span>Research view (with analytics)</span>
                </label>
              </div>
            </div>
          </div>

          <div class="flex justify-between items-center pt-2">
            <p class="text-[11px] text-slate-500 max-w-xs">
              Unsupervised extractive model (no neural fine-tuning). Safe for viva explanations and methodology diagrams.
            </p>
            <button type="submit"
                    class="inline-flex items-center px-5 py-2.5 rounded-full bg-teal-600 text-white text-sm font-semibold shadow-md shadow-teal-500/40 hover:bg-teal-700 hover:shadow-lg transition">
              <i class="fa-solid fa-wand-magic-sparkles mr-2 text-xs"></i>
              Generate Summary
            </button>
          </div>
        </form>
      </div>
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
  <script src="https://cdn.tailwindcss.com"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ['Inter', 'system-ui', 'sans-serif'],
          },
          colors: {
            teal: {
              50: '#f0fdfa',
              100: '#ccfbf1',
              200: '#99f6e4',
              300: '#5eead4',
              400: '#2dd4bf',
              500: '#14b8a6',
              600: '#0d9488',
              700: '#0f766e',
              800: '#115e59',
              900: '#134e4a',
            }
          }
        }
      }
    }
  </script>
  <link rel="stylesheet"
        href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body class="bg-slate-50 text-slate-800 min-h-screen flex items-center justify-center py-10">

  <div class="max-w-4xl w-full px-4">
    <div class="bg-white shadow-xl shadow-slate-200/80 rounded-2xl border border-slate-200/80 p-7 sm:p-8 relative overflow-hidden">
      <div class="absolute -top-24 -right-24 w-64 h-64 bg-teal-100 rounded-full blur-3xl opacity-60 pointer-events-none"></div>
      <div class="absolute -bottom-24 -left-24 w-64 h-64 bg-cyan-100 rounded-full blur-3xl opacity-60 pointer-events-none"></div>

      <div class="relative z-10">
        <div class="flex flex-wrap items-start justify-between gap-3 mb-4">
          <div>
            <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-50 border border-teal-200 text-teal-800 text-[11px] font-bold uppercase tracking-wide mb-2">
              <span class="w-2 h-2 rounded-full bg-teal-500 animate-pulse"></span>
              Unsupervised Extractive Summary
            </div>
            <h1 class="text-xl sm:text-2xl font-extrabold text-slate-900">
              {{ title }}
            </h1>
            <p class="text-xs text-slate-500 mt-1">
              {{ subtitle }}
            </p>
          </div>
          <div class="text-[11px] text-slate-500 bg-slate-50 border border-slate-200 rounded-xl px-3 py-2">
            <div><span class="font-semibold text-slate-700">Summary sentences:</span> {{ stats.summary_sentences }}</div>
            <div><span class="font-semibold text-slate-700">Original sentences:</span> {{ stats.original_sentences }}</div>
            <div><span class="font-semibold text-slate-700">Compression:</span> {{ stats.compression_ratio }}%</div>
          </div>
        </div>

        <div class="border-t border-slate-200 pt-4 mt-2 space-y-5 text-sm leading-relaxed">
          <section>
            <h2 class="text-sm font-bold text-slate-900 mb-1.5 flex items-center gap-2">
              <span class="w-6 h-6 rounded-full bg-teal-100 text-teal-700 flex items-center justify-center text-xs">
                <i class="fa-solid fa-align-left"></i>
              </span>
              Abstract
            </h2>
            <p class="text-sm text-slate-700">
              {{ abstract }}
            </p>
          </section>

          {% if sections %}
          <section>
            <h2 class="text-sm font-bold text-slate-900 mb-2 flex items-center gap-2">
              <span class="w-6 h-6 rounded-full bg-slate-100 text-slate-700 flex items-center justify-center text-xs">
                <i class="fa-solid fa-list-check"></i>
              </span>
              Structured Summary
            </h2>
            <div class="space-y-3">
              {% for sec in sections %}
                <div>
                  <h3 class="text-xs font-semibold text-slate-900 mb-1 flex items-center gap-1.5">
                    <span class="w-1.5 h-1.5 rounded-full bg-teal-500"></span>
                    {{ sec.title }}
                  </h3>
                  <ul class="list-disc pl-5 space-y-1">
                    {% for bullet in sec.bullets %}
                      <li class="text-[13px] text-slate-700">{{ bullet }}</li>
                    {% endfor %}
                  </ul>
                </div>
              {% endfor %}
            </div>
          </section>
          {% endif %}

          {% if extras.key_table %}
          <section>
            <h2 class="text-sm font-bold text-slate-900 mb-2 flex items-center gap-2">
              <span class="w-6 h-6 rounded-full bg-slate-100 text-slate-700 flex items-center justify-center text-xs">
                <i class="fa-solid fa-table"></i>
              </span>
              Summary Coverage by Category
            </h2>
            <div class="overflow-x-auto rounded-xl border border-slate-200 bg-slate-50/60">
              <table class="min-w-full text-xs">
                <thead class="bg-slate-100/80 text-slate-700">
                  <tr>
                    <th class="px-3 py-2 text-left font-semibold border-b border-slate-200">Category</th>
                    <th class="px-3 py-2 text-left font-semibold border-b border-slate-200">Number of Points</th>
                  </tr>
                </thead>
                <tbody>
                  {% for cat, count in category_counts.items() %}
                  <tr class="odd:bg-white even:bg-slate-50/80">
                    <td class="px-3 py-2 border-b border-slate-100 capitalize">{{ cat }}</td>
                    <td class="px-3 py-2 border-b border-slate-100">{{ count }}</td>
                  </tr>
                  {% endfor %}
                </tbody>
              </table>
            </div>
            <p class="text-[11px] text-slate-500 mt-1">
              Categories are derived automatically using keyword-based grouping (goals, principles, delivery, prevention, HR, finance, digital, AYUSH, implementation, other).
            </p>
          </section>
          {% endif %}

          {% if extras.goals_chart %}
          <section>
            <h2 class="text-sm font-bold text-slate-900 mb-2 flex items-center gap-2">
              <span class="w-6 h-6 rounded-full bg-slate-100 text-slate-700 flex items-center justify-center text-xs">
                <i class="fa-solid fa-chart-column"></i>
              </span>
              Category Distribution
            </h2>
            <canvas id="catChart" height="150" class="bg-slate-50 border border-slate-200 rounded-xl"></canvas>
            <p class="text-[11px] text-slate-500 mt-1">
              Visual overview of how the summary is distributed across conceptual categories.
            </p>
          </section>
          {% endif %}

          {% if extras.roadmap and implementation_points %}
          <section>
            <h2 class="text-sm font-bold text-slate-900 mb-2 flex items-center gap-2">
              <span class="w-6 h-6 rounded-full bg-slate-100 text-slate-700 flex items-center justify-center text-xs">
                <i class="fa-solid fa-route"></i>
              </span>
              Implementation Roadmap (Extractive)
            </h2>
            <ul class="list-disc pl-5 space-y-1">
              {% for s in implementation_points %}
                <li class="text-[13px] text-slate-700">{{ s }}</li>
              {% endfor %}
            </ul>
            <p class="text-[11px] text-slate-500 mt-1">
              Automatically selected sentences related to implementation, strategy, governance, or “way forward”.
            </p>
          </section>
          {% endif %}
        </div>

        <div class="mt-6 flex flex-wrap items-center justify-between gap-3 pt-4 border-t border-slate-200">
          <a href="{{ url_for('index') }}"
             class="inline-flex items-center text-xs font-semibold text-slate-600 hover:text-teal-700">
            <i class="fa-solid fa-arrow-left mr-2"></i>
            Summarize another document
          </a>

          <form method="post" action="{{ url_for('download_pdf') }}">
            <input type="hidden" name="pdf_text" value="{{ pdf_text|e }}">
            <button type="submit"
                    class="inline-flex items-center px-4 py-2 rounded-full bg-teal-600 text-white text-xs font-semibold shadow-md shadow-teal-500/40 hover:bg-teal-700 hover:shadow-lg transition">
              <i class="fa-solid fa-file-arrow-down mr-2 text-xs"></i>
              Download Summary (PDF)
            </button>
          </form>
        </div>
      </div>
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
          label: 'Summary points',
          data: data,
        }]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: {
            ticks: { color: '#0f172a', font: { size: 10 } },
            grid: { display: false }
          },
          y: {
            ticks: { color: '#64748b', font: { size: 10 } },
            grid: { color: '#e2e8f0' }
          }
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
    """Filter out TOC-like lines but keep legitimate goal sentences."""
    s_lower = s.lower()
    digits = sum(c.isdigit() for c in s)
    if digits >= 10 and len(s) > 80 and not re.search(
        r"\b(reduce|increase|improve|achieve|eliminate|raise|reach|decrease|enhance)\b",
        s_lower,
    ):
        return True
    if re.search(r"\bcontents\b", s_lower):
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
    sections: List[Tuple[str, str]] = []
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
        return s
    return "Policy Document"

# ---------------------- GOAL & CATEGORY HELPERS ---------------------- #

GOAL_METRIC_WORDS = [
    "life expectancy", "mortality", "imr", "u5mr", "mmr",
    "coverage", "immunization", "immunisation", "incidence",
    "prevalence", "%", " per ", "gdp", "reduction", "rate",
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
    "enhance", "enhancing",
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

# ---------------------- TF-IDF + TextRank + MMR ---------------------- #

def build_tfidf(sentences: List[str]):
    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=1,
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

# ---------------------- EXTRACTIVE SUMMARIZER ---------------------- #

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

    # section scores with boost for goals/principles sections
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

    # force-include numeric goal sentences
    goal_indices = [i for i, s in enumerate(sentences) if is_goal_sentence(s)]
    goal_indices_sorted = sorted(goal_indices, key=lambda i: tr_scores.get(i, 0.0), reverse=True)
    if goal_indices_sorted:
        max_goal = max(1, min(3, int(0.25 * target)))
        forced_goal = goal_indices_sorted[:max_goal]
    else:
        forced_goal = []

    combined = set(selected_idxs) | set(forced_goal)
    if len(combined) > target:
        goal_set = set(forced_goal)
        non_goal = [i for i in combined if i not in goal_set]
        non_goal_sorted = sorted(non_goal, key=lambda i: tr_scores.get(i, 0.0), reverse=True)
        keep_non_goal = non_goal_sorted[: max(0, target - len(goal_set))]
        combined = goal_set | set(keep_non_goal)

    selected_idxs = sorted(combined)
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

# ---------------------- STRUCTURED SUMMARY ---------------------- #

def simplify_for_easy_english(s: str) -> str:
    s = re.sub(r"\([^)]{1,30}\)", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

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

# ---------------------- PDF GENERATION ---------------------- #

def wrap_text_for_pdf(text: str, max_chars: int = 90) -> List[str]:
    lines: List[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            lines.append("")
            continue
        words = para.split()
        current = words[0]
        for w in words[1:]:
            if len(current) + 1 + len(w) <= max_chars:
                current += " " + w
            else:
                lines.append(current)
                current = w
        lines.append(current)
    return lines

def build_pdf_response(pdf_text: str, filename: str = "summary.pdf"):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    x_margin = 50
    y = height - 60

    lines = wrap_text_for_pdf(pdf_text, max_chars=95)
    c.setFont("Helvetica", 11)

    for line in lines:
        if y < 50:
            c.showPage()
            c.setFont("Helvetica", 11)
            y = height - 60
        c.drawString(x_margin, y, line)
        y -= 14

    c.save()
    buffer.seek(0)

    resp = make_response(buffer.getvalue())
    resp.headers["Content-Type"] = "application/pdf"
    resp.headers["Content-Disposition"] = f"attachment; filename={filename}"
    return resp

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
    view = request.form.get("view", "standard").lower()

    try:
        summary_sentences, stats = summarize_extractive(raw_text, length_choice=length_choice)
        structured = build_structured_summary(summary_sentences, tone=tone)
    except Exception as e:
        abort(500, f"Error during summarization: {e}")

    # choose which sections to display based on view
    all_sections = structured["sections"]
    if view == "actions":
        allowed_titles = {
            "Key Goals",
            "Strengthening Healthcare Delivery",
            "Financing & Private Sector Engagement",
            "Implementation & Way Forward",
        }
        display_sections = [sec for sec in all_sections if sec["title"] in allowed_titles]
        extras = {
            "key_table": True,
            "goals_chart": False,
            "roadmap": True,
        }
    elif view == "research":
        display_sections = all_sections
        extras = {
            "key_table": True,
            "goals_chart": True,
            "roadmap": True,
        }
    else:  # standard
        display_sections = all_sections
        extras = {
            "key_table": True,
            "goals_chart": False,
            "roadmap": True,
        }

    doc_title_raw = detect_title(raw_text)
    doc_title = doc_title_raw.strip()
    if doc_title:
        title = f"{doc_title} — Summary"
    else:
        title = "Policy Brief Summary"

    subtitle = "Automatic extractive summary organized as abstract and bullet points (unsupervised: TF-IDF + TextRank + MMR)."

    category_counts = structured["category_counts"]
    if category_counts:
        labels = list(category_counts.keys())
        values = [category_counts[k] for k in labels]
    else:
        labels, values = [], []

    # build plain text for PDF export
    pdf_lines: List[str] = []
    pdf_lines.append(title)
    pdf_lines.append("")
    pdf_lines.append("Abstract")
    pdf_lines.append(structured["abstract"])
    pdf_lines.append("")
    if display_sections:
        pdf_lines.append("Structured Summary")
        for sec in display_sections:
            pdf_lines.append("")
            pdf_lines.append(sec["title"])
            for bullet in sec["bullets"]:
                pdf_lines.append("• " + bullet)
    pdf_text = "\n".join(pdf_lines)

    return render_template_string(
        RESULT_HTML,
        title=title,
        subtitle=subtitle,
        abstract=structured["abstract"],
        sections=display_sections,
        stats=stats,
        extras=extras,
        implementation_points=structured["implementation_points"],
        category_counts=category_counts,
        category_labels=labels,
        category_values=values,
        pdf_text=pdf_text,
    )

@app.route("/download_pdf", methods=["POST"])
def download_pdf():
    pdf_text = request.form.get("pdf_text", "").strip()
    if not pdf_text:
        abort(400, "No summary available to export.")
    return build_pdf_response(pdf_text, filename="policy_summary.pdf")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
