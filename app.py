import io
import os
import re
import uuid
from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np
import networkx as nx
from flask import (
    Flask,
    request,
    render_template_string,
    abort,
    send_from_directory,
    jsonify,
    url_for,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from werkzeug.utils import secure_filename

from PIL import Image
import pytesseract

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit

import google.generativeai as genai

# ---------------------- CONFIG ---------------------- #

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
SUMMARY_FOLDER = os.path.join(BASE_DIR, "summaries")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SUMMARY_FOLDER"] = SUMMARY_FOLDER

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        # Fail gracefully if config fails
        GEMINI_API_KEY = None


# ---------------------- HTML TEMPLATES ---------------------- #

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en" class="scroll-smooth">
<head>
  <meta charset="UTF-8">
  <title>Med | Policy Brief Summarizer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet"
        href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ['Inter', 'sans-serif'],
            mono: ['JetBrains Mono', 'monospace'],
          },
          colors: {
            teal: {
              50: '#f0fdfa', 100: '#ccfbf1', 200: '#99f6e4', 300: '#5eead4',
              400: '#2dd4bf', 500: '#14b8a6', 600: '#0d9488', 700: '#0f766e',
              800: '#115e59', 900: '#134e4a'
            },
          }
        }
      }
    }
  </script>
  <style>
    .card-glass {
      @apply bg-white/90 backdrop-blur-xl border border-slate-200/80 shadow-xl rounded-3xl;
    }
    .chip {
      @apply inline-flex items-center gap-2 px-3 py-1 rounded-full text-xs font-semibold;
    }
    .chip-badge {
      @apply w-2 h-2 rounded-full;
    }
  </style>
</head>
<body class="bg-slate-50 text-slate-800 relative overflow-x-hidden">

  <!-- Background blobs (keep your nice aesthetic) -->
  <div class="blob blob-1 animate-float-slow"></div>
  <div class="blob blob-2 animate-float-medium"></div>

  <!-- NAVBAR -->
  <nav class="fixed w-full z-50 bg-white/80 backdrop-blur-lg border-b border-slate-200/60">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-16 items-center">
        <div class="flex items-center gap-3">
          <div class="w-9 h-9 bg-gradient-to-tr from-teal-600 to-cyan-600 rounded-xl flex items-center justify-center shadow-lg shadow-teal-500/30">
            <i class="fa-solid fa-staff-snake text-white text-lg"></i>
          </div>
          <span class="font-extrabold text-xl tracking-tight text-slate-900">
            Med<span class="text-teal-600">.AI</span>
          </span>
        </div>
        <div class="hidden md:flex items-center gap-8 text-sm font-semibold">
          <span class="text-slate-500">Unsupervised · TF-IDF · TextRank · MMR</span>
          <a href="#workspace" class="px-4 py-2 rounded-full bg-teal-600 text-white hover:bg-teal-700 shadow-md shadow-teal-500/30 text-xs uppercase tracking-wide">
            Open Workspace
          </a>
        </div>
      </div>
    </div>
  </nav>

  <!-- HERO + WORKSPACE -->
  <main class="pt-28 pb-20 px-4">
    <div class="max-w-7xl mx-auto grid lg:grid-cols-12 gap-10 items-start">
      <!-- Left: hero text -->
      <section class="lg:col-span-5 space-y-6 fade-up">
        <div class="chip bg-teal-50 text-teal-700 border border-teal-200">
          <span class="chip-badge bg-teal-500 animate-pulse"></span>
          Primary Healthcare Policy · Extractive NLP
        </div>
        <h1 class="text-4xl lg:text-5xl font-extrabold text-slate-900 leading-tight">
          Summarize policy briefs<br>
          <span class="text-gradient-teal">for Primary Healthcare</span>
        </h1>
        <p class="text-slate-600 text-base lg:text-lg font-medium leading-relaxed">
          Upload PDF, text, or capture a photo of a document. Our unsupervised
          engine (TF-IDF + TextRank + MMR) produces an abstract and structured
          bullet summary optimized for policy and primary healthcare use-cases.
        </p>

        <div class="space-y-3 text-sm text-slate-600">
          <p class="font-semibold text-slate-700 flex items-center gap-2">
            <i class="fa-solid fa-circle-check text-teal-500"></i>
            Designed for policy briefs, guidelines, health missions.
          </p>
          <p class="font-semibold text-slate-700 flex items-center gap-2">
            <i class="fa-solid fa-circle-check text-teal-500"></i>
            Document preview · Gemini Q&A · PDF export of summary.
          </p>
        </div>
      </section>

      <!-- Right: upload workspace -->
      <section id="workspace" class="lg:col-span-7">
        <div class="card-glass p-6 lg:p-8 fade-up delay-100">
          <div class="flex flex-wrap items-center justify-between gap-3 mb-6">
            <div>
              <h2 class="text-lg font-bold text-slate-900 flex items-center gap-2">
                <i class="fa-solid fa-cloud-arrow-up text-teal-600"></i>
                Upload Policy Document
              </h2>
              <p class="text-xs text-slate-500 mt-1">
                Supported: PDF · TXT · Image (camera capture). Max size depends on server limits.
              </p>
            </div>
            <div class="flex flex-wrap gap-2 text-[0.65rem] font-semibold">
              <span class="chip bg-slate-100 text-slate-700">
                <span class="chip-badge bg-emerald-500"></span>
                Extractive · Unsupervised
              </span>
              <span class="chip bg-slate-100 text-slate-700">
                <span class="chip-badge bg-indigo-500"></span>
                Gemini Chat
              </span>
              <span class="chip bg-slate-100 text-slate-700">
                <span class="chip-badge bg-amber-500"></span>
                Summary PDF Export
              </span>
            </div>
          </div>

          <form action="{{ url_for('summarize') }}" method="post" enctype="multipart/form-data" class="space-y-6">
            <!-- File / camera input -->
            <div class="border-2 border-dashed border-slate-200 rounded-2xl p-5 bg-slate-50/60 hover:border-teal-300 hover:bg-teal-50/40 transition relative">
              <div class="flex flex-col md:flex-row items-center gap-4">
                <div class="flex-1 text-sm text-slate-600">
                  <p class="font-semibold text-slate-800 mb-1">
                    Drop your file here, or browse from device
                  </p>
                  <p class="text-xs">
                    Accepts <span class="font-semibold">.pdf, .txt, .jpg, .jpeg, .png</span>.
                    On mobile, you can use the camera directly.
                  </p>
                </div>
                <div class="flex flex-col sm:flex-row gap-3">
                  <label class="inline-flex items-center justify-center px-4 py-2 rounded-full bg-white text-slate-700 border border-slate-200 text-xs font-semibold cursor-pointer hover:border-teal-300 hover:text-teal-700">
                    <i class="fa-solid fa-folder-open mr-2 text-teal-500"></i>
                    Browse Files
                    <input id="file" type="file" name="file"
                           accept=".pdf,.txt,image/*"
                           capture="environment"
                           class="hidden" required>
                  </label>
                  <label class="inline-flex items-center justify-center px-4 py-2 rounded-full bg-teal-600 text-white text-xs font-semibold cursor-pointer shadow-md shadow-teal-500/30 hover:bg-teal-700">
                    <i class="fa-solid fa-camera mr-2"></i>
                    Use Camera
                    <input type="file" accept="image/*" capture="environment"
                           name="file_camera" class="hidden">
                  </label>
                </div>
              </div>
              <p class="mt-3 text-[0.7rem] text-slate-500">
                If both file and camera are provided, the uploaded file takes priority.
              </p>
            </div>

            <!-- Options row -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div class="space-y-2">
                <p class="text-xs font-semibold text-slate-500 uppercase tracking-wide">Summary length</p>
                <div class="space-y-1">
                  <label class="flex items-center gap-2 text-xs">
                    <input type="radio" name="length" value="short">
                    <span>Short (high compression)</span>
                  </label>
                  <label class="flex items-center gap-2 text-xs">
                    <input type="radio" name="length" value="medium" checked>
                    <span>Medium (balanced)</span>
                  </label>
                  <label class="flex items-center gap-2 text-xs">
                    <input type="radio" name="length" value="long">
                    <span>Long (more detail)</span>
                  </label>
                </div>
              </div>

              <div class="space-y-2">
                <p class="text-xs font-semibold text-slate-500 uppercase tracking-wide">Tone</p>
                <div class="space-y-1">
                  <label class="flex items-center gap-2 text-xs">
                    <input type="radio" name="tone" value="academic" checked>
                    <span>Academic</span>
                  </label>
                  <label class="flex items-center gap-2 text-xs">
                    <input type="radio" name="tone" value="easy">
                    <span>Easy English</span>
                  </label>
                </div>
              </div>

              <div class="space-y-2">
                <p class="text-xs font-semibold text-slate-500 uppercase tracking-wide">
                  You will get
                </p>
                <ul class="text-xs text-slate-600 space-y-1">
                  <li class="flex items-center gap-2">
                    <i class="fa-solid fa-file-lines text-teal-500"></i>
                    Abstract + bullet summary
                  </li>
                  <li class="flex items-center gap-2">
                    <i class="fa-solid fa-eye text-amber-500"></i>
                    Original document preview
                  </li>
                  <li class="flex items-center gap-2">
                    <i class="fa-solid fa-message text-violet-500"></i>
                    Gemini Q&A on this document
                  </li>
                  <li class="flex items-center gap-2">
                    <i class="fa-solid fa-file-export text-emerald-500"></i>
                    Downloadable summary PDF
                  </li>
                </ul>
              </div>
            </div>

            <div class="flex justify-end">
              <button type="submit"
                class="inline-flex items-center px-6 py-2.5 rounded-full bg-teal-600 text-white text-sm font-bold shadow-md shadow-teal-500/40 hover:bg-teal-700 hover:shadow-lg hover:-translate-y-0.5 transition">
                <i class="fa-solid fa-wand-magic-sparkles mr-2"></i>
                Generate Summary
              </button>
            </div>
          </form>
        </div>
      </section>
    </div>
  </main>

  <script src="{{ url_for('static', filename='script.js') }}"></script>
</body>
</html>
"""

RESULT_HTML = """
<!DOCTYPE html>
<html lang="en" class="scroll-smooth">
<head>
  <meta charset="UTF-8">
  <title>{{ title }}</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet"
        href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ['Inter', 'sans-serif'],
            mono: ['JetBrains Mono', 'monospace'],
          },
          colors: {
            teal: {
              50: '#f0fdfa', 100: '#ccfbf1', 200: '#99f6e4', 300: '#5eead4',
              400: '#2dd4bf', 500: '#14b8a6', 600: '#0d9488', 700: '#0f766e',
              800: '#115e59', 900: '#134e4a'
            },
          }
        }
      }
    }
  </script>
</head>
<body class="bg-slate-50 text-slate-800">

  <nav class="fixed w-full z-40 bg-white/80 backdrop-blur-lg border-b border-slate-200/60">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-16 items-center">
        <div class="flex items-center gap-3">
          <div class="w-9 h-9 bg-gradient-to-tr from-teal-600 to-cyan-600 rounded-xl flex items-center justify-center shadow-lg shadow-teal-500/30">
            <i class="fa-solid fa-staff-snake text-white text-lg"></i>
          </div>
          <span class="font-extrabold text-xl tracking-tight text-slate-900">
            Med<span class="text-teal-600">.AI</span>
          </span>
        </div>
        <a href="{{ url_for('index') }}"
           class="hidden md:inline-flex items-center px-4 py-2 text-xs font-semibold rounded-full border border-slate-200 hover:border-teal-300 hover:text-teal-700 bg-white">
          <i class="fa-solid fa-arrow-left mr-2"></i>
          New document
        </a>
      </div>
    </div>
  </nav>

  <main class="pt-24 pb-12 px-4">
    <div class="max-w-7xl mx-auto grid lg:grid-cols-12 gap-8">
      <!-- LEFT: summary & analytics -->
      <section class="lg:col-span-7 space-y-6">
        <div class="bg-white rounded-3xl shadow-xl border border-slate-200/80 p-6 md:p-8">
          <div class="flex flex-wrap items-start justify-between gap-3 mb-4">
            <div>
              <h1 class="text-xl md:text-2xl font-extrabold text-slate-900">{{ title }}</h1>
              <p class="text-xs text-slate-500 mt-1">
                Automatic extractive summary (TF-IDF + TextRank + MMR) with policy-aware heuristics.
              </p>
            </div>
            <div class="text-xs text-slate-500 space-y-1">
              <p><span class="font-semibold">Summary sentences:</span> {{ stats.summary_sentences }}</p>
              <p><span class="font-semibold">Original sentences:</span> {{ stats.original_sentences }}</p>
              <p><span class="font-semibold">Compression:</span> {{ stats.compression_ratio }}%</p>
            </div>
          </div>

          <div class="border-t border-slate-200 pt-4 space-y-4">
            <div>
              <h2 class="text-base font-semibold text-slate-900 flex items-center gap-2">
                <span class="w-6 h-6 rounded-full bg-teal-50 text-teal-700 flex items-center justify-center text-xs">
                  A
                </span>
                Abstract
              </h2>
              <p class="mt-2 text-sm text-slate-700 leading-relaxed">{{ abstract }}</p>
            </div>

            {% if sections %}
            <div class="mt-4">
              <h2 class="text-base font-semibold text-slate-900 flex items-center gap-2">
                <span class="w-6 h-6 rounded-full bg-amber-50 text-amber-600 flex items-center justify-center text-xs">
                  Σ
                </span>
                Structured Summary
              </h2>
              <div class="mt-2 space-y-3">
                {% for sec in sections %}
                <div class="border border-slate-200 rounded-2xl p-3 bg-slate-50/80">
                  <h3 class="text-sm font-semibold text-slate-900 mb-1 flex items-center gap-2">
                    <span class="w-1.5 h-1.5 rounded-full bg-teal-500"></span>
                    {{ sec.title }}
                  </h3>
                  <ul class="list-disc list-inside text-xs text-slate-700 space-y-1">
                    {% for bullet in sec.bullets %}
                      <li>{{ bullet }}</li>
                    {% endfor %}
                  </ul>
                </div>
                {% endfor %}
              </div>
            </div>
            {% endif %}
          </div>

          {% if summary_pdf_url %}
          <div class="mt-5 flex flex-wrap gap-3 items-center border-t border-slate-200 pt-3">
            <a href="{{ summary_pdf_url }}" class="inline-flex items-center px-4 py-2 rounded-full bg-teal-600 text-white text-xs font-semibold shadow-md shadow-teal-500/40 hover:bg-teal-700 hover:shadow-lg">
              <i class="fa-solid fa-file-pdf mr-2"></i>
              Download summary as PDF
            </a>
            <p class="text-[0.7rem] text-slate-500">
              The PDF contains this abstract and structured bullet summary, formatted for printing or submission.
            </p>
          </div>
          {% endif %}
        </div>

        <!-- Analytics: key data table + chart -->
        <div class="bg-white rounded-3xl shadow-xl border border-slate-200/80 p-6 md:p-8 space-y-4">
          <h2 class="text-base font-semibold text-slate-900 flex items-center gap-2">
            <i class="fa-solid fa-chart-column text-teal-600"></i>
            Summary Coverage by Category
          </h2>

          {% if category_counts %}
          <div class="overflow-x-auto">
            <table class="min-w-full text-xs text-left text-slate-600 border border-slate-200 rounded-xl overflow-hidden">
              <thead class="bg-slate-50">
                <tr>
                  <th class="px-3 py-2 border-b border-slate-200 font-semibold">Category</th>
                  <th class="px-3 py-2 border-b border-slate-200 font-semibold">Number of Summary Points</th>
                </tr>
              </thead>
              <tbody>
                {% for cat, count in category_counts.items() %}
                <tr class="odd:bg-white even:bg-slate-50/80">
                  <td class="px-3 py-2 capitalize">{{ cat }}</td>
                  <td class="px-3 py-2">{{ count }}</td>
                </tr>
                {% endfor %}
              </tbody>
            </table>
          </div>
          {% else %}
          <p class="text-xs text-slate-500">No category distribution available.</p>
          {% endif %}

          {% if category_labels %}
          <div class="mt-3">
            <canvas id="catChart" height="140"></canvas>
            <p class="mt-2 text-[0.7rem] text-slate-500">
              Bars show how the extracted summary is distributed across conceptual categories
              (goals, service delivery, financing, etc.).
            </p>
          </div>
          {% endif %}

          {% if implementation_points %}
          <div class="mt-4 border-t border-slate-200 pt-3">
            <h3 class="text-sm font-semibold text-slate-900 mb-1 flex items-center gap-2">
              <i class="fa-solid fa-road text-amber-500"></i>
              Implementation / Way Forward (Extractive)
            </h3>
            <ul class="list-disc list-inside text-xs text-slate-700 space-y-1">
              {% for s in implementation_points %}
              <li>{{ s }}</li>
              {% endfor %}
            </ul>
          </div>
          {% endif %}
        </div>
      </section>

      <!-- RIGHT: original document + Gemini chat -->
      <section class="lg:col-span-5 space-y-6">
        <!-- Original document viewer -->
        <div class="bg-white rounded-3xl shadow-xl border border-slate-200/80 p-5 md:p-6">
          <h2 class="text-base font-semibold text-slate-900 flex items-center gap-2 mb-3">
            <i class="fa-solid fa-file-medical text-teal-600"></i>
            Original Document
          </h2>
          {% if orig_type == 'pdf' %}
            <div class="border border-slate-200 rounded-2xl overflow-hidden bg-slate-50 h-[350px]">
              <iframe src="{{ orig_url }}" class="w-full h-full" title="Original PDF"></iframe>
            </div>
          {% elif orig_type == 'text' %}
            <div class="border border-slate-200 rounded-2xl bg-slate-50 p-3 h-[350px] overflow-y-auto">
              <pre class="whitespace-pre-wrap text-xs text-slate-700">{{ orig_text }}</pre>
            </div>
          {% elif orig_type == 'image' %}
            <div class="border border-slate-200 rounded-2xl bg-slate-50 p-2 flex items-center justify-center h-[350px] overflow-hidden">
              <img src="{{ orig_url }}" alt="Uploaded document image" class="max-h-full max-w-full object-contain">
            </div>
          {% else %}
            <p class="text-xs text-slate-500">Original document preview unavailable.</p>
          {% endif %}
          <p class="mt-2 text-[0.7rem] text-slate-500">
            The summary above was generated purely from this document using unsupervised extractive methods.
          </p>
        </div>

        <!-- Gemini chat panel -->
        <div class="bg-white rounded-3xl shadow-xl border border-slate-200/80 p-5 md:p-6">
          <div class="flex items-center justify-between gap-3 mb-3">
            <h2 class="text-base font-semibold text-slate-900 flex items-center gap-2">
              <i class="fa-solid fa-message text-violet-500"></i>
              Ask questions about this policy
            </h2>
            <span class="text-[0.65rem] px-2 py-1 rounded-full bg-violet-50 text-violet-700 border border-violet-200 font-semibold">
              Gemini-connected
            </span>
          </div>

          <div id="chat-panel"
               class="border border-slate-200 rounded-2xl bg-slate-50 p-3 h-64 overflow-y-auto text-xs space-y-2">
            <div class="flex items-start gap-2">
              <div class="w-6 h-6 rounded-full bg-teal-600 text-white flex items-center justify-center text-[0.7rem]">
                <i class="fa-solid fa-robot"></i>
              </div>
              <div class="bg-white rounded-2xl px-3 py-2 shadow-sm max-w-[80%]">
                <p class="text-[0.7rem] text-slate-800">
                  Ask me anything about this policy brief — goals, strategies, financing, or implications for primary healthcare.
                </p>
              </div>
            </div>
          </div>

          <div class="mt-3 flex items-center gap-2">
            <input id="chat-input"
                   class="flex-1 text-xs rounded-full border border-slate-200 px-3 py-2 focus:outline-none focus:ring-1 focus:ring-teal-500 focus:border-teal-500"
                   type="text"
                   placeholder="Example: What are the key quantitative goals by 2025?">
            <button id="chat-send"
                    class="px-4 py-2 rounded-full bg-teal-600 text-white text-xs font-semibold hover:bg-teal-700">
              Ask
            </button>
          </div>
          <p class="mt-1 text-[0.65rem] text-slate-500">
            Powered by Gemini. Responses are generated using only this document as context (no external browsing).
          </p>

          <textarea id="doc-context" class="hidden">{{ doc_context }}</textarea>
        </div>
      </section>
    </div>
  </main>

  {% if category_labels %}
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script>
    const ctx = document.getElementById('catChart').getContext('2d');
    const labels = {{ category_labels|tojson }};
    const dataVals = {{ category_values|tojson }};
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Summary points',
          data: dataVals,
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { ticks: { color: '#64748b' }, grid: { display: false } },
          y: { ticks: { color: '#64748b' }, grid: { color: '#e2e8f0' } }
        }
      }
    });
  </script>
  {% endif %}

  <!-- Gemini chat JS -->
  <script>
    (function(){
      const panel = document.getElementById('chat-panel');
      const input = document.getElementById('chat-input');
      const sendBtn = document.getElementById('chat-send');
      const ctxArea = document.getElementById('doc-context');
      if (!panel || !input || !sendBtn || !ctxArea) return;
      const docText = ctxArea.value || "";

      function addMessage(role, text) {
        const row = document.createElement('div');
        row.className = 'flex items-start gap-2';
        if (role === 'user') row.classList.add('justify-end');
        const bubble = document.createElement('div');
        bubble.className = 'rounded-2xl px-3 py-2 max-w-[80%] text-[0.7rem] leading-relaxed';
        if (role === 'user') {
          bubble.classList.add('bg-teal-600','text-white','shadow');
        } else {
          bubble.classList.add('bg-white','text-slate-800','shadow-sm');
        }
        bubble.textContent = text;
        if (role === 'assistant') {
          const icon = document.createElement('div');
          icon.className = 'w-6 h-6 rounded-full bg-teal-600 text-white flex items-center justify-center text-[0.7rem]';
          icon.innerHTML = '<i class="fa-solid fa-robot"></i>';
          row.appendChild(icon);
        }
        row.appendChild(bubble);
        panel.appendChild(row);
        panel.scrollTop = panel.scrollHeight;
      }

      async function send() {
        const msg = input.value.trim();
        if (!msg) return;
        addMessage('user', msg);
        input.value = '';
        try {
          const res = await fetch('{{ url_for("chat") }}', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg, doc_text: docText })
          });
          const data = await res.json();
          addMessage('assistant', data.reply || 'No response received.');
        } catch (e) {
          addMessage('assistant', 'Error contacting server or Gemini backend.');
        }
      }

      sendBtn.addEventListener('click', send);
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          send();
        }
      });
    })();
  </script>

  <script src="{{ url_for('static', filename='script.js') }}"></script>
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
    """Aggressive TOC filter, but keep possible goal sentences."""
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
        parts = re.split(r"(?<=[\.\?\!])\s+(?=[A-Z0-9“'\"-])", chunk)
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


def extract_text_from_pdf_bytes(raw: bytes) -> str:
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
    "life expectancy",
    "mortality",
    "imr",
    "u5mr",
    "mmr",
    "coverage",
    "immunization",
    "immunisation",
    "incidence",
    "prevalence",
    "%",
    " per ",
    "gdp",
    "reduction",
    "rate",
]

GOAL_VERBS = [
    "reduce",
    "reducing",
    "reduction",
    "increase",
    "increasing",
    "improve",
    "improving",
    "achieve",
    "achieving",
    "eliminate",
    "eliminating",
    "raise",
    "raising",
    "reach",
    "reaching",
    "decrease",
    "decreasing",
    "enhance",
    "enhancing",
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

    if any(
        w in s_lower
        for w in [
            "principle",
            "values",
            "equity",
            "universal access",
            "universal health",
            "right to health",
            "accountability",
            "integrity",
            "patient-centred",
            "patient-centered",
        ]
    ):
        return "policy principles"

    if any(
        w in s_lower
        for w in [
            "primary care",
            "secondary care",
            "tertiary care",
            "health and wellness centre",
            "health & wellness centre",
            "health & wellness center",
            "hospital",
            "service delivery",
            "referral",
            "emergency services",
            "free drugs",
            "free diagnostics",
        ]
    ):
        return "service delivery"

    if any(
        w in s_lower
        for w in [
            "prevention",
            "preventive",
            "promotive",
            "promotion",
            "sanitation",
            "nutrition",
            "tobacco",
            "alcohol",
            "air pollution",
            "road safety",
            "lifestyle",
            "behaviour change",
            "behavior change",
            "swachh",
            "clean water",
        ]
    ):
        return "prevention & promotion"

    if any(
        w in s_lower
        for w in [
            "human resources for health",
            "hrh",
            "health workforce",
            "doctors",
            "nurses",
            "mid-level",
            "medical college",
            "nursing college",
            "public health management cadre",
            "training",
            "capacity building",
        ]
    ):
        return "human resources"

    if any(
        w in s_lower
        for w in [
            "financing",
            "financial protection",
            "insurance",
            "strategic purchasing",
            "public spending",
            "gdp",
            "expenditure",
            "catastrophic",
            "private sector",
            "ppp",
            "reimbursement",
            "fees",
            "empanelment",
        ]
    ):
        return "financing & private sector"

    if any(
        w in s_lower
        for w in [
            "digital health",
            "health information",
            "ehr",
            "electronic health record",
            "telemedicine",
            "information system",
            "surveillance",
            "ndha",
            "health data",
        ]
    ):
        return "digital health"

    if any(w in s_lower for w in ["ayush", "ayurveda", "yoga", "unani", "siddha", "homeopathy"]):
        return "ayush integration"

    if any(
        w in s_lower
        for w in [
            "implementation",
            "way forward",
            "roadmap",
            "action plan",
            "strategy",
            "governance",
            "monitoring",
            "evaluation",
            "framework",
        ]
    ):
        return "implementation"

    return "other"


# ---------------------- TF-IDF + TEXTRANK + MMR ---------------------- #

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

    # force-in goal sentences
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


# ---------------------- SUMMARY PDF GENERATION ---------------------- #

def save_summary_pdf(title: str, abstract: str, sections: List[Dict], out_path: str):
    c = canvas.Canvas(out_path, pagesize=A4)
    width, height = A4
    margin_x = 40
    margin_y = 40
    max_width = width - 2 * margin_x

    def draw_paragraph(text, y, font="Helvetica", size=10, leading=13):
        c.setFont(font, size)
        lines = simpleSplit(text, font, size, max_width)
        for line in lines:
            if y < margin_y:
                c.showPage()
                y = height - margin_y
                c.setFont(font, size)
            c.drawString(margin_x, y, line)
            y -= leading
        return y

    y = height - margin_y
    c.setFont("Helvetica-Bold", 14)
    for line in simpleSplit(title, "Helvetica-Bold", 14, max_width):
        c.drawString(margin_x, y, line)
        y -= 18

    y -= 10
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin_x, y, "Abstract")
    y -= 14
    y = draw_paragraph(abstract, y)

    y -= 8
    for sec in sections:
        if y < margin_y + 40:
            c.showPage()
            y = height - margin_y
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin_x, y, sec["title"])
        y -= 14
        for bullet in sec["bullets"]:
            bullet_text = "• " + bullet
            y = draw_paragraph(bullet_text, y)
            y -= 2
        y -= 6

    c.showPage()
    c.save()


# ---------------------- GEMINI CHAT ---------------------- #

def gemini_answer(user_message: str, doc_text: str) -> str:
    if not GEMINI_API_KEY:
        return "Gemini API key is not configured on the server."

    try:
        model = genai.GenerativeModel("gemini-3-pro-preview")
        prompt = (
            "You are an AI assistant helping a student understand a healthcare policy document.\n"
            "Answer concisely and only using information from the document.\n\n"
            "DOCUMENT:\n"
            f"{doc_text[:120000]}\n\n"
            f"USER QUESTION: {user_message}\n\n"
            "ANSWER:"
        )
        resp = model.generate_content(prompt)
        return (resp.text or "").strip() or "No response generated."
    except Exception as e:
        return f"Error while contacting Gemini API: {e}"


# ---------------------- FLASK ROUTES ---------------------- #

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/summaries/<path:filename>")
def summary_file(filename):
    return send_from_directory(app.config["SUMMARY_FOLDER"], filename, as_attachment=True)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    doc_text = data.get("doc_text") or ""
    if not message:
        return jsonify({"reply": "Please type a question first."})
    reply = gemini_answer(message, doc_text)
    return jsonify({"reply": reply})


@app.route("/summarize", methods=["POST"])
def summarize():
    # Decide which file field to use: main "file" or fallback "file_camera"
    f = request.files.get("file")
    if not f or (f and f.filename == ""):
        f = request.files.get("file_camera")

    if not f or f.filename == "":
        abort(400, "No file uploaded. Please upload a PDF, text file, or image.")

    filename = f.filename or "document"
    safe_name = secure_filename(filename)
    raw_bytes = f.read()
    if not raw_bytes:
        abort(400, "Uploaded document appears to be empty or unreadable.")

    # Save original file for preview
    uid = uuid.uuid4().hex
    stored_name = f"{uid}_{safe_name}"
    stored_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
    with open(stored_path, "wb") as out:
        out.write(raw_bytes)

    # Detect type and extract text
    lower_name = filename.lower()
    orig_type = "unknown"
    orig_text = ""
    try:
        if lower_name.endswith(".pdf"):
            orig_type = "pdf"
            raw_text = extract_text_from_pdf_bytes(raw_bytes)
        elif lower_name.endswith(".txt"):
            orig_type = "text"
            raw_text = raw_bytes.decode("utf-8", errors="ignore")
            orig_text = raw_text
        else:
            # treat as image (camera or uploaded)
            orig_type = "image"
            img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            raw_text = pytesseract.image_to_string(img)
    except Exception as e:
        abort(500, f"Error while extracting text from the uploaded file: {e}")

    if not raw_text or len(raw_text.strip()) < 50:
        abort(400, "Could not extract enough text from the uploaded document.")

    length_choice = request.form.get("length", "medium").lower()
    tone = request.form.get("tone", "academic").lower()

    try:
        summary_sentences, stats = summarize_extractive(raw_text, length_choice=length_choice)
        structured = build_structured_summary(summary_sentences, tone=tone)
    except Exception as e:
        abort(500, f"Error during summarization: {e}")

    doc_title_raw = detect_title(raw_text)
    doc_title = doc_title_raw.strip()
    if doc_title:
        page_title = f"{doc_title} — Summary"
    else:
        page_title = "Policy Brief Summary"

    doc_context = raw_text[:8000]

    # generate summary PDF
    summary_pdf_url = None
    try:
        summary_filename = f"{uid}_summary.pdf"
        summary_path = os.path.join(app.config["SUMMARY_FOLDER"], summary_filename)
        save_summary_pdf(
            page_title,
            structured["abstract"],
            structured["sections"],
            summary_path,
        )
        summary_pdf_url = url_for("summary_file", filename=summary_filename)
    except Exception:
        summary_pdf_url = None

    subtitle = "Automatic extractive summary organized as abstract and bullet points (unsupervised: TF-IDF + TextRank + MMR)."

    extras = {
        "key_table": True,
        "goals_chart": True,
        "roadmap": True,
    }

    category_counts = structured["category_counts"]
    if category_counts:
        labels = list(category_counts.keys())
        values = [category_counts[k] for k in labels]
    else:
        labels, values = [], []

    # For text preview if not already
    if orig_type == "text" and not orig_text:
        orig_text = raw_text[:20000]

    return render_template_string(
        RESULT_HTML,
        title=page_title,
        subtitle=subtitle,
        abstract=structured["abstract"],
        sections=structured["sections"],
        stats=stats,
        extras=extras,
        implementation_points=structured["implementation_points"],
        category_counts=category_counts,
        category_labels=labels,
        category_values=values,
        orig_type=orig_type,
        orig_url=url_for("uploaded_file", filename=stored_name),
        orig_text=orig_text,
        doc_context=doc_context,
        summary_pdf_url=summary_pdf_url,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
