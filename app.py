import io
import os
import re
import uuid
import textwrap
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
    send_file,
    jsonify,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import google.generativeai as genai

# ---------------------- FLASK APP CONFIG ---------------------- #

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory document store: doc_id -> {text, summary_sentences, title, kind, file_name}
DOC_STORE: Dict[str, Dict] = {}

# ---------------------- GEMINI CONFIG ---------------------- #

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("Warning: GEMINI_API_KEY is not set. Gemini features (camera OCR, chat, actions, Q&A) will not work.")


def get_gemini_model():
    """
    Helper to create a Gemini model instance.
    Uses gemini-1.5-flash for speed + image support.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("Gemini API key not configured. Set GEMINI_API_KEY environment variable.")
    return genai.GenerativeModel("gemini-1.5-flash")


def extract_text_with_gemini_from_image(image_bytes: bytes) -> str:
    """
    Use Gemini to extract text from a document image.
    Assumes PNG (canvas capture defaults to image/png).
    """
    try:
        model = get_gemini_model()
        prompt = (
            "You are an OCR engine. Extract all textual content from this policy/health document image. "
            "Return only the cleaned text, preserving headings and numbering where possible."
        )
        img = {
            "mime_type": "image/png",
            "data": image_bytes,
        }
        resp = model.generate_content([prompt, img])
        return (resp.text or "").strip()
    except Exception as e:
        print("Error in Gemini image OCR:", e)
        return ""


def gemini_answer_for_doc(doc_text: str, question: str) -> str:
    """
    Use Gemini to answer a user question based only on doc_text.
    """
    try:
        model = get_gemini_model()
        prompt = (
            "You are an assistant restricted to the provided policy document. "
            "Answer the question using ONLY this document. If the answer is not clearly present, "
            "say you cannot find it in the document.\n\n"
            "DOCUMENT:\n"
            f"{doc_text[:120000]}\n\n"  # truncate very long docs for safety
            "QUESTION:\n"
            f"{question}\n\n"
            "Answer clearly and concisely."
        )
        resp = model.generate_content(prompt)
        return (resp.text or "").strip()
    except Exception as e:
        print("Error in Gemini chat:", e)
        return "Sorry, Gemini is not available right now or there was an error processing your request."


def gemini_action_items_for_doc(doc_text: str) -> List[str]:
    """
    Use Gemini to create action items from the policy document.
    """
    try:
        model = get_gemini_model()
        prompt = (
            "You are an expert in health policy implementation. Based ONLY on the document below, "
            "produce 5–10 concrete, actionable items that a health department or implementation team "
            "should focus on. Each action item should be a clear, single bullet.\n\n"
            "DOCUMENT:\n"
            f"{doc_text[:120000]}\n\n"
            "Return the action items as a simple bullet list (no extra commentary)."
        )
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        bullets = []
        for line in text.splitlines():
            line = line.strip("-•* \t")
            if line:
                bullets.append(line)
        return bullets
    except Exception as e:
        print("Error in Gemini action items:", e)
        return ["Gemini is not available right now or there was an error generating action items."]


def gemini_qa_for_doc(doc_text: str) -> List[Dict[str, str]]:
    """
    Use Gemini to generate Q&A pairs from the policy document.
    """
    try:
        model = get_gemini_model()
        prompt = (
            "You are a tutor designing questions to test understanding of the following policy document. "
            "Based ONLY on this document, generate 5–8 important question-and-answer pairs. "
            "Questions should be short and focused; answers should be accurate and concise.\n\n"
            "DOCUMENT:\n"
            f"{doc_text[:120000]}\n\n"
            "Return the Q&A as numbered items in the format:\n"
            "1. Q: ...\n"
            "   A: ...\n"
            "2. Q: ...\n"
            "   A: ...\n"
        )
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip()
        qas: List[Dict[str, str]] = []
        current_q = None
        current_a = None
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            q_match = re.match(r"(\d+\.\s*)?Q[:：]\s*(.+)", line, flags=re.IGNORECASE)
            a_match = re.match(r"A[:：]\s*(.+)", line, flags=re.IGNORECASE)
            if q_match:
                if current_q and current_a:
                    qas.append({"question": current_q, "answer": current_a})
                current_q = q_match.group(2).strip()
                current_a = None
            elif a_match:
                current_a = a_match.group(1).strip()
            else:
                if current_a is not None:
                    current_a += " " + line
                elif current_q is not None:
                    current_q += " " + line
        if current_q and current_a:
            qas.append({"question": current_q, "answer": current_a})
        return qas or [{"question": "No Q&A generated", "answer": "Gemini could not generate Q&A for this document."}]
    except Exception as e:
        print("Error in Gemini Q&A:", e)
        return [{"question": "Error", "answer": "Gemini is not available right now or there was an error generating Q&A."}]


# ---------------------- HTML TEMPLATES ---------------------- #

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en" class="scroll-smooth">
<head>
  <meta charset="UTF-8">
  <title>Med.AI Workspace | Policy Summarizer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap">
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: {
            sans: ['Inter', 'sans-serif'],
          },
          colors: {
            teal: { 50: '#f0fdfa', 100: '#ccfbf1', 200: '#99f6e4', 300: '#5eead4', 400: '#2dd4bf', 500: '#14b8a6', 600: '#0d9488', 700: '#0f766e', 800: '#115e59', 900: '#134e4a' },
          }
        }
      }
    }
  </script>
  <style>
    body { font-family: 'Inter', sans-serif; }
    .text-gradient-teal {
      background: linear-gradient(135deg, #0d9488, #14b8a6, #2dd4bf);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .dropzone-dashed {
      border-radius: 1rem;
      border-width: 2px;
      border-style: dashed;
    }
  </style>
</head>
<body class="bg-slate-50 text-slate-800">

  <!-- Nav (matching Med.AI style) -->
  <nav class="fixed w-full z-50 bg-white/80 backdrop-blur-lg border-b border-slate-200/60">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-16 items-center">
        <div class="flex-shrink-0 flex items-center group cursor-pointer">
          <div class="w-9 h-9 bg-gradient-to-tr from-teal-600 to-cyan-600 rounded-xl flex items-center justify-center shadow-lg shadow-teal-500/30 mr-3 group-hover:scale-105 transition transform">
            <i class="fa-solid fa-staff-snake text-white text-lg"></i>
          </div>
          <span class="font-extrabold text-xl tracking-tight text-slate-900">Med<span class="text-teal-600">.AI</span></span>
        </div>
        <div class="hidden md:flex space-x-8 items-center text-sm font-semibold uppercase tracking-wider">
          <a href="index.html" class="text-slate-600 hover:text-teal-600 transition">Home</a>
          <span class="text-teal-700 border-b-2 border-teal-600 pb-1">Workspace</span>
          <a href="about.html" class="text-slate-600 hover:text-teal-600 transition">About</a>
        </div>
      </div>
    </div>
  </nav>

  <!-- Workspace -->
  <main class="pt-28 pb-16">
    <div class="max-w-6xl mx-auto px-4">
      <div class="grid lg:grid-cols-2 gap-8 items-start">
        <!-- Left: Headline + What you get -->
        <section>
          <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-teal-50 border border-teal-200 text-teal-800 text-xs font-bold uppercase tracking-wide mb-4">
            <span class="w-2 h-2 rounded-full bg-teal-500 animate-pulse"></span>
            Primary Healthcare • Policy Briefs
          </div>
          <h1 class="text-3xl md:text-4xl font-extrabold text-slate-900 mb-3 leading-tight">
            Summarize health policy briefs<br>
            <span class="text-gradient-teal">and interact with them using AI</span>
          </h1>
          <p class="text-slate-600 text-sm md:text-base mb-6 max-w-xl">
            Upload a PDF or text file, or capture a document with your camera. The engine
            uses an unsupervised pipeline (TF-IDF + TextRank + MMR) to generate an abstract
            plus structured bullet summary, then Gemini gives you action items and Q&amp;A.
          </p>

          <div class="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 space-y-3">
            <h2 class="text-xs font-bold uppercase tracking-wide text-slate-500 flex items-center gap-2">
              <span class="w-1.5 h-1.5 rounded-full bg-teal-500"></span>
              What you’ll get
            </h2>
            <ul class="space-y-2 text-sm text-slate-600">
              <li class="flex gap-2">
                <i class="fa-solid fa-circle-check text-teal-500 mt-1"></i>
                <span><strong>Abstract + bullet summary</strong> optimized for primary healthcare policies.</span>
              </li>
              <li class="flex gap-2">
                <i class="fa-solid fa-bullseye text-teal-500 mt-1"></i>
                <span>Automatic grouping into <strong>Key Goals, Service Delivery, Financing, etc.</strong></span>
              </li>
              <li class="flex gap-2">
                <i class="fa-solid fa-clipboard-list text-teal-500 mt-1"></i>
                <span><strong>Action items</strong> and <strong>Q&amp;A mode</strong> powered by Gemini, using only your document.</span>
              </li>
              <li class="flex gap-2">
                <i class="fa-solid fa-file-arrow-down text-teal-500 mt-1"></i>
                <span>Downloadable <strong>summary PDF</strong> for reports and viva.</span>
              </li>
            </ul>
          </div>
        </section>

        <!-- Right: Upload / Camera / Options -->
        <section>
          <form id="upload-form" action="{{ url_for('summarize') }}" method="post" enctype="multipart/form-data"
                class="bg-white rounded-2xl border border-slate-200 shadow-lg shadow-slate-200/70 p-6 space-y-6">
            <h2 class="text-sm font-bold uppercase tracking-wide text-slate-500 mb-2 flex items-center gap-2">
              <i class="fa-solid fa-cloud-arrow-up text-teal-500"></i>
              Upload or Capture Document
            </h2>

            <!-- File upload -->
            <div class="space-y-2">
              <label class="text-xs font-semibold text-slate-500">PDF or Text File</label>
              <div id="dropzone"
                   class="dropzone-dashed border-slate-300 hover:border-teal-400 bg-slate-50 hover:bg-teal-50/40 transition cursor-pointer px-4 py-6 flex flex-col items-center justify-center text-center">
                <i class="fa-solid fa-file-pdf text-teal-500 text-2xl mb-2"></i>
                <p class="text-sm font-medium text-slate-700">Click to browse or drag &amp; drop your file</p>
                <p class="text-[11px] text-slate-500 mt-1">Supported: .pdf, .txt</p>
                <input id="file-input" type="file" name="file" accept=".pdf,.txt"
                       class="hidden">
              </div>
              <p id="file-name" class="text-[11px] text-slate-500 mt-1"></p>
            </div>

            <!-- Camera capture -->
            <div class="space-y-2">
              <label class="text-xs font-semibold text-slate-500 flex items-center gap-2">
                <i class="fa-solid fa-camera text-slate-500"></i>
                Capture from Camera (Gemini OCR)
              </label>
              <div class="bg-slate-50 rounded-xl border border-slate-200 p-3 space-y-3">
                <div class="flex flex-wrap items-center gap-2">
                  <button type="button" id="start-camera"
                          class="px-3 py-1.5 text-xs rounded-full border border-slate-300 text-slate-700 hover:border-teal-500 hover:text-teal-600 hover:bg-teal-50 transition">
                    <i class="fa-solid fa-video mr-1"></i> Start Camera
                  </button>
                  <button type="button" id="capture-photo" disabled
                          class="px-3 py-1.5 text-xs rounded-full border border-slate-300 text-slate-400 cursor-not-allowed transition">
                    <i class="fa-solid fa-circle-dot mr-1"></i> Capture
                  </button>
                  <span id="camera-status" class="text-[11px] text-slate-500">Camera off</span>
                </div>
                <div class="flex gap-3 items-center">
                  <video id="video" class="w-40 h-28 bg-black/5 rounded-lg object-cover border border-slate-200 hidden" autoplay></video>
                  <canvas id="canvas" class="hidden"></canvas>
                  <img id="photo-preview" class="w-40 h-28 rounded-lg border border-slate-200 object-cover hidden" alt="Captured preview">
                </div>
                <input type="hidden" name="image_data" id="image-data">
                <p class="text-[11px] text-slate-500">If both file and camera image are provided, the file will be used as the main source.</p>
              </div>
            </div>

            <!-- Summary options -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div class="space-y-2">
                <label class="text-xs font-semibold text-slate-500">Summary Length</label>
                <div class="flex flex-col gap-1.5 text-xs text-slate-600">
                  <label class="flex items-center gap-2">
                    <input type="radio" name="length" value="short">
                    <span>Short (high compression)</span>
                  </label>
                  <label class="flex items-center gap-2">
                    <input type="radio" name="length" value="medium" checked>
                    <span>Medium (balanced)</span>
                  </label>
                  <label class="flex items-center gap-2">
                    <input type="radio" name="length" value="long">
                    <span>Long (more detail)</span>
                  </label>
                </div>
              </div>
              <div class="space-y-2">
                <label class="text-xs font-semibold text-slate-500">Tone</label>
                <div class="flex flex-col gap-1.5 text-xs text-slate-600">
                  <label class="flex items-center gap-2">
                    <input type="radio" name="tone" value="academic" checked>
                    <span>Academic</span>
                  </label>
                  <label class="flex items-center gap-2">
                    <input type="radio" name="tone" value="easy">
                    <span>Easy English</span>
                  </label>
                </div>
              </div>
            </div>

            <button type="submit"
                    class="w-full mt-2 inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-full bg-teal-600 hover:bg-teal-700 text-white text-sm font-semibold shadow-md hover:shadow-lg shadow-teal-500/30 transition">
              <i class="fa-solid fa-wand-magic-sparkles"></i>
              Generate Summary
            </button>
          </form>
        </section>
      </div>
    </div>
  </main>

  <script>
    // File input + dropzone
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const fileNameLabel = document.getElementById('file-name');

    dropzone.addEventListener('click', () => fileInput.click());
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('border-teal-500', 'bg-teal-50/40');
    });
    dropzone.addEventListener('dragleave', () => {
      dropzone.classList.remove('border-teal-500', 'bg-teal-50/40');
    });
    dropzone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropzone.classList.remove('border-teal-500', 'bg-teal-50/40');
      if (e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        fileNameLabel.textContent = e.dataTransfer.files[0].name;
      }
    });
    fileInput.addEventListener('change', () => {
      if (fileInput.files.length > 0) {
        fileNameLabel.textContent = fileInput.files[0].name;
      } else {
        fileNameLabel.textContent = "";
      }
    });

    // Camera capture
    const startBtn = document.getElementById('start-camera');
    const captureBtn = document.getElementById('capture-photo');
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const photoPreview = document.getElementById('photo-preview');
    const imageDataInput = document.getElementById('image-data');
    const cameraStatus = document.getElementById('camera-status');

    let stream = null;

    startBtn.addEventListener('click', async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.classList.remove('hidden');
        photoPreview.classList.add('hidden');
        captureBtn.disabled = false;
        captureBtn.classList.remove('text-slate-400', 'cursor-not-allowed');
        captureBtn.classList.add('text-slate-700');
        cameraStatus.textContent = 'Camera on';
      } catch (err) {
        console.error(err);
        cameraStatus.textContent = 'Camera error / blocked by browser';
      }
    });

    captureBtn.addEventListener('click', () => {
      if (!stream) return;
      const track = stream.getVideoTracks()[0];
      const settings = track.getSettings();
      const width = settings.width || 640;
      const height = settings.height || 480;
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, width, height);
      const dataURL = canvas.toDataURL('image/png');
      imageDataInput.value = dataURL;
      photoPreview.src = dataURL;
      photoPreview.classList.remove('hidden');
      video.classList.add('hidden');
      cameraStatus.textContent = 'Photo captured';
      // stop camera
      stream.getTracks().forEach(t => t.stop());
      stream = null;
      captureBtn.disabled = true;
      captureBtn.classList.add('text-slate-400', 'cursor-not-allowed');
      captureBtn.classList.remove('text-slate-700');
    });
  </script>
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
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap">
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: { sans: ['Inter', 'sans-serif'] },
          colors: {
            teal: { 50: '#f0fdfa', 100: '#ccfbf1', 200: '#99f6e4', 300: '#5eead4', 400: '#2dd4bf', 500: '#14b8a6', 600: '#0d9488', 700: '#0f766e', 800: '#115e59', 900: '#134e4a' },
          }
        }
      }
    }
  </script>
  <style>
    body { font-family: 'Inter', sans-serif; }
    .text-gradient-teal {
      background: linear-gradient(135deg, #0d9488, #14b8a6, #2dd4bf);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
  </style>
</head>
<body class="bg-slate-50 text-slate-800">
  <nav class="fixed w-full z-50 bg-white/80 backdrop-blur-lg border-b border-slate-200/60">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex justify-between h-16 items-center">
        <div class="flex-shrink-0 flex items-center group cursor-pointer">
          <div class="w-9 h-9 bg-gradient-to-tr from-teal-600 to-cyan-600 rounded-xl flex items-center justify-center shadow-lg shadow-teal-500/30 mr-3 group-hover:scale-105 transition transform">
            <i class="fa-solid fa-staff-snake text-white text-lg"></i>
          </div>
          <span class="font-extrabold text-xl tracking-tight text-slate-900">Med<span class="text-teal-600">.AI</span></span>
        </div>
        <div class="hidden md:flex space-x-8 items-center text-sm font-semibold uppercase tracking-wider">
          <a href="{{ url_for('index') }}" class="text-slate-600 hover:text-teal-600 transition">Back to Workspace</a>
        </div>
      </div>
    </div>
  </nav>

  <main class="pt-24 pb-12">
    <div class="max-w-6xl mx-auto px-4 space-y-6">
      <div class="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 class="text-2xl md:text-3xl font-extrabold text-slate-900">
            {{ title }}
          </h1>
          <p class="text-xs md:text-sm text-slate-500 mt-1">
            Automatic extractive summary (TF-IDF + TextRank + MMR) with Gemini-powered insights.
          </p>
          <div class="mt-2 flex flex-wrap gap-3 text-[11px] md:text-xs text-slate-500">
            <span class="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-teal-50 border border-teal-200 text-teal-700 font-semibold">
              <i class="fa-solid fa-layer-group text-[10px]"></i>
              <span>{{ stats.summary_sentences }} sentences</span>
            </span>
            <span class="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-slate-100 border border-slate-200">
              <i class="fa-solid fa-file-lines text-[10px]"></i>
              <span>{{ stats.original_sentences }} original sentences</span>
            </span>
            <span class="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-slate-100 border border-slate-200">
              <i class="fa-solid fa-scissors text-[10px]"></i>
              <span>Compression: {{ stats.compression_ratio }}%</span>
            </span>
          </div>
        </div>
        <div class="flex flex-wrap gap-3">
          <a href="{{ url_for('download_summary_pdf', doc_id=doc_id) }}"
             class="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white border border-slate-200 text-xs md:text-sm font-semibold text-slate-700 hover:border-teal-500 hover:text-teal-600 hover:bg-teal-50 transition shadow-sm">
            <i class="fa-solid fa-file-arrow-down"></i>
            Download Summary (PDF)
          </a>
        </div>
      </div>

      <div class="grid lg:grid-cols-2 gap-6 items-start">
        <!-- Summary -->
        <section class="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 space-y-4">
          <h2 class="text-sm font-bold uppercase tracking-wide text-slate-500 flex items-center gap-2">
            <i class="fa-solid fa-align-left text-teal-500"></i>
            Abstract
          </h2>
          <p class="text-sm text-slate-700 leading-relaxed">
            {{ abstract }}
          </p>

          {% if sections %}
          <div class="pt-4 border-t border-slate-200 space-y-4">
            <h2 class="text-sm font-bold uppercase tracking-wide text-slate-500 flex items-center gap-2">
              <i class="fa-solid fa-list text-teal-500"></i>
              Structured Summary
            </h2>
            <div class="space-y-3">
              {% for sec in sections %}
                <div class="space-y-1.5">
                  <h3 class="text-xs font-semibold uppercase tracking-wide text-slate-500">{{ sec.title }}</h3>
                  <ul class="list-disc pl-4 space-y-1.5">
                    {% for bullet in sec.bullets %}
                      <li class="text-sm text-slate-700 leading-relaxed">{{ bullet }}</li>
                    {% endfor %}
                  </ul>
                </div>
              {% endfor %}
            </div>
          </div>
          {% endif %}

          {% if category_counts %}
          <div class="pt-4 border-t border-slate-200 space-y-2">
            <h2 class="text-sm font-bold uppercase tracking-wide text-slate-500 flex items-center gap-2">
              <i class="fa-solid fa-table-list text-teal-500"></i>
              Category Snapshot
            </h2>
            <div class="overflow-x-auto">
              <table class="min-w-full text-xs border border-slate-200 rounded-lg overflow-hidden">
                <thead class="bg-slate-50">
                  <tr>
                    <th class="px-3 py-2 text-left font-semibold text-slate-600 border-b border-slate-200">Category</th>
                    <th class="px-3 py-2 text-left font-semibold text-slate-600 border-b border-slate-200">Summary Points</th>
                  </tr>
                </thead>
                <tbody>
                  {% for cat, count in category_counts.items() %}
                  <tr class="border-b border-slate-100">
                    <td class="px-3 py-2 text-slate-700">{{ cat }}</td>
                    <td class="px-3 py-2 text-slate-700">{{ count }}</td>
                  </tr>
                  {% endfor %}
                </tbody>
              </table>
            </div>
          </div>
          {% endif %}
        </section>

        <!-- Original Document + AI Toolkit -->
        <section class="space-y-5">
          <!-- Original doc -->
          <div class="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 space-y-3">
            <h2 class="text-sm font-bold uppercase tracking-wide text-slate-500 flex items-center gap-2">
              <i class="fa-solid fa-file text-teal-500"></i>
              Original Document
            </h2>
            {% if doc_kind == 'pdf' and doc_file_url %}
              <div class="border border-slate-200 rounded-lg overflow-hidden h-72 bg-slate-50">
                <iframe src="{{ doc_file_url }}" class="w-full h-full border-0"></iframe>
              </div>
              <p class="text-[11px] text-slate-500 mt-1">Embedded viewer uses your browser’s PDF support.</p>
            {% elif doc_kind == 'text' %}
              <div class="border border-slate-200 rounded-lg bg-slate-50 p-3 h-72 overflow-y-auto text-[12px] whitespace-pre-wrap text-slate-700">
                {{ doc_text }}
              </div>
            {% elif doc_kind == 'image' and doc_file_url %}
              <div class="border border-slate-200 rounded-lg bg-slate-50 p-2 flex justify-center items-center h-72">
                <img src="{{ doc_file_url }}" alt="Captured document" class="max-h-full max-w-full rounded-lg shadow-sm">
              </div>
            {% else %}
              <p class="text-xs text-slate-500">Original document preview not available.</p>
            {% endif %}
          </div>

          <!-- AI Toolkit -->
          <div class="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 space-y-4">
            <h2 class="text-sm font-bold uppercase tracking-wide text-slate-500 flex items-center gap-2">
              <i class="fa-solid fa-robot text-teal-500"></i>
              Gemini Toolkit (on this document)
            </h2>

            <!-- Chat -->
            <div class="space-y-2 border-b border-slate-200 pb-3">
              <h3 class="text-xs font-semibold uppercase tracking-wide text-slate-500 flex items-center gap-2">
                <i class="fa-solid fa-comments text-teal-500"></i>
                Chat about this policy
              </h3>
              <div id="chat-box" class="border border-slate-200 rounded-lg bg-slate-50 p-2 h-40 overflow-y-auto text-[12px] space-y-1.5">
                <div class="text-slate-500 italic">Ask anything about this policy. Responses are grounded only in the uploaded document.</div>
              </div>
              <form id="chat-form" class="flex mt-1 gap-2">
                <input id="chat-input" type="text" placeholder="Ask a question..." autocomplete="off"
                       class="flex-1 text-xs px-2 py-1.5 rounded-full border border-slate-300 focus:outline-none focus:ring-1 focus:ring-teal-500 focus:border-teal-500">
                <button type="submit"
                        class="px-3 py-1.5 rounded-full bg-teal-600 hover:bg-teal-700 text-white text-xs font-semibold">
                  Send
                </button>
              </form>
            </div>

            <!-- Action items -->
            <div class="space-y-2 border-b border-slate-200 pb-3">
              <div class="flex items-center justify-between">
                <h3 class="text-xs font-semibold uppercase tracking-wide text-slate-500 flex items-center gap-2">
                  <i class="fa-solid fa-list-check text-teal-500"></i>
                  Action Items
                </h3>
                <button id="generate-actions"
                        class="text-[11px] px-2 py-1 rounded-full border border-slate-300 text-slate-700 hover:border-teal-500 hover:text-teal-600 hover:bg-teal-50 transition">
                  Generate
                </button>
              </div>
              <ul id="actions-list" class="list-disc pl-4 space-y-1 text-[12px] text-slate-700">
                <li class="text-slate-500 italic">Click “Generate” to get implementation-focused action items.</li>
              </ul>
            </div>

            <!-- Q&A mode -->
            <div class="space-y-2">
              <div class="flex items-center justify-between">
                <h3 class="text-xs font-semibold uppercase tracking-wide text-slate-500 flex items-center gap-2">
                  <i class="fa-solid fa-circle-question text-teal-500"></i>
                  Q&amp;A Mode
                </h3>
                <button id="generate-qa"
                        class="text-[11px] px-2 py-1 rounded-full border border-slate-300 text-slate-700 hover:border-teal-500 hover:text-teal-600 hover:bg-teal-50 transition">
                  Generate
                </button>
              </div>
              <div id="qa-list" class="space-y-2 text-[12px] text-slate-700">
                <p class="text-slate-500 italic">Click “Generate” to create Q&amp;A pairs for revision or viva prep.</p>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>

    <div id="app-data" data-doc-id="{{ doc_id }}"></div>
  </main>

  <script>
    const appDataEl = document.getElementById('app-data');
    const DOC_ID = appDataEl ? appDataEl.dataset.docId : null;

    async function postJSON(url, payload) {
      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload || {})
      });
      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(txt || 'Request failed');
      }
      return await resp.json();
    }

    // Chat
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatBox = document.getElementById('chat-box');

    function addChatMessage(role, text) {
      const div = document.createElement('div');
      div.className = 'flex gap-2';
      const badge = document.createElement('span');
      badge.className = 'px-1.5 py-0.5 rounded-full text-[10px] font-semibold ' +
                        (role === 'user' ? 'bg-slate-200 text-slate-700' : 'bg-teal-100 text-teal-700');
      badge.textContent = role === 'user' ? 'You' : 'Gemini';
      const msg = document.createElement('div');
      msg.className = 'text-[12px] text-slate-700';
      msg.textContent = text;
      div.appendChild(badge);
      div.appendChild(msg);
      chatBox.appendChild(div);
      chatBox.scrollTop = chatBox.scrollHeight;
    }

    if (chatForm) {
      chatForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = (chatInput.value || '').trim();
        if (!question || !DOC_ID) return;
        addChatMessage('user', question);
        chatInput.value = '';
        try {
          addChatMessage('assistant', 'Thinking...');
          const data = await postJSON('{{ url_for("chat") }}', { doc_id: DOC_ID, message: question });
          chatBox.lastChild.remove(); // remove "Thinking..."
          addChatMessage('assistant', data.answer || 'No answer returned.');
        } catch (err) {
          console.error(err);
          chatBox.lastChild.remove();
          addChatMessage('assistant', 'Error contacting Gemini or server.');
        }
      });
    }

    // Action items
    const actionsBtn = document.getElementById('generate-actions');
    const actionsList = document.getElementById('actions-list');
    if (actionsBtn && actionsList) {
      actionsBtn.addEventListener('click', async () => {
        if (!DOC_ID) return;
        actionsBtn.disabled = true;
        actionsBtn.textContent = 'Generating...';
        actionsList.innerHTML = '<li class="text-[12px] text-slate-500 italic">Generating action items with Gemini...</li>';
        try {
          const data = await postJSON('{{ url_for("actions") }}', { doc_id: DOC_ID });
          actionsList.innerHTML = '';
          (data.items || []).forEach(item => {
            const li = document.createElement('li');
            li.className = 'text-[12px] text-slate-700';
            li.textContent = item;
            actionsList.appendChild(li);
          });
        } catch (err) {
          console.error(err);
          actionsList.innerHTML = '<li class="text-[12px] text-red-500">Error generating action items.</li>';
        } finally {
          actionsBtn.disabled = false;
          actionsBtn.textContent = 'Generate';
        }
      });
    }

    // Q&A
    const qaBtn = document.getElementById('generate-qa');
    const qaList = document.getElementById('qa-list');
    if (qaBtn && qaList) {
      qaBtn.addEventListener('click', async () => {
        if (!DOC_ID) return;
        qaBtn.disabled = true;
        qaBtn.textContent = 'Generating...';
        qaList.innerHTML = '<p class="text-[12px] text-slate-500 italic">Generating Q&A pairs with Gemini...</p>';
        try {
          const data = await postJSON('{{ url_for("qa") }}', { doc_id: DOC_ID });
          qaList.innerHTML = '';
          (data.qas || []).forEach(item => {
            const wrapper = document.createElement('div');
            wrapper.className = 'bg-slate-50 border border-slate-200 rounded-lg p-2';
            const q = document.createElement('p');
            q.className = 'font-semibold text-[12px] text-slate-800';
            q.textContent = 'Q: ' + (item.question || '');
            const a = document.createElement('p');
            a.className = 'text-[12px] text-slate-700 mt-1';
            a.textContent = 'A: ' + (item.answer || '');
            wrapper.appendChild(q);
            wrapper.appendChild(a);
            qaList.appendChild(wrapper);
          });
        } catch (err) {
          console.error(err);
          qaList.innerHTML = '<p class="text-[12px] text-red-500">Error generating Q&A.</p>';
        } finally {
          qaBtn.disabled = false;
          qaBtn.textContent = 'Generate';
        }
      });
    }
  </script>
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
        r"\b(reduce|increase|improve|achieve|eliminate|raise|reach|decrease|enhance)\b", s_lower
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


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
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
        "integrity", "patient-centred", "patient-centered",
    ]):
        return "policy principles"

    if any(w in s_lower for w in [
        "primary care", "secondary care", "tertiary care",
        "health and wellness centre", "health & wellness centre",
        "health & wellness center", "hospital", "service delivery",
        "referral", "emergency services", "free drugs", "free diagnostics",
    ]):
        return "service delivery"

    if any(w in s_lower for w in [
        "prevention", "preventive", "promotive", "promotion",
        "sanitation", "nutrition", "tobacco", "alcohol",
        "air pollution", "road safety", "lifestyle", "behaviour change",
        "behavior change", "swachh", "clean water",
    ]):
        return "prevention & promotion"

    if any(w in s_lower for w in [
        "human resources for health", "hrh", "health workforce",
        "doctors", "nurses", "mid-level", "medical college",
        "nursing college", "public health management cadre",
        "training", "capacity building",
    ]):
        return "human resources"

    if any(w in s_lower for w in [
        "financing", "financial protection", "insurance",
        "strategic purchasing", "public spending", "gdp",
        "expenditure", "catastrophic", "private sector", "ppp",
        "reimbursement", "fees", "empanelment",
    ]):
        return "financing & private sector"

    if any(w in s_lower for w in [
        "digital health", "health information", "ehr",
        "electronic health record", "telemedicine",
        "information system", "surveillance", "ndha", "health data",
    ]):
        return "digital health"

    if any(w in s_lower for w in [
        "ayush", "ayurveda", "yoga", "unani", "siddha", "homeopathy",
    ]):
        return "ayush integration"

    if any(w in s_lower for w in [
        "implementation", "way forward", "roadmap",
        "action plan", "strategy", "governance",
        "monitoring", "evaluation", "framework",
    ]):
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


# ---------------------- FLASK ROUTES ---------------------- #

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/summarize", methods=["POST"])
def summarize():
    file_storage = request.files.get("file")
    image_data_url = request.form.get("image_data", "").strip()
    length_choice = request.form.get("length", "medium").lower()
    tone = request.form.get("tone", "academic").lower()

    raw_text = ""
    doc_kind = None
    saved_filename = None

    if file_storage and file_storage.filename:
        filename = file_storage.filename.lower()
        if filename.endswith(".pdf"):
            pdf_bytes = file_storage.read()
            raw_text = extract_text_from_pdf_bytes(pdf_bytes)
            doc_kind = "pdf"
            doc_id = uuid.uuid4().hex
            saved_filename = f"{doc_id}.pdf"
            with open(os.path.join(UPLOAD_FOLDER, saved_filename), "wb") as f:
                f.write(pdf_bytes)
        elif filename.endswith(".txt"):
            try:
                raw_bytes = file_storage.read()
                raw_text = raw_bytes.decode("utf-8", errors="ignore")
            except Exception as e:
                abort(400, f"Error reading text file: {e}")
            doc_kind = "text"
            doc_id = uuid.uuid4().hex
        else:
            abort(400, "Unsupported file format. Please upload a PDF or .txt file.")
    elif image_data_url:
        m = re.match(r"data:image/[^;]+;base64,(.+)", image_data_url)
        if not m:
            abort(400, "Invalid image data.")
        import base64
        image_bytes = base64.b64decode(m.group(1))
        raw_text = extract_text_with_gemini_from_image(image_bytes)
        if not raw_text:
            abort(400, "Could not extract text from image using Gemini.")
        doc_kind = "image"
        doc_id = uuid.uuid4().hex
        saved_filename = f"{doc_id}.png"
        with open(os.path.join(UPLOAD_FOLDER, saved_filename), "wb") as f:
            f.write(image_bytes)
    else:
        abort(400, "Please upload a PDF/TXT file or capture a document with the camera.")

    if not raw_text or len(raw_text.strip()) < 50:
        abort(400, "Uploaded document appears to be empty or too short after text extraction.")

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

    DOC_STORE[doc_id] = {
        "text": raw_text,
        "summary_sentences": summary_sentences,
        "title": doc_title or "Policy Document",
        "kind": doc_kind,
        "file_name": saved_filename,
    }

    doc_file_url = None
    if doc_kind in ("pdf", "image") and saved_filename:
        doc_file_url = request.url_root.rstrip("/") + "/uploads/" + saved_filename

    doc_text_for_view = raw_text if doc_kind == "text" else None

    return render_template_string(
        RESULT_HTML,
        title=title,
        abstract=structured["abstract"],
        sections=structured["sections"],
        category_counts=structured["category_counts"],
        implementation_points=structured["implementation_points"],
        stats=stats,
        doc_id=doc_id,
        doc_kind=doc_kind,
        doc_file_url=doc_file_url,
        doc_text=doc_text_for_view,
    )


@app.route("/download_summary/<doc_id>", methods=["GET"])
def download_summary_pdf(doc_id):
    doc = DOC_STORE.get(doc_id)
    if not doc:
        abort(404, "Unknown document ID.")

    summary_sentences = doc.get("summary_sentences", [])
    title = doc.get("title", "Policy Summary")

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    textobject = c.beginText(40, height - 60)
    textobject.setFont("Helvetica-Bold", 14)
    textobject.textLine(title)
    textobject.moveCursor(0, 14)
    textobject.setFont("Helvetica", 11)

    wrapper = textwrap.TextWrapper(width=90)
    for sent in summary_sentences:
        for line in wrapper.wrap(sent):
            textobject.textLine(line)
        textobject.textLine("")  # blank line between bullets

    c.drawText(textobject)
    c.showPage()
    c.save()
    buffer.seek(0)

    safe_title = re.sub(r"[^A-Za-z0-9]+", "_", title)[:50] or "summary"
    return send_file(
        buffer,
        as_attachment=True,
        download_name=f"{safe_title}.pdf",
        mimetype="application/pdf",
    )


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    doc_id = data.get("doc_id")
    message = (data.get("message") or "").strip()
    if not doc_id or not message:
        abort(400, "doc_id and message are required.")
    doc = DOC_STORE.get(doc_id)
    if not doc:
        abort(404, "Unknown document ID.")
    answer = gemini_answer_for_doc(doc["text"], message)
    return jsonify({"answer": answer})


@app.route("/actions", methods=["POST"])
def actions():
    data = request.get_json(force=True)
    doc_id = data.get("doc_id")
    if not doc_id:
        abort(400, "doc_id is required.")
    doc = DOC_STORE.get(doc_id)
    if not doc:
        abort(404, "Unknown document ID.")
    items = gemini_action_items_for_doc(doc["text"])
    return jsonify({"items": items})


@app.route("/qa", methods=["POST"])
def qa():
    data = request.get_json(force=True)
    doc_id = data.get("doc_id")
    if not doc_id:
        abort(400, "doc_id is required.")
    doc = DOC_STORE.get(doc_id)
    if not doc:
        abort(404, "Unknown document ID.")
    qas = gemini_qa_for_doc(doc["text"])
    return jsonify({"qas": qas})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
