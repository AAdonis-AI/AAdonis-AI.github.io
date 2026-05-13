---
title: "Curriculum Vitae"
layout: "single"
url: "/cv/"
summary: "Curriculum Vitae"
ShowToc: false
hideMeta: true
---

<style>
.cv-section {
  margin-bottom: 1.2rem;
}
.cv-section-title {
  font-size: 0.79rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--secondary);
  margin-bottom: 0.5rem;
  padding-bottom: 0.3rem;
  border-bottom: 1px solid var(--border);
}
.cv-entry {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.65rem 1rem;
  margin-bottom: 0.5rem;
  background: var(--entry);
}
.cv-entry h3 {
  margin: 0 0 0.1rem 0;
  font-size: 0.9rem;
}
.cv-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 0.3rem;
  margin-bottom: 0.25rem;
}
.cv-entry .role {
  font-weight: 500;
  font-size: 0.82rem;
}
.cv-entry .meta {
  color: var(--secondary);
  font-size: 0.75rem;
}
.cv-entry p {
  font-size: 0.78rem;
  line-height: 1.45;
  margin: 0;
  color: var(--secondary);
}

/* Athletics achievement grid */
.cv-achievements {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.5rem;
  margin-top: 0.5rem;
}
@media (max-width: 480px) {
  .cv-achievements {
    grid-template-columns: 1fr;
  }
}
.cv-achievement {
  display: flex;
  flex-direction: column;
  gap: 0.1rem;
}
.cv-achievement-rank {
  font-size: 1.15rem;
  font-weight: 700;
  color: var(--primary);
  line-height: 1.2;
}
.cv-achievement-desc {
  font-size: 0.73rem;
  color: var(--secondary);
  line-height: 1.35;
}

/* Compact project list inside a single card */
.cv-projects {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.65rem 1rem;
  background: var(--entry);
}
.cv-project-item {
  display: grid;
  grid-template-columns: auto 1fr;
  column-gap: 0.7rem;
  align-items: baseline;
  padding: 0.25rem 0;
  border-bottom: 1px solid var(--border);
}
.cv-project-item:last-child {
  border-bottom: none;
}
.cv-project-title {
  font-size: 0.82rem;
  font-weight: 600;
  white-space: nowrap;
}
.cv-project-desc {
  font-size: 0.76rem;
  color: var(--secondary);
  line-height: 1.4;
  text-align: left;
}
@media (max-width: 600px) {
  .cv-project-item {
    grid-template-columns: 1fr;
    row-gap: 0.05rem;
  }
  .cv-project-title {
    font-size: 0.75rem;
  }
  .cv-project-desc {
    font-size: 0.68rem;
  }
}

/* Languages — plain, no card */
.cv-languages {
  font-size: 0.8rem;
  color: var(--secondary);
}
.cv-languages strong {
  color: var(--primary);
}
</style>

<div style="display:flex; justify-content:space-between; align-items:baseline; margin-top:0.1rem; margin-bottom:1.6rem;">
  <span style="font-size:0.85rem; color:var(--secondary);">Adonis Asonitis</span>
  <a href="mailto:aasonitis@ethz.ch" style="font-size:0.78rem; color:var(--secondary);">aasonitis@ethz.ch</a>
</div>

<div class="cv-section">
<div class="cv-section-title">Experience</div>

<div class="cv-entry">
<h3>AGIGO</h3>
<div class="cv-header">
  <span class="role">AI Research Engineer (Internship) — Conversational AI, speech processing and post-training</span>
  <span class="meta">Zurich · Jan 2026 – Present</span>
</div>
<ul style="font-size:0.78rem; line-height:1.45; margin:0.25rem 0 0 1rem; padding:0; color:var(--secondary);">
  <li>Designed post-training methods giving fine-grained control over zero-shot pronunciation and cross-lingual phonetic nuances.</li>
  <li>Built end-to-end data pipelines and evaluation infrastructure, including internal blind listening tests, automated objective eval suites, and dataset curation/QA workflows used across TTS model training.</li>
</ul>
</div>

<div class="cv-entry">
<h3>ETH Zurich — DISCO Lab</h3>
<div class="cv-header">
  <span class="role">Student Researcher, Distributed Computing</span>
  <span class="meta">Zurich · Feb 2025 – Present</span>
</div>
<p>Research student under Prof. Roger Wattenhofer. Speech generation, multilingual audio data, codec language models, and reinforcement learning. First-authored WorldSpeech, the largest publicly available human-transcribed multilingual speech corpus (65k hours across 80+ languages, 35k+ HuggingFace downloads in first week). Additional work on multilingual speech editing, zero-shot voice conversion, speech enhancement language models, and RL post-training for speech enhancement.</p>
</div>
</div>

<div class="cv-section">
<div class="cv-section-title">Education</div>

<div class="cv-entry">
<h3>ETH Zurich — MSc Computer Science</h3>
<div class="cv-header">
  <span class="meta">2025 – Present · Zurich</span>
</div>
<p>Machine Intelligence · minor in Data Management Systems.</p>
</div>

<div class="cv-entry">
<h3>ETH Zurich — BSc Computer Science</h3>
<div class="cv-header">
  <span class="meta">2022 – 2025 · Zurich</span>
</div>
<p>Statistical modeling, ML, computer systems, applied mathematics. Top-graded thesis (6/6), developed into a top-tier ML conference submission.</p>
</div>
</div>

<div class="cv-section">
<div class="cv-section-title">Competitions & Projects</div>

<div class="cv-projects">
  <div class="cv-project-item">
    <span class="cv-project-title">Swiss AI Datathon 2024</span>
    <span class="cv-project-desc">1st place (200+ participants). Ensemble models for photovoltaic energy forecasting (one-day-ahead market, >90% accuracy).</span>
  </div>
  <div class="cv-project-item">
    <span class="cv-project-title">Greek Urban Planning RAG</span>
    <span class="cv-project-desc">RAG system for lawyers and engineers navigating Greek urban planning legislation.</span>
  </div>
  <div class="cv-project-item">
    <span class="cv-project-title">Gen-AI for DolliBar</span>
    <span class="cv-project-desc">Generative AI pipeline converting receipt images into structured enterprise expense entries.</span>
  </div>
  <div class="cv-project-item">
    <span class="cv-project-title">Arbitrage System</span>
    <span class="cv-project-desc">High-performance cross-platform arbitrage detection and execution engine in C++.</span>
  </div>
</div>
</div>

<div class="cv-section">
<div class="cv-section-title">Athletics</div>

<div class="cv-entry">
<h3>Swiss Olympic Sailing Team</h3>
<div class="cv-header">
  <span class="role">National Team Athlete</span>
  <span class="meta">2016 – 2020</span>
</div>
<div style="font-size:0.75rem; color:var(--secondary); margin-bottom:0.5rem;">ILCA · 220k+ boats worldwide</div>
<div class="cv-achievements">
  <div class="cv-achievement">
    <span class="cv-achievement-rank">1st</span>
    <span class="cv-achievement-desc">European Cup standings (2018)</span>
  </div>
  <div class="cv-achievement">
    <span class="cv-achievement-rank">4×</span>
    <span class="cv-achievement-desc">Swiss national medals</span>
  </div>
  <div class="cv-achievement">
    <span class="cv-achievement-rank">8th</span>
    <span class="cv-achievement-desc">U17 European Championship (2019)</span>
  </div>
  <div class="cv-achievement">
    <span class="cv-achievement-rank">16th</span>
    <span class="cv-achievement-desc">U16 World Championship (2017)</span>
  </div>
</div>
</div>
</div>

<div class="cv-section">
<div class="cv-section-title">Languages</div>
<div class="cv-languages">
  <strong>English</strong> · Fluent&emsp;
  <strong>French</strong> · Fluent&emsp;
  <strong>Greek</strong> · Fluent&emsp;
  <strong>German</strong> · Advanced
</div>
</div>
