/**
 * News Category Classifier - Modern Dashboard JavaScript
 */

document.addEventListener("DOMContentLoaded", () => {
  // Determine Flask API URL
  const isServedFromBackend = (window.location.protocol === "http:" || window.location.protocol === "https:")
    && (window.location.pathname.includes("/app") || window.location.port === "5050" || window.location.port === "5000");

  const API_BASE = isServedFromBackend
    ? window.location.origin
    : (sessionStorage.getItem("API_BASE") || "http://127.0.0.1:5050");

  // Category Configuration
  const CATEGORY_META = {
    "World": {
      tag: "WO",
      icon: "",
      color: "#2563EB",
      bg: "#EFF6FF",
      border: "#BFDBFE",
      classSlug: "cat-world",
      pillClass: "pill-world",
    },
    "Sports": {
      tag: "SP",
      icon: "",
      color: "#059669",
      bg: "#ECFDF5",
      border: "#A7F3D0",
      classSlug: "cat-sports",
      pillClass: "pill-sports",
    },
    "Business": {
      tag: "BU",
      icon: "",
      color: "#D97706",
      bg: "#FFFBEB",
      border: "#FDE68A",
      classSlug: "cat-business",
      pillClass: "pill-business",
    },
    "Sci/Tech": {
      tag: "ST",
      icon: "",
      color: "#7C3AED",
      bg: "#F5F3FF",
      border: "#DDD6FE",
      classSlug: "cat-scitech",
      pillClass: "pill-scitech",
    },
    "Unknown": {
      tag: "--",
      icon: "",
      color: "#64748B",
      bg: "#F1F5F9",
      border: "#CBD5E1",
      classSlug: "",
      pillClass: "",
    },
  };

  // State
  let currentFile = null;
  let currentBatchResults = [];

  // DOM Elements - Navigation
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");
  const apiStatus = document.getElementById("apiStatus");
  const apiStatusText = document.getElementById("apiStatusText");
  const statusDot = apiStatus.querySelector(".status-dot");

  // DOM Elements - Tab 1 (Single Prediction)
  const predictionForm = document.getElementById("predictionForm");
  const headlineInput = document.getElementById("headlineInput");
  const charCount = document.getElementById("charCount");
  const modelSelect = document.getElementById("modelSelect");
  const modelHint = document.getElementById("modelHint");
  const btnClassify = document.getElementById("btnClassify");
  const predictSpinner = document.getElementById("predictSpinner");
  const btnClearText = document.getElementById("btnClearText");
  const presetChips = document.querySelectorAll(".preset-chip");

  const predictEmptyState = document.getElementById("predictEmptyState");
  const predictResultBody = document.getElementById("predictResultBody");
  const predictErrorAlert = document.getElementById("predictErrorAlert");
  const predictErrorMsg = document.getElementById("predictErrorMsg");
  const latencyBadge = document.getElementById("latencyBadge");
  const lowConfidenceAlert = document.getElementById("lowConfidenceAlert");

  const predictedBanner = document.getElementById("predictedBanner");
  const categoryBadge = document.getElementById("categoryBadge");
  const categoryIcon = document.getElementById("categoryIcon");
  const categoryLabel = document.getElementById("categoryLabel");
  const confidenceValue = document.getElementById("confidenceValue");
  const confidenceBarFill = document.getElementById("confidenceBarFill");
  const probabilityBars = document.getElementById("probabilityBars");
  const explanationSection = document.getElementById("explanationSection");
  const explanationBars = document.getElementById("explanationBars");
  const explanationEmpty = document.getElementById("explanationEmpty");

  // DOM Elements - Tab 2 (Batch Prediction)
  const uploadDropzone = document.getElementById("uploadDropzone");
  const batchFileInput = document.getElementById("batchFileInput");
  const btnBrowseFile = document.getElementById("btnBrowseFile");
  const fileConfigBar = document.getElementById("fileConfigBar");
  const batchFileName = document.getElementById("batchFileName");
  const batchFileMeta = document.getElementById("batchFileMeta");
  const textColSelect = document.getElementById("textColSelect");
  const btnRunBatch = document.getElementById("btnRunBatch");
  const batchSpinner = document.getElementById("batchSpinner");
  const btnRemoveFile = document.getElementById("btnRemoveFile");
  const btnDownloadSampleCsv = document.getElementById("btnDownloadSampleCsv");
  const batchErrorAlert = document.getElementById("batchErrorAlert");
  const batchErrorMsg = document.getElementById("batchErrorMsg");
  const batchResultsSection = document.getElementById("batchResultsSection");
  const batchSummaryStats = document.getElementById("batchSummaryStats");
  const batchTableBody = document.getElementById("batchTableBody");
  const btnDownloadResultsCsv = document.getElementById("btnDownloadResultsCsv");

  // DOM Elements - Tab 3 (Evaluation)
  const confusionMatrixImg = document.getElementById("confusionMatrixImg");
  const valAccuracy = document.getElementById("valAccuracy");
  const valMacroF1 = document.getElementById("valMacroF1");
  const valMacroPrecision = document.getElementById("valMacroPrecision");
  const valLatency = document.getElementById("valLatency");
  const perClassTableBody = document.getElementById("perClassTableBody");

  // ==========================================================
  // 1. Initial Backend Connectivity & Setup
  // ==========================================================
  async function checkBackendHealth() {
    try {
      const res = await fetch(`${API_BASE}/api/health`, { method: "GET" });
      if (res.ok) {
        const data = await res.json();
        if (data.status === "ok") {
          statusDot.className = "status-dot status-online";
          apiStatusText.textContent = "API Connected";
          apiStatus.title = `Flask Backend Online (${API_BASE})`;
          checkModelReadiness();
          return true;
        }
      }
      throw new Error("Invalid status");
    } catch (err) {
      statusDot.className = "status-dot status-offline";
      apiStatusText.textContent = "API Offline";
      apiStatus.title = `Unable to connect to the prediction server (${API_BASE}). Start backend with: python -m backend.app`;
      return false;
    }
  }

  async function checkModelReadiness() {
    try {
      const res = await fetch(`${API_BASE}/api/models`);
      if (res.ok) {
        const data = await res.json();
        const distilbert = data.models?.find(m => m.id === "distilbert");
        const distilbertOpt = document.getElementById("distilbertOption");
        if (distilbertOpt) {
          if (distilbert && distilbert.available) {
            distilbertOpt.disabled = false;
            distilbertOpt.textContent = "DistilBERT — Fine-tuned Transformer (Available)";
          } else {
            distilbertOpt.disabled = true;
            distilbertOpt.textContent = "DistilBERT — Fine-tuned Transformer (Not Trained on Disk)";
          }
        }
      }
    } catch (_) {
      // Keep baseline default
    }
  }

  // ==========================================================
  // 2. Navigation Tabs Logic
  // ==========================================================
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetTabId = btn.getAttribute("data-tab");

      tabBtns.forEach(b => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      tabPanels.forEach(p => {
        p.classList.remove("active");
        p.classList.add("hidden");
      });

      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");

      const activePanel = document.getElementById(targetTabId);
      if (activePanel) {
        activePanel.classList.remove("hidden");
        activePanel.classList.add("active");
      }

      if (targetTabId === "tabEvaluation") {
        loadEvaluationMetrics();
      }
    });
  });

  // ==========================================================
  // 3. Tab 1: Single Prediction Logic
  // ==========================================================
  function updateCharCount() {
    const len = headlineInput.value.length;
    charCount.textContent = `${len} character${len === 1 ? "" : "s"}`;
  }

  headlineInput.addEventListener("input", updateCharCount);
  headlineInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      predictionForm.requestSubmit();
    }
  });
  updateCharCount();

  // Copy prediction result to clipboard
  const btnCopyPrediction = document.getElementById("btnCopyPrediction");
  if (btnCopyPrediction) {
    btnCopyPrediction.addEventListener("click", () => {
      const label = categoryLabel ? categoryLabel.textContent : "";
      const conf = confidenceValue ? confidenceValue.textContent : "";
      const copyText = `Predicted Category: ${label} (Confidence: ${conf})`;
      navigator.clipboard.writeText(copyText).then(() => {
        btnCopyPrediction.textContent = "Copied!";
        btnCopyPrediction.classList.add("copied");
        setTimeout(() => {
          btnCopyPrediction.textContent = "Copy";
          btnCopyPrediction.classList.remove("copied");
        }, 2000);
      }).catch(() => {
        btnCopyPrediction.textContent = "Copied!";
        setTimeout(() => {
          btnCopyPrediction.textContent = "Copy";
        }, 2000);
      });
    });
  }

  // Preset chips handler
  presetChips.forEach(chip => {
    chip.addEventListener("click", () => {
      headlineInput.value = chip.getAttribute("data-text");
      updateCharCount();
      headlineInput.focus();
    });
  });

  // Clear button
  btnClearText.addEventListener("click", () => {
    headlineInput.value = "";
    updateCharCount();
    predictEmptyState.classList.remove("hidden");
    predictResultBody.classList.add("hidden");
    predictErrorAlert.classList.add("hidden");
    latencyBadge.classList.add("hidden");
    if (explanationBars) explanationBars.innerHTML = "";
    if (explanationEmpty) explanationEmpty.classList.add("hidden");
    headlineInput.focus();
  });

  // Submit prediction
  predictionForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const text = headlineInput.value.trim();
    if (!text) {
      showPredictError("Please enter some news text.");
      headlineInput.focus();
      return;
    }

    if (text.length > 50000) {
      showPredictError("Input text is too long (maximum 50,000 characters). Please provide a shorter headline or excerpt.");
      return;
    }

    const model = modelSelect.value;
    setPredictLoading(true);

    const startTime = performance.now();

    try {
      const response = await fetch(`${API_BASE}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, model }),
      });

      const elapsedMs = (performance.now() - startTime).toFixed(2);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `Server responded with status ${response.status}`);
      }

      if (data.label === "Unknown" && (!data.confidence || data.confidence === 0)) {
        showPredictError(data.warning || "Please enter some news text.");
        return;
      }

      renderPredictionResult(data, elapsedMs);
    } catch (err) {
      const isNetworkErr = err.name === "TypeError" || String(err).includes("Failed to fetch") || String(err).includes("NetworkError");
      if (isNetworkErr) {
        showPredictError("Unable to connect to the prediction server. Please make sure the Flask backend is running on http://127.0.0.1:5000.");
      } else {
        showPredictError(err.message || "Unable to connect to the prediction server.");
      }
    } finally {
      setPredictLoading(false);
    }
  });

  function setPredictLoading(isLoading) {
    btnClassify.disabled = isLoading;
    if (isLoading) {
      predictSpinner.classList.remove("hidden");
    } else {
      predictSpinner.classList.add("hidden");
    }
  }

  function showPredictError(message) {
    predictErrorMsg.textContent = message;
    predictErrorAlert.classList.remove("hidden");
    predictResultBody.classList.add("hidden");
    predictEmptyState.classList.add("hidden");
    latencyBadge.classList.add("hidden");
  }

  function renderPredictionResult(result, elapsedMs) {
    predictErrorAlert.classList.add("hidden");
    predictEmptyState.classList.add("hidden");
    predictResultBody.classList.remove("hidden");

    // Display inference latency
    latencyBadge.textContent = `${elapsedMs} ms`;
    latencyBadge.classList.remove("hidden");

    const label = result.label || "Unknown";
    const confidence = typeof result.confidence === "number" ? result.confidence : 0;
    const confidencePct = (confidence * 100).toFixed(1) + "%";
    const meta = CATEGORY_META[label] || CATEGORY_META["Unknown"];

    // Update Banner
    predictedBanner.className = `predicted-banner ${meta.classSlug}`;
    categoryBadge.textContent = label;
    categoryBadge.className = `banner-badge ${meta.pillClass}`;
    categoryIcon.textContent = meta.tag;
    categoryIcon.style.color = meta.color;
    categoryLabel.textContent = label.toUpperCase();
    confidenceValue.textContent = confidencePct;
    confidenceBarFill.style.width = confidencePct;

    // F7 & Task 4 & Task 10: Low-confidence / uncertainty warning or short text notice
    const warningStrong = lowConfidenceAlert.querySelector(".alert-content strong");
    const warningP = lowConfidenceAlert.querySelector(".alert-content p");

    if (result.warning) {
      lowConfidenceAlert.classList.remove("hidden");
      if (warningStrong) warningStrong.textContent = "Reliability Notice";
      if (warningP) warningP.textContent = result.warning;
    } else if (result.uncertain === true || confidence < 0.60) {
      lowConfidenceAlert.classList.remove("hidden");
      if (warningStrong) warningStrong.textContent = "Uncertain Prediction";
      if (warningP) warningP.textContent = "The model has low confidence in this classification (< 60%). The text may be ambiguous, multi-topic, or out-of-domain.";
    } else {
      lowConfidenceAlert.classList.add("hidden");
    }

    // Render Top-K Probabilities
    renderProbabilityBars(result.top_k_probs || []);

    // Render Explainable AI (Why this prediction?)
    renderExplanation(result.explanation || []);
  }

  function renderExplanation(explanationList) {
    explanationBars.innerHTML = "";

    if (!explanationList || explanationList.length === 0) {
      explanationEmpty.classList.remove("hidden");
      return;
    }

    explanationEmpty.classList.add("hidden");

    let maxAbs = 0.0001;
    explanationList.forEach(item => {
      const absVal = Math.abs(item.contribution || 0);
      if (absVal > maxAbs) maxAbs = absVal;
    });

    explanationList.forEach(item => {
      const feat = item.feature;
      const contrib = typeof item.contribution === "number" ? item.contribution : 0;
      const isPositive = contrib >= 0;
      const sign = isPositive ? "+" : "";
      const widthPct = Math.min(100, Math.max(6, (Math.abs(contrib) / maxAbs) * 100)).toFixed(1);

      const row = document.createElement("div");
      row.className = "explanation-row";
      row.innerHTML = `
        <div class="explanation-meta">
          <span class="feature-name">${escapeHtml(feat)}</span>
          <span class="feature-score ${isPositive ? "positive" : "negative"}">${sign}${contrib.toFixed(4)}</span>
        </div>
        <div class="feature-track">
          <div class="feature-fill ${isPositive ? "positive" : "negative"}" style="width: ${widthPct}%;"></div>
        </div>
      `;
      explanationBars.appendChild(row);
    });
  }

  function renderProbabilityBars(probsList) {
    probabilityBars.innerHTML = "";

    probsList.forEach(item => {
      const cat = item.label;
      const prob = item.probability || 0;
      const pct = (prob * 100).toFixed(1);
      const meta = CATEGORY_META[cat] || CATEGORY_META["Unknown"];

      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <div class="prob-meta">
          <span class="prob-label">
            <span>${cat}</span>
          </span>
          <span class="prob-val">${pct}%</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill" style="width: ${pct}%; background-color: ${meta.color};"></div>
        </div>
      `;
      probabilityBars.appendChild(row);
    });
  }

  // ==========================================================
  // 4. Tab 2: Batch CSV Prediction Logic
  // ==========================================================
  // Dropzone drag/drop listeners
  ["dragenter", "dragover"].forEach(evtName => {
    uploadDropzone.addEventListener(evtName, (e) => {
      e.preventDefault();
      uploadDropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach(evtName => {
    uploadDropzone.addEventListener(evtName, (e) => {
      e.preventDefault();
      uploadDropzone.classList.remove("dragover");
    });
  });

  uploadDropzone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileSelected(files[0]);
    }
  });

  btnBrowseFile.addEventListener("click", () => {
    batchFileInput.click();
  });

  batchFileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    if (!file.name.toLowerCase().endsWith(".csv")) {
      showBatchError("Invalid file type. Please upload a valid CSV file (.csv).");
      return;
    }

    currentFile = file;
    batchErrorAlert.classList.add("hidden");

    // Read header line to detect available columns
    const reader = new FileReader();
    reader.onload = (evt) => {
      const content = evt.target.result;
      const lines = content.split(/\r?\n/).filter(line => line.trim().length > 0);
      if (lines.length === 0) {
        showBatchError("The uploaded CSV file contains no data rows.");
        return;
      }

      // Simple CSV column parser for first line
      const headerLine = lines[0];
      const headers = parseCsvRow(headerLine);
      const estRows = Math.max(0, lines.length - 1);

      if (estRows === 0) {
        showBatchError("The uploaded CSV file contains only headers with no data rows.");
        return;
      }

      batchFileName.textContent = file.name;
      batchFileMeta.textContent = `${estRows} rows detected • ${(file.size / 1024).toFixed(1)} KB`;

      // Populate column selector
      textColSelect.innerHTML = "";
      let autoSelected = false;

      headers.forEach(h => {
        const colClean = h.trim();
        const opt = document.createElement("option");
        opt.value = colClean;
        opt.textContent = colClean;

        const lower = colClean.toLowerCase();
        if (!autoSelected && (lower.includes("headline") || lower.includes("text") || lower.includes("title") || lower.includes("news") || lower.includes("content"))) {
          opt.selected = true;
          autoSelected = true;
        }
        textColSelect.appendChild(opt);
      });

      fileConfigBar.classList.remove("hidden");
    };

    reader.onerror = () => {
      showBatchError("Failed to read the local CSV file.");
    };

    // Read initial 32KB to parse columns quickly
    reader.readAsText(file.slice(0, 32768));
  }

  function parseCsvRow(rowStr) {
    const cells = [];
    let insideQuotes = false;
    let currentCell = "";

    for (let i = 0; i < rowStr.length; i++) {
      const char = rowStr[i];
      if (char === '"' || char === "'") {
        insideQuotes = !insideQuotes;
      } else if (char === "," && !insideQuotes) {
        cells.push(currentCell.replace(/^["']|["']$/g, "").trim());
        currentCell = "";
      } else {
        currentCell += char;
      }
    }
    cells.push(currentCell.replace(/^["']|["']$/g, "").trim());
    return cells;
  }

  btnRemoveFile.addEventListener("click", () => {
    currentFile = null;
    currentBatchResults = [];
    batchFileInput.value = "";
    fileConfigBar.classList.add("hidden");
    batchResultsSection.classList.add("hidden");
    batchErrorAlert.classList.add("hidden");
  });

  // Run Batch Prediction
  btnRunBatch.addEventListener("click", async () => {
    if (!currentFile) {
      showBatchError("Please select a CSV file first.");
      return;
    }

    const textCol = textColSelect.value;
    if (!textCol) {
      showBatchError("Please select a valid text column.");
      return;
    }

    setBatchLoading(true);
    batchErrorAlert.classList.add("hidden");

    const formData = new FormData();
    formData.append("file", currentFile);
    formData.append("text_column", textCol);
    formData.append("model", "baseline");
    formData.append("format", "json");

    try {
      const response = await fetch(`${API_BASE}/api/batch-predict`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `Server responded with status ${response.status}`);
      }

      currentBatchResults = data.predictions || [];
      renderBatchResults(currentBatchResults);
    } catch (err) {
      const isNetworkErr = err.name === "TypeError" || String(err).includes("Failed to fetch") || String(err).includes("NetworkError");
      if (isNetworkErr) {
        showBatchError("Unable to connect to the prediction server. Please make sure the Flask backend is running on http://127.0.0.1:5000.");
      } else {
        showBatchError(err.message || "Unable to connect to the prediction server.");
      }
    } finally {
      setBatchLoading(false);
    }
  });

  function setBatchLoading(isLoading) {
    btnRunBatch.disabled = isLoading;
    if (isLoading) {
      batchSpinner.classList.remove("hidden");
    } else {
      batchSpinner.classList.add("hidden");
    }
  }

  function showBatchError(msg) {
    batchErrorMsg.textContent = msg;
    batchErrorAlert.classList.remove("hidden");
  }

  function renderBatchResults(predictions) {
    if (!predictions || predictions.length === 0) {
      showBatchError("No prediction records were returned.");
      return;
    }

    batchSummaryStats.textContent = `Successfully classified ${predictions.length} headlines`;
    batchTableBody.innerHTML = "";

    predictions.forEach((row, idx) => {
      const cat = row.predicted_category || "Unknown";
      const conf = typeof row.confidence === "number" ? (row.confidence * 100).toFixed(1) : "0.0";
      const meta = CATEGORY_META[cat] || CATEGORY_META["Unknown"];
      const isLowConf = row.uncertain || row.confidence < 0.60;

      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${idx + 1}</td>
        <td class="table-text" title="${escapeHtml(row.text)}">${escapeHtml(row.text)}</td>
        <td>
          <span class="pill ${meta.pillClass}">
            ${cat}
          </span>
        </td>
        <td><strong>${conf}%</strong></td>
        <td>
          ${isLowConf 
            ? `<span style="color: #B45309; font-weight: 700; font-size: 0.78rem;">Low Confidence</span>`
            : `<span style="color: #059669; font-weight: 600; font-size: 0.78rem;">Confident</span>`
          }
        </td>
      `;
      batchTableBody.appendChild(tr);
    });

    batchResultsSection.classList.remove("hidden");
  }

  // Download Sample CSV
  btnDownloadSampleCsv.addEventListener("click", () => {
    const sampleCsv = 
`headline
"Messi scores twice in Champions League final as team secures historic treble victory"
"NASA James Webb telescope reveals high-resolution spectroscopic data of distant exoplanet atmosphere"
"Federal Reserve signals interest rate adjustments following quarterly corporate earnings surge"
"United Nations security council diplomats convene in Geneva for international peace treaty negotiations"
"Lakers capture NBA championship victory in thrilling Game 7 overtime thriller"
"Tech giants unveil open-source large language model benchmark evaluation suite"
"Oil futures surge amid supply chain shifts in European energy markets"
"Global climate summit leads to multi-nation renewable infrastructure pact"
`;
    downloadTextAsFile("sample_news_headlines.csv", sampleCsv, "text/csv");
  });

  // Download Batch Results CSV
  btnDownloadResultsCsv.addEventListener("click", () => {
    if (!currentBatchResults || currentBatchResults.length === 0) return;

    let csvContent = "text,predicted_category,confidence,uncertain\n";
    currentBatchResults.forEach(r => {
      const safeText = `"${(r.text || "").replace(/"/g, '""')}"`;
      csvContent += `${safeText},"${r.predicted_category}",${r.confidence},${r.uncertain ? "true" : "false"}\n`;
    });

    downloadTextAsFile("classified_news_predictions.csv", csvContent, "text/csv");
  });

  function downloadTextAsFile(filename, text, mimeType) {
    const blob = new Blob([text], { type: mimeType });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(link.href);
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // ==========================================================
  // 5. Tab 3: Evaluation Dashboard Logic
  // ==========================================================
  async function loadEvaluationMetrics() {
    // Ensure image src is set
    confusionMatrixImg.src = `${API_BASE}/api/confusion-matrix?t=${Date.now()}`;

    try {
      const res = await fetch(`${API_BASE}/api/metrics`);
      if (!res.ok) throw new Error("Failed to load metrics");
      const data = await res.json();

      // Overall metrics
      if (data.accuracy !== undefined) {
        valAccuracy.textContent = `${(data.accuracy * 100).toFixed(2)}%`;
      }
      if (data.macro_f1 !== undefined) {
        valMacroF1.textContent = Number(data.macro_f1).toFixed(4);
      }
      if (data.macro_precision !== undefined) {
        valMacroPrecision.textContent = Number(data.macro_precision).toFixed(4);
      }
      if (data.avg_latency_ms_per_sample !== undefined) {
        valLatency.textContent = `${Number(data.avg_latency_ms_per_sample).toFixed(2)} ms`;
      }

      // Per-Class breakdown table
      if (data.per_class) {
        perClassTableBody.innerHTML = "";
        Object.entries(data.per_class).forEach(([catName, pcm]) => {
          const meta = CATEGORY_META[catName] || CATEGORY_META["Unknown"];
          const prec = (pcm.precision * 100).toFixed(2) + "%";
          const rec = (pcm.recall * 100).toFixed(2) + "%";
          const f1 = Number(pcm["f1-score"]).toFixed(4);
          const support = Number(pcm.support).toLocaleString();

          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td><span class="pill ${meta.pillClass}">${catName}</span></td>
            <td>${prec}</td>
            <td>${rec}</td>
            <td><strong>${f1}</strong></td>
            <td>${support}</td>
          `;
          perClassTableBody.appendChild(tr);
        });
      }
    } catch (err) {
      console.warn("Could not fetch real-time evaluation metrics:", err);
    }
  }

  // Initial check
  checkBackendHealth();
});
