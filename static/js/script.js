// ===================================================================
// NeuraScan AI — upload + analysis flow controller
// ===================================================================

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const previewImg = document.getElementById('previewImg');
const analyzeBtn = document.getElementById('analyzeBtn');
const resetBtn = document.getElementById('resetBtn');

const scanningState = document.getElementById('scanningState');
const resultContent = document.getElementById('resultContent');

let selectedFile = null;

// ---------------- Upload interactions ----------------
dropzone.addEventListener('click', () => fileInput.click());

dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});
dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  if (!file.type.startsWith('image/')) {
    alert('Please upload an image file (JPG or PNG).');
    return;
  }
  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    dropzone.classList.add('has-image');
  };
  reader.readAsDataURL(file);

  analyzeBtn.disabled = false;
}

// ---------------- Analyze → real backend prediction ----------------
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  resultContent.classList.remove('active');
  scanningState.classList.add('active');

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    const result = await res.json();

    if (result.error) {
      alert('Something went wrong: ' + result.error);
      scanningState.classList.remove('active');
      analyzeBtn.disabled = false;
      return;
    }

    // small delay so the scanning animation feels intentional, not instant
    setTimeout(() => showResult(result), 500);

  } catch (err) {
    alert('Could not reach the prediction server. Is app.py running?');
    scanningState.classList.remove('active');
    analyzeBtn.disabled = false;
  }
});

function showResult(result) {
  scanningState.classList.remove('active');
  resultContent.classList.add('active');
  analyzeBtn.disabled = false;

  const badge = document.getElementById('resultBadge');
  const title = document.getElementById('resultTitle');
  const confidence = document.getElementById('resultConfidence');
  const probList = document.getElementById('probList');

  if (result.is_tumor) {
    badge.className = 'result-badge tumor';
    badge.textContent = '⚠';
    title.textContent = `${result.prediction} Detected`;
  } else {
    badge.className = 'result-badge clear';
    badge.textContent = '✓';
    title.textContent = 'No Tumor Detected';
  }
  confidence.textContent = `Model confidence: ${result.confidence}%`;

  probList.innerHTML = result.all_probabilities.map((p, i) => `
    <div class="prob-row ${i === 0 ? 'top-pred' : ''}">
      <div class="prob-row-top">
        <span class="prob-row-label">${p.label}</span>
        <span class="prob-row-value">${p.probability}%</span>
      </div>
      <div class="prob-bar-track">
        <div class="prob-bar-fill" style="width:0%" data-target="${p.probability}"></div>
      </div>
    </div>
  `).join('');

  // animate bars after paint
  requestAnimationFrame(() => {
    document.querySelectorAll('.prob-bar-fill').forEach(bar => {
      bar.style.width = bar.dataset.target + '%';
    });
  });
}

resetBtn.addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  dropzone.classList.remove('has-image');
  previewImg.src = '';
  analyzeBtn.disabled = true;
  resultContent.classList.remove('active');
  document.getElementById('analyze').scrollIntoView({ behavior: 'smooth' });
});
