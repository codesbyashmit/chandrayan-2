let sourceFile = null;
let referenceFile = null;
let currentResults = null;
let matchViz = null;
let distViz = null;
const API_BASE = '';
document.addEventListener('DOMContentLoaded', () => {
    initUploadZones();
    initSliders();
    initParticles();
    checkGPU();
    matchViz = new MatchVisualizer('match-canvas');
    distViz = new DistributionVisualizer('distribution-canvas');
});
async function checkGPU() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();
        const el = document.getElementById('gpu-status');
        const textEl = el.querySelector('.status-text');

        if (data.gpu_available) {
            el.classList.add('active');
            textEl.textContent = data.gpu_name;
        } else {
            textEl.textContent = 'CPU Mode';
        }
    } catch (e) {
        const el = document.getElementById('gpu-status');
        el.querySelector('.status-text').textContent = 'Server offline';
    }
}
function initUploadZones() {
    setupDropZone('source');
    setupDropZone('reference');
}
function setupDropZone(type) {
    const zone = document.getElementById(`${type}-drop-zone`);
    const input = document.getElementById(`${type}-input`);
    zone.addEventListener('click', () => input.click());
    input.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(type, e.target.files[0]);
        }
    });
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('drag-over');
    });
    zone.addEventListener('dragleave', () => {
        zone.classList.remove('drag-over');
    });
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFile(type, e.dataTransfer.files[0]);
        }
    });
}
function handleFile(type, file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }
    if (type === 'source') {
        sourceFile = file;
    } else {
        referenceFile = file;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
        const zone = document.getElementById(`${type}-drop-zone`);
        const preview = document.getElementById(`${type}-preview`);
        const img = document.getElementById(`${type}-preview-img`);
        const info = document.getElementById(`${type}-info`);
        zone.style.display = 'none';
        preview.style.display = 'block';
        img.src = e.target.result;
        const tempImg = new Image();
        tempImg.onload = () => {
            info.textContent = `${tempImg.width}×${tempImg.height} | ${(file.size / 1024).toFixed(1)} KB`;
        };
        tempImg.src = e.target.result;
    };
    reader.readAsDataURL(file);
    updateRunButton();
}
function removeImage(type) {
    if (type === 'source') {
        sourceFile = null;
    } else {
        referenceFile = null;
    }
    const zone = document.getElementById(`${type}-drop-zone`);
    const preview = document.getElementById(`${type}-preview`);
    const input = document.getElementById(`${type}-input`);
    zone.style.display = 'flex';
    preview.style.display = 'none';
    input.value = '';
    updateRunButton();
}
function updateRunButton() {
    const btn = document.getElementById('btn-run');
    btn.disabled = !(sourceFile && referenceFile);
}
function initSliders() {
    const keypointsSlider = document.getElementById('config-keypoints');
    const ransacSlider = document.getElementById('config-ransac');
    keypointsSlider.addEventListener('input', () => {
        document.getElementById('keypoints-value').textContent = keypointsSlider.value;
    });
    ransacSlider.addEventListener('input', () => {
        document.getElementById('ransac-value').textContent =
            parseFloat(ransacSlider.value).toFixed(1);
    });
}
function buildConfig() {
    return {
        preprocessing: {
            method: document.getElementById('config-preprocessing').value,
        },
        feature_detection: {
            method: document.getElementById('config-detector').value,
            max_keypoints: parseInt(document.getElementById('config-keypoints').value),
        },
        matching: {
            method: document.getElementById('config-matcher').value,
        },
        geometric: {
            model: document.getElementById('config-geometric').value,
            ransac_threshold: parseFloat(document.getElementById('config-ransac').value),
        },
        subpixel: {
            enabled: document.getElementById('config-subpixel').checked,
        },
        uniform_distribution: {
            enabled: document.getElementById('config-uniform').checked,
        },
    };
}




async function runPipeline() {
    if (!sourceFile || !referenceFile) return;

    const btn = document.getElementById('btn-run');
    const progressSection = document.getElementById('progress-section');
    const resultsSection = document.getElementById('results-section');


    btn.disabled = true;
    btn.querySelector('.btn-run-text').textContent = 'Processing...';
    progressSection.style.display = 'block';
    resultsSection.style.display = 'none';


    const progressBar = document.getElementById('progress-bar');
    const progressTitle = document.getElementById('progress-title');
    const progressDetail = document.getElementById('progress-detail');

    const stages = [{
        pct: 10,
        text: 'Uploading images...'
    }, {
        pct: 20,
        text: 'Preprocessing (illumination normalization)...'
    }, {
        pct: 40,
        text: 'Detecting features (keypoints + descriptors)...'
    }, {
        pct: 55,
        text: 'Matching features between images...'
    }, {
        pct: 70,
        text: 'Estimating geometric transformation (RANSAC)...'
    }, {
        pct: 80,
        text: 'Sub-pixel refinement...'
    }, {
        pct: 90,
        text: 'Evaluating registration quality...'
    }, ];

    let stageIdx = 0;
    const progressInterval = setInterval(() => {
        if (stageIdx < stages.length) {
            progressBar.style.width = stages[stageIdx].pct + '%';
            progressDetail.textContent = stages[stageIdx].text;
            stageIdx++;
        }
    }, 800);

    try {

        const formData = new FormData();
        formData.append('source', sourceFile);
        formData.append('reference', referenceFile);
        formData.append('config', JSON.stringify(buildConfig()));


        const response = await fetch(`${API_BASE}/api/register`, {
            method: 'POST',
            body: formData,
        });

        clearInterval(progressInterval);
        progressBar.style.width = '100%';
        progressDetail.textContent = 'Complete!';

        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        const data = await response.json();

        if (data.error) {
            progressTitle.textContent = 'Error';
            progressDetail.textContent = data.error;
            btn.disabled = false;
            btn.querySelector('.btn-run-text').textContent = 'Run Registration Pipeline';
            return;
        }

        currentResults = data;


        await new Promise(r => setTimeout(r, 500));


        progressSection.style.display = 'none';
        resultsSection.style.display = 'block';


        renderMetrics(data);
        renderVisualizations(data);
        renderDetailedMetrics(data);
        renderTimingBars(data);
        renderTransformMatrix(data);

    } catch (error) {
        clearInterval(progressInterval);
        progressTitle.textContent = 'Error';
        progressDetail.textContent = error.message;
        console.error('Pipeline error:', error);
    } finally {
        btn.disabled = false;
        btn.querySelector('.btn-run-text').textContent = 'Run Registration Pipeline';
    }
}




function renderMetrics(data) {
    const metrics = data.metrics;
    const dist = data.distribution_stats || {};

    const updates = {
        'rmse': metrics.rmse !== undefined ? metrics.rmse.toFixed(3) : '—',
        'inlier_ratio': metrics.inlier_ratio !== undefined ?
            (metrics.inlier_ratio * 100).toFixed(1) + '%' : '—',
        'final_match_count': data.final_match_count || '—',
        'ncc': metrics.ncc !== undefined ? metrics.ncc.toFixed(4) : '—',
        'ssim': metrics.ssim !== undefined ? metrics.ssim.toFixed(4) : '—',
        'coverage': dist.coverage !== undefined ?
            (dist.coverage * 100).toFixed(1) + '%' : '—',
        'sub_pixel_percentage': metrics.sub_pixel_percentage !== undefined ?
            metrics.sub_pixel_percentage.toFixed(1) + '%' : '—',
        'total_time': data.timings?.total !== undefined ?
            data.timings.total.toFixed(2) + 's' : '—',
    };


    document.querySelectorAll('.metric-value').forEach(el => {
        const key = el.getAttribute('data-metric');
        if (updates[key]) {
            animateValue(el, updates[key]);
        }
    });


    setTimeout(() => {
        const fillPcts = {
            'metric-rmse': Math.max(0, 100 - (metrics.rmse || 0) * 20),
            'metric-inlier-ratio': (metrics.inlier_ratio || 0) * 100,
            'metric-matches': Math.min(100, (data.final_match_count || 0) / 10),
            'metric-ncc': (metrics.ncc || 0) * 100,
            'metric-ssim': (metrics.ssim || 0) * 100,
            'metric-coverage': (dist.coverage || 0) * 100,
            'metric-subpx': metrics.sub_pixel_percentage || 0,
            'metric-time': Math.max(0, 100 - (data.timings?.total || 0) * 10),
        };

        for (const [id, pct] of Object.entries(fillPcts)) {
            const el = document.querySelector(`#${id} .metric-fill`);
            if (el) el.style.width = Math.min(100, Math.max(0, pct)) + '%';
        }
    }, 300);
}

function animateValue(el, targetText) {
    el.style.opacity = '0';
    el.style.transform = 'translateY(10px)';
    setTimeout(() => {
        el.textContent = targetText;
        el.style.transition = 'all 0.5s ease';
        el.style.opacity = '1';
        el.style.transform = 'translateY(0)';
    }, 200);
}




async function renderVisualizations(data) {
    const images = data.images;


    if (images.warped) {
        document.getElementById('registered-image').src =
            'data:image/png;base64,' + images.warped;
    }
    if (images.checkerboard) {
        document.getElementById('checkerboard-image').src =
            'data:image/png;base64,' + images.checkerboard;
    }
    if (images.difference) {
        document.getElementById('difference-image').src =
            'data:image/png;base64,' + images.difference;
    }
    if (images.source_preprocessed) {
        document.getElementById('source-preprocessed-img').src =
            'data:image/png;base64,' + images.source_preprocessed;
    }
    if (images.reference_preprocessed) {
        document.getElementById('reference-preprocessed-img').src =
            'data:image/png;base64,' + images.reference_preprocessed;
    }


    const srcDataUrl = document.getElementById('source-preview-img').src;
    const refDataUrl = document.getElementById('reference-preview-img').src;

    await matchViz.loadImages(srcDataUrl, refDataUrl);
    matchViz.setMatchData(data);
    matchViz.draw();


    if (data.distribution_stats && data.distribution_stats.grid_counts) {
        distViz.draw(data.distribution_stats.grid_counts, data.distribution_stats);
    }
}




function switchTab(tabName) {

    document.querySelectorAll('.viz-tab').forEach(t => t.classList.remove('active'));
    document.querySelector(`.viz-tab[data-tab="${tabName}"]`).classList.add('active');


    document.querySelectorAll('.viz-panel').forEach(p => p.classList.remove('active'));
    document.getElementById(`panel-${tabName}`).classList.add('active');


    if (tabName === 'matches' && matchViz) {
        setTimeout(() => matchViz.draw(), 100);
    }
    if (tabName === 'distribution' && distViz && currentResults) {
        setTimeout(() => {
            const stats = currentResults.distribution_stats;
            if (stats && stats.grid_counts) {
                distViz.draw(stats.grid_counts, stats);
            }
        }, 100);
    }
}




function updateMatchViz() {
    if (!matchViz) return;
    matchViz.showInliers = document.getElementById('toggle-inliers').checked;
    matchViz.showOutliers = document.getElementById('toggle-outliers').checked;
    matchViz.draw();
}




function renderDetailedMetrics(data) {
    const metrics = data.metrics;
    const tbody = document.getElementById('metrics-tbody');
    tbody.innerHTML = '';

    const rows = [
        ['RMSE', metrics.rmse?.toFixed(4) + ' px', 'Root Mean Square Error of reprojection residuals'],
        ['MAE', metrics.mae?.toFixed(4) + ' px', 'Mean Absolute Error of reprojection'],
        ['Median Error', metrics.median_error?.toFixed(4) + ' px', 'Median reprojection error'],
        ['Max Error', metrics.max_error?.toFixed(4) + ' px', 'Maximum reprojection error'],
        ['Std Error', metrics.std_error?.toFixed(4) + ' px', 'Standard deviation of errors'],
        ['—', '', ''],
        ['Total Matches', metrics.total_matches, 'Raw feature matches before filtering'],
        ['Inlier Count', metrics.inlier_count, 'Geometrically consistent matches'],
        ['Outlier Count', metrics.outlier_count, 'Rejected matches (outliers)'],
        ['Inlier Ratio', (metrics.inlier_ratio * 100)?.toFixed(2) + '%', 'Inliers / Total matches'],
        ['Final Matches', metrics.final_match_count, 'After uniform distribution enforcement'],
        ['—', '', ''],
        ['NCC', metrics.ncc?.toFixed(4), 'Normalized Cross-Correlation (1.0 = perfect)'],
        ['SSIM', metrics.ssim?.toFixed(4), 'Structural Similarity Index (1.0 = identical)'],
        ['Sub-pixel %', metrics.sub_pixel_percentage?.toFixed(1) + '%', 'Matches with < 1px reprojection error'],
        ['Sub-px Mean Shift', metrics.subpixel_mean_shift?.toFixed(4) + ' px', 'Mean sub-pixel refinement magnitude'],
    ];

    rows.forEach(([name, value, desc]) => {
        if (name === '—') {
            const tr = document.createElement('tr');
            tr.innerHTML = `<td colspan="3" style="padding:2px;border:none;"></td>`;
            tbody.appendChild(tr);
            return;
        }
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${name}</td><td>${value ?? '—'}</td><td style="color:var(--text-muted);font-size:0.75rem;">${desc}</td>`;
        tbody.appendChild(tr);
    });
}




function renderTimingBars(data) {
    const timings = data.timings || {};
    const container = document.getElementById('timing-bars');
    container.innerHTML = '';

    const total = timings.total || 1;
    const colors = [
        'var(--gradient-primary)',
        'var(--gradient-accent)',
        'var(--gradient-warm)',
        'var(--gradient-success)',
        'var(--gradient-primary)',
        'var(--gradient-accent)',
        'var(--gradient-warm)',
        'var(--gradient-success)',
    ];

    const stages = ['load', 'preprocessing', 'detection', 'matching',
        'geometric', 'subpixel', 'distribution', 'warping', 'evaluation'
    ];

    stages.forEach((stage, idx) => {
        if (timings[stage] === undefined) return;

        const pct = (timings[stage] / total) * 100;
        const row = document.createElement('div');
        row.className = 'timing-row';
        row.innerHTML = `
            <span class="timing-label">${stage}</span>
            <div class="timing-bar-bg">
                <div class="timing-bar-fill" style="width:0%;background:${colors[idx % colors.length]}"></div>
            </div>
            <span class="timing-bar-value">${(timings[stage] * 1000).toFixed(0)}ms</span>
        `;
        container.appendChild(row);


        setTimeout(() => {
            row.querySelector('.timing-bar-fill').style.width = Math.max(3, pct) + '%';
        }, 100 + idx * 100);
    });
}




function renderTransformMatrix(data) {
    const display = document.getElementById('transform-display');
    if (!data.transform_matrix) {
        display.textContent = 'No transformation computed';
        return;
    }

    const M = data.transform_matrix;
    const lines = [
        'Estimated Transformation Matrix (3×3):',
        '',
        `┌ ${M[0][0].toFixed(8)}   ${M[0][1].toFixed(8)}   ${M[0][2].toFixed(8)} ┐`,
        `│ ${M[1][0].toFixed(8)}   ${M[1][1].toFixed(8)}   ${M[1][2].toFixed(8)} │`,
        `└ ${M[2][0].toFixed(8)}   ${M[2][1].toFixed(8)}   ${M[2][2].toFixed(8)} ┘`,
        '',
        `Model: ${data.config?.geometric?.model || 'homography'}`,
        `RANSAC Threshold: ${data.config?.geometric?.ransac_threshold || 5.0} px`,
    ];

    display.textContent = lines.join('\n');
}

function initParticles() {
    const canvas = document.getElementById('particles-canvas');
    const ctx = canvas.getContext('2d');
    let particles = [];
    let animId;

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    function createParticles() {
        particles = [];
        const count = Math.floor(window.innerWidth * window.innerHeight / 15000);
        for (let i = 0; i < count; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                r: Math.random() * 1.5 + 0.3,
                vx: (Math.random() - 0.5) * 0.15,
                vy: (Math.random() - 0.5) * 0.15,
                alpha: Math.random() * 0.5 + 0.1,
            });
        }
    }

    function drawParticles() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        particles.forEach(p => {
            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(148, 163, 184, ${p.alpha})`;
            ctx.fill();
        });
        animId = requestAnimationFrame(drawParticles);
    }
    resize();
    createParticles();
    drawParticles();
    window.addEventListener('resize', () => {
        resize();
        createParticles();
    });
}