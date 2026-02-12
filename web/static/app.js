// HCR2 Dashboard — status polling, control & touch input

const POLL_INTERVAL = 3000;

// --- Status polling ---

function updateBadge(id, dockerStatus) {
    const badge = document.getElementById(`badge-${id}`);
    if (!badge) return;
    badge.textContent = dockerStatus;
    badge.className = 'badge';
    if (dockerStatus === 'running') {
        badge.classList.add('badge-running');
    } else if (dockerStatus === 'exited' || dockerStatus === 'stopped') {
        badge.classList.add('badge-exited');
    } else {
        badge.classList.add('badge-not-found');
    }
}

function updateStatus() {
    fetch('/api/status')
        .then(r => r.json())
        .then(emulators => {
            const countEl = document.getElementById('emu-count');
            const running = emulators.filter(e => e.docker_status === 'running').length;
            countEl.textContent = `${running}/${emulators.length} запущено`;

            emulators.forEach(emu => {
                updateBadge(emu.id, emu.docker_status);
                const dockerEl = document.getElementById(`docker-${emu.id}`);
                const bootEl = document.getElementById(`boot-${emu.id}`);
                if (dockerEl) dockerEl.textContent = emu.docker_status;
                if (bootEl) bootEl.textContent = emu.boot_status;
            });
        })
        .catch(err => console.error('Status poll failed:', err));
}

function showToast(message, success) {
    const toast = document.createElement('div');
    toast.className = `toast ${success ? 'success' : 'error'}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

function action(id, act) {
    const btn = event.target;
    btn.disabled = true;

    fetch(`/api/emulator/${id}/${act}`, { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            showToast(data.message, data.ok);
            setTimeout(updateStatus, 1000);
        })
        .catch(err => {
            showToast(`Ошибка: ${err}`, false);
        })
        .finally(() => {
            btn.disabled = false;
        });
}

// --- Touch input ---

function getNormCoords(img, clientX, clientY) {
    const rect = img.getBoundingClientRect();

    // Account for object-fit: contain (image may not fill the whole element)
    const natW = img.naturalWidth || rect.width;
    const natH = img.naturalHeight || rect.height;
    const imgAspect = natW / natH;
    const elemAspect = rect.width / rect.height;

    let displayW, displayH, offsetX, offsetY;
    if (imgAspect > elemAspect) {
        displayW = rect.width;
        displayH = rect.width / imgAspect;
        offsetX = 0;
        offsetY = (rect.height - displayH) / 2;
    } else {
        displayH = rect.height;
        displayW = rect.height * imgAspect;
        offsetX = (rect.width - displayW) / 2;
        offsetY = 0;
    }

    const x = (clientX - rect.left - offsetX) / displayW;
    const y = (clientY - rect.top - offsetY) / displayH;
    return {
        x: Math.max(0, Math.min(1, x)),
        y: Math.max(0, Math.min(1, y))
    };
}

function sendTouch(emuId, action, x, y) {
    fetch(`/api/emulator/${emuId}/touch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, x: x || 0, y: y || 0 })
    }).catch(err => console.error('Touch failed:', err));
}

function setupTouchHandlers() {
    document.querySelectorAll('.stream').forEach(img => {
        const emuId = img.closest('.card').dataset.id;

        // Mouse events
        img.addEventListener('mousedown', e => {
            e.preventDefault();
            const coords = getNormCoords(img, e.clientX, e.clientY);
            sendTouch(emuId, 'down', coords.x, coords.y);
        });

        img.addEventListener('mouseup', e => {
            e.preventDefault();
            sendTouch(emuId, 'up');
        });

        img.addEventListener('mouseleave', e => {
            // Release touch if cursor leaves the stream
            sendTouch(emuId, 'up');
        });

        // Touch events (mobile)
        img.addEventListener('touchstart', e => {
            e.preventDefault();
            const touch = e.touches[0];
            const coords = getNormCoords(img, touch.clientX, touch.clientY);
            sendTouch(emuId, 'down', coords.x, coords.y);
        });

        img.addEventListener('touchend', e => {
            e.preventDefault();
            sendTouch(emuId, 'up');
        });

        img.addEventListener('touchcancel', e => {
            sendTouch(emuId, 'up');
        });
    });
}

// --- WebSocket streaming (scrcpy → JPEG push, ~30 FPS) ---

function startWebSocketStream(img, emuId) {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/ws/stream/${emuId}`;
    const ws = new WebSocket(url);
    ws.binaryType = 'blob';

    let prevUrl = null;

    ws.onmessage = (event) => {
        const blobUrl = URL.createObjectURL(event.data);
        img.onload = () => {
            if (prevUrl) URL.revokeObjectURL(prevUrl);
            prevUrl = blobUrl;
        };
        img.src = blobUrl;
        img.style.display = 'block';
        const placeholder = img.nextElementSibling;
        if (placeholder) placeholder.style.display = 'none';
    };

    ws.onclose = () => {
        // Reconnect after 2s
        setTimeout(() => startWebSocketStream(img, emuId), 2000);
    };

    ws.onerror = () => ws.close();
}

// --- Snapshot polling (fallback when WebSocket not available) ---

const SNAPSHOT_INTERVAL = 800;

function refreshSnapshots() {
    document.querySelectorAll('.stream[data-fallback="true"]').forEach(img => {
        const emuId = img.dataset.emuId;
        if (emuId === undefined) return;
        if (img.dataset.loading === 'true') return;
        img.dataset.loading = 'true';
        const newImg = new Image();
        newImg.onload = () => {
            img.src = newImg.src;
            img.style.display = 'block';
            const placeholder = img.nextElementSibling;
            if (placeholder) placeholder.style.display = 'none';
            img.dataset.loading = 'false';
        };
        newImg.onerror = () => {
            img.dataset.loading = 'false';
        };
        newImg.src = `/snapshot/${emuId}?t=${Date.now()}`;
    });
}

function initStreams() {
    document.querySelectorAll('.stream').forEach(img => {
        const emuId = img.dataset.emuId;
        if (emuId === undefined) return;
        startWebSocketStream(img, emuId);
    });
}

// --- Training metrics polling ---

const TRAINING_POLL = 5000;  // 5s — enough for dashboard view
let distanceChart = null;

function updateTrainingStatus() {
    fetch('/api/training/status')
        .then(r => r.json())
        .then(data => {
            const badge = document.getElementById('training-status');
            if (!data.training_active && !data.current_episode) {
                badge.textContent = 'No data';
                badge.className = 'training-badge inactive';
                return;
            }
            if (!data.training_active) {
                badge.textContent = 'Stopped';
                badge.className = 'training-badge inactive';
            } else if (data.stale) {
                badge.textContent = 'Stale';
                badge.className = 'training-badge stale';
            } else {
                badge.textContent = 'Active';
                badge.className = 'training-badge active';
            }

            const fmt = n => n != null ? Number(n).toLocaleString() : '-';
            document.getElementById('stat-episode').textContent = fmt(data.current_episode);
            document.getElementById('stat-timesteps').textContent = fmt(data.total_timesteps);
            document.getElementById('stat-best').textContent = data.best_distance != null
                ? Math.round(data.best_distance) + 'm' : '-';
            document.getElementById('stat-avg').textContent = data.avg_distance_10 != null
                ? Math.round(data.avg_distance_10) + 'm' : '-';
            document.getElementById('stat-eph').textContent = fmt(data.episodes_per_hour);
        })
        .catch(() => {});
}

function updateChart() {
    fetch('/api/training/episodes?limit=200')
        .then(r => r.json())
        .then(data => {
            const eps = data.episodes;
            if (!eps || eps.length === 0) return;

            const labels = eps.map(e => e.episode);
            const distances = eps.map(e => e.distance);
            const rewards = eps.map(e => e.reward);

            const ctx = document.getElementById('distance-chart');
            if (!ctx) return;

            if (distanceChart) {
                distanceChart.data.labels = labels;
                distanceChart.data.datasets[0].data = distances;
                distanceChart.data.datasets[1].data = rewards;
                distanceChart.update('none');
            } else {
                distanceChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels,
                        datasets: [
                            {
                                label: 'Distance (m)',
                                data: distances,
                                borderColor: '#3b82f6',
                                backgroundColor: 'rgba(59,130,246,0.1)',
                                borderWidth: 1.5,
                                pointRadius: 0,
                                tension: 0.3,
                                fill: true,
                                yAxisID: 'y',
                            },
                            {
                                label: 'Reward',
                                data: rewards,
                                borderColor: '#f59e0b',
                                borderWidth: 1,
                                pointRadius: 0,
                                tension: 0.3,
                                yAxisID: 'y1',
                            },
                        ],
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        animation: false,
                        plugins: {
                            legend: { labels: { color: '#aaa', boxWidth: 12, font: { size: 11 } } },
                        },
                        scales: {
                            x: { display: false },
                            y: {
                                beginAtZero: true,
                                grid: { color: 'rgba(255,255,255,0.06)' },
                                ticks: { color: '#888', font: { size: 10 } },
                            },
                            y1: {
                                position: 'right',
                                grid: { drawOnChartArea: false },
                                ticks: { color: '#888', font: { size: 10 } },
                            },
                        },
                    },
                });
            }
        })
        .catch(() => {});
}

function updateEvents() {
    fetch('/api/training/events?limit=15')
        .then(r => r.json())
        .then(data => {
            const el = document.getElementById('events-list');
            if (!data.events || data.events.length === 0) {
                el.innerHTML = '<span style="color:#555">No events</span>';
                return;
            }
            el.innerHTML = data.events.slice().reverse().map(ev => {
                const t = new Date(ev.timestamp * 1000).toLocaleTimeString();
                const cls = ev.event || '';
                const detail = ev.details ? ` (${JSON.stringify(ev.details)})` : '';
                return `<div class="ev ${cls}">[${t}] env${ev.env_idx}: ${ev.state_before} → ${ev.state_after || '?'} | ${ev.action}${detail}</div>`;
            }).join('');
        })
        .catch(() => {});
}

function initTraining() {
    updateTrainingStatus();
    updateChart();
    updateEvents();
    setInterval(updateTrainingStatus, TRAINING_POLL);
    setInterval(updateChart, TRAINING_POLL);
    setInterval(updateEvents, TRAINING_POLL);
}

// --- Init ---

updateStatus();
setInterval(updateStatus, POLL_INTERVAL);
initStreams();
setInterval(refreshSnapshots, SNAPSHOT_INTERVAL);
setupTouchHandlers();
initTraining();
