// HCR2 Dashboard — status polling & control

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

// --- Snapshot polling ---

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
        // Snapshot polling mode (low CPU, no scrcpy needed)
        img.dataset.fallback = 'true';
    });
}

// --- Training metrics polling ---

const TRAINING_POLL = 30000;  // 30s — low CPU overhead
let distanceChart = null;

// --- Color helpers ---
const C = {
    red:    '#ef4444',
    amber:  '#f59e0b',
    green:  '#22c55e',
    blue:   '#6366f1',
    cyan:   '#06b6d4',
    purple: '#a78bfa',
    muted:  '#52525b',
    white:  '#fafafa',
};

function lerpColor(a, b, t) {
    const pa = [parseInt(a.slice(1,3),16), parseInt(a.slice(3,5),16), parseInt(a.slice(5,7),16)];
    const pb = [parseInt(b.slice(1,3),16), parseInt(b.slice(3,5),16), parseInt(b.slice(5,7),16)];
    const r = pa.map((v, i) => Math.round(v + (pb[i] - v) * t));
    return `rgb(${r[0]},${r[1]},${r[2]})`;
}

function colorByPct(pct) {
    // 0% → red, 50% → amber, 100% → green (smooth gradient)
    pct = Math.max(0, Math.min(100, pct));
    if (pct < 50) return lerpColor(C.red, C.amber, pct / 50);
    return lerpColor(C.amber, C.green, (pct - 50) / 50);
}

function colorByThresholds(val, lo, hi) {
    // val < lo → red, lo..hi → amber, > hi → green
    if (val < lo) return C.red;
    if (val < hi) return C.amber;
    return C.green;
}

function setStat(id, text, color) {
    const el = document.getElementById(id);
    if (!el) return;
    el.textContent = text;
    el.style.color = color || C.white;
}

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

            // Episode — accent blue
            setStat('stat-episode', fmt(data.current_episode), C.blue);

            // Timesteps — progress % toward target
            const tsPct = (data.total_target > 0)
                ? data.total_timesteps / data.total_target * 100 : 0;
            setStat('stat-timesteps', fmt(data.total_timesteps), colorByPct(tsPct));

            // Best distance — green intensity by distance
            if (data.best_distance != null) {
                const bestPct = Math.min(data.best_distance / 500 * 100, 100);
                setStat('stat-best', Math.round(data.best_distance) + 'm',
                    colorByPct(bestPct));
            } else {
                setStat('stat-best', '-', C.muted);
            }

            // Avg(10) — ratio to best distance
            if (data.avg_distance_10 != null && data.best_distance > 0) {
                const avgRatio = data.avg_distance_10 / data.best_distance * 100;
                setStat('stat-avg', Math.round(data.avg_distance_10) + 'm',
                    colorByPct(avgRatio));
            } else {
                setStat('stat-avg', data.avg_distance_10 != null
                    ? Math.round(data.avg_distance_10) + 'm' : '-', C.muted);
            }

            // Eps/hour — throughput
            if (data.episodes_per_hour != null) {
                setStat('stat-eph', fmt(data.episodes_per_hour),
                    colorByThresholds(data.episodes_per_hour, 50, 150));
            } else {
                setStat('stat-eph', '-', C.muted);
            }

            // Steps/min — throughput
            if (data.steps_per_min != null) {
                setStat('stat-spm', fmt(data.steps_per_min),
                    colorByThresholds(data.steps_per_min, 300, 800));
            } else {
                setStat('stat-spm', '-', C.muted);
            }

            // Uptime — cyan (informational)
            if (data.uptime_s != null) {
                const h = Math.floor(data.uptime_s / 3600);
                const m = Math.floor((data.uptime_s % 3600) / 60);
                setStat('stat-uptime', `${h}h ${m}m`, C.cyan);
            } else {
                setStat('stat-uptime', '-', C.muted);
            }

            // Reward(10) — negative=red, 0-30=amber, >30=green
            if (data.avg_reward_10 != null) {
                const rw = data.avg_reward_10;
                const rwColor = rw < 0 ? C.red
                    : colorByPct(Math.min(rw / 50 * 100, 100));
                setStat('stat-reward', Number(rw).toFixed(1), rwColor);
            } else {
                setStat('stat-reward', '-', C.muted);
            }

            // MemReader — percentage based
            if (data.mem_reader_total != null && data.mem_reader_total > 0) {
                const pct = Math.round(data.mem_reader_ok / data.mem_reader_total * 100);
                setStat('stat-memreader',
                    `${data.mem_reader_ok}/${data.mem_reader_total} (${pct}%)`,
                    colorByPct(pct));
            } else {
                setStat('stat-memreader', '-', C.muted);
            }
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
                                borderColor: '#6366f1',
                                backgroundColor: 'rgba(99,102,241,0.08)',
                                borderWidth: 1.5,
                                pointRadius: 0,
                                tension: 0.3,
                                fill: true,
                                yAxisID: 'y',
                            },
                            {
                                label: 'Reward',
                                data: rewards,
                                borderColor: '#22c55e',
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
                            legend: { labels: { color: '#a1a1aa', boxWidth: 12, font: { size: 11, family: 'Inter' } } },
                        },
                        scales: {
                            x: { display: false },
                            y: {
                                beginAtZero: true,
                                grid: { color: 'rgba(255,255,255,0.04)' },
                                ticks: { color: '#52525b', font: { size: 10, family: 'Inter' } },
                            },
                            y1: {
                                position: 'right',
                                grid: { drawOnChartArea: false },
                                ticks: { color: '#52525b', font: { size: 10, family: 'Inter' } },
                            },
                        },
                    },
                });
            }
        })
        .catch(() => {});
}

function initTraining() {
    updateTrainingStatus();
    updateChart();
    setInterval(updateTrainingStatus, TRAINING_POLL);
    setInterval(updateChart, TRAINING_POLL);
}

// --- Init ---

updateStatus();
setInterval(updateStatus, POLL_INTERVAL);
initStreams();
setInterval(refreshSnapshots, SNAPSHOT_INTERVAL);
initTraining();
