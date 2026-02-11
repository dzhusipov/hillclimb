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

// --- Snapshot polling (avoids browser 6-connection limit for MJPEG) ---

const SNAPSHOT_INTERVAL = 800;  // ms between snapshot refreshes

function refreshSnapshots() {
    document.querySelectorAll('.stream').forEach(img => {
        const emuId = img.dataset.emuId;
        if (emuId === undefined) return;
        // Only refresh if previous load finished (avoid piling up requests)
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

// --- Init ---

updateStatus();
setInterval(updateStatus, POLL_INTERVAL);
refreshSnapshots();
setInterval(refreshSnapshots, SNAPSHOT_INTERVAL);
setupTouchHandlers();
