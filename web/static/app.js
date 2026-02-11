// HCR2 Dashboard — status polling & control

const POLL_INTERVAL = 3000;

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
            // Refresh status after action
            setTimeout(updateStatus, 1000);
        })
        .catch(err => {
            showToast(`Ошибка: ${err}`, false);
        })
        .finally(() => {
            btn.disabled = false;
        });
}

// Initial status update & polling
updateStatus();
setInterval(updateStatus, POLL_INTERVAL);
