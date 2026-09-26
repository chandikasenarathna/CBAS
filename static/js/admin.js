// static/js/admin.js — Admin dashboard client-side logic

// ── Anomaly Timeline Chart (Admin Dashboard) ──────────────────────────────────
function initAnomalyTimeline(anomalyData) {
  const canvas = document.getElementById('anomaly-timeline-chart');
  if (!canvas || !anomalyData || !anomalyData.length) return;

  // Bucket anomalies into 1-hour bins
  const now = Date.now() / 1000;
  const bins = {};
  const typeColors = {
    warning:      { bg: 'rgba(245,158,11,0.6)', border: '#f59e0b' },
    mfa_trigger:  { bg: 'rgba(239,68,68,0.6)',  border: '#ef4444' },
    session_lock: { bg: 'rgba(168,85,247,0.6)', border: '#a855f7' },
    mfa_fail:     { bg: 'rgba(239,68,68,0.3)',  border: '#ef4444' },
    mfa_pass:     { bg: 'rgba(34,197,94,0.6)',  border: '#22c55e' },
  };

  for (let i = 23; i >= 0; i--) {
    const h = Math.floor((now - i * 3600) / 3600) * 3600;
    bins[h] = { warning: 0, mfa_trigger: 0, session_lock: 0, mfa_fail: 0, mfa_pass: 0 };
  }

  anomalyData.forEach(ev => {
    const h = Math.floor(ev.timestamp / 3600) * 3600;
    if (bins[h] && bins[h][ev.event_type] !== undefined) {
      bins[h][ev.event_type]++;
    }
  });

  const labels = Object.keys(bins).map(t =>
    new Date(parseInt(t) * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  );

  const datasets = Object.entries(typeColors).map(([type, colors]) => ({
    label: type.replace('_', ' ').toUpperCase(),
    data: Object.values(bins).map(b => b[type]),
    backgroundColor: colors.bg,
    borderColor: colors.border,
    borderWidth: 1,
    borderRadius: 3,
  }));

  new Chart(canvas.getContext('2d'), {
    type: 'bar',
    data: { labels, datasets },
    options: {
      responsive: true,
      animation: { duration: 800, easing: 'easeOutQuart' },
      scales: {
        x: { stacked: true, grid: { color: '#2a2a3a' }, ticks: { color: '#94a3b8', maxTicksLimit: 12 } },
        y: { stacked: true, grid: { color: '#2a2a3a' }, ticks: { color: '#94a3b8', stepSize: 1 },
             beginAtZero: true },
      },
      plugins: {
        legend: { labels: { color: '#e2e8f0', boxWidth: 12, padding: 16 } },
        tooltip: { mode: 'index', intersect: false },
      },
    },
  });
}


// ── User Trust Timeline Chart (User Detail Page) ─────────────────────────────
function initUserTrustTimeline(trustData) {
  const canvas = document.getElementById('trust-timeline-chart');
  if (!canvas || !trustData || !trustData.length) return;

  const labels = trustData.map(d =>
    new Date(d.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  );
  const scores = trustData.map(d => d.trust_score);
  const colors = scores.map(s => s >= 0.58 ? '#22c55e' : s >= 0.46 ? '#f59e0b' : '#ef4444');

  // Determine anomaly points (score < 0.46)
  const anomalyPoints = scores.map((s, i) => s < 0.46 ? s : null);

  new Chart(canvas.getContext('2d'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Trust Score',
          data: scores,
          borderColor: '#6366f1',
          backgroundColor: 'rgba(99,102,241,0.08)',
          borderWidth: 2,
          pointRadius: 1.5,
          pointHoverRadius: 5,
          tension: 0.3,
          fill: true,
        },
        {
          label: 'Anomaly Points',
          data: anomalyPoints,
          borderColor: 'transparent',
          backgroundColor: '#ef4444',
          pointRadius: 5,
          pointStyle: 'triangle',
          showLine: false,
        },
      ]
    },
    options: {
      responsive: true,
      animation: { duration: 1000, easing: 'easeOutQuart' },
      scales: {
        y: {
          min: 0, max: 1,
          grid: { color: '#2a2a3a' },
          ticks: { color: '#94a3b8' },
        },
        x: {
          grid: { color: 'transparent' },
          ticks: { color: '#94a3b8', maxTicksLimit: 15, maxRotation: 45 },
        },
      },
      plugins: {
        legend: { labels: { color: '#e2e8f0', boxWidth: 12 } },
        tooltip: { mode: 'index', intersect: false },
        annotation: {
          annotations: {
            warnLine: { type: 'line', yMin: 0.52, yMax: 0.52, borderColor: '#f59e0b', borderDash: [5, 5], borderWidth: 1 },
            lockLine: { type: 'line', yMin: 0.46, yMax: 0.46, borderColor: '#ef4444', borderDash: [5, 5], borderWidth: 1 },
          }
        }
      },
    },
  });
}


// ── Anomaly Doughnut Chart (User Detail Page) ─────────────────────────────────
function initAnomalyDoughnut(summaryData) {
  const canvas = document.getElementById('anomaly-doughnut');
  if (!canvas || !summaryData || !summaryData.length) return;

  const colorMap = {
    warning:      '#f59e0b',
    mfa_trigger:  '#ef4444',
    session_lock: '#a855f7',
    mfa_fail:     '#dc2626',
  };

  const labels = summaryData.map(d => d.event_type.replace('_', ' ').toUpperCase());
  const values = summaryData.map(d => d.cnt);
  const bgColors = summaryData.map(d => colorMap[d.event_type] || '#6366f1');

  new Chart(canvas.getContext('2d'), {
    type: 'doughnut',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: bgColors,
        borderColor: '#1a1a2e',
        borderWidth: 2,
        hoverOffset: 6,
      }],
    },
    options: {
      responsive: true,
      cutout: '65%',
      animation: { animateRotate: true, duration: 800 },
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#e2e8f0', padding: 12, boxWidth: 12 },
        },
      },
    },
  });
}


// ── Auto-refresh Anomaly Feed ─────────────────────────────────────────────────
function startAnomalyFeedRefresh() {
  const feedEl = document.getElementById('anomaly-feed');
  const indicator = document.getElementById('feed-live');
  if (!feedEl || !indicator) return;

  setInterval(async () => {
    try {
      indicator.style.color = '#6366f1';
      const r = await fetch('/api/admin/anomalies');
      if (!r.ok) return;
      const data = await r.json();
      if (!Array.isArray(data)) return;

      // Update the feed with new data
      feedEl.innerHTML = '';
      if (data.length === 0) {
        feedEl.innerHTML = `<div class="anomaly-empty"><div class="anomaly-empty-icon">🛡️</div><p>No anomalies detected. All systems nominal.</p></div>`;
        return;
      }

      data.forEach(ev => {
        const icon = ev.event_type === 'mfa_trigger' ? '🔐' :
                     ev.event_type === 'session_lock' ? '🔒' :
                     ev.event_type === 'warning' ? '⚠️' :
                     ev.event_type === 'mfa_fail' ? '❌' :
                     ev.event_type === 'mfa_pass' ? '✅' : '📝';

        const time = new Date(ev.timestamp * 1000).toLocaleString();
        const score = ev.trust_score ? `Score: ${ev.trust_score.toFixed(3)}` : '';

        feedEl.innerHTML += `
          <a href="/admin/users/${ev.user_id}" class="anomaly-entry anomaly-${ev.event_type}" title="View analysis for ${ev.username}">
            <div class="anomaly-icon">${icon}</div>
            <div class="anomaly-content">
              <div class="anomaly-main">
                <span class="anomaly-user">${ev.username}</span>
                <span class="anomaly-type">${ev.event_type.replace(/_/g, ' ').toUpperCase()}</span>
                <span class="anomaly-arrow">→</span>
              </div>
              <div class="anomaly-meta">
                ${score ? `<span class="anomaly-score">${score}</span>` : ''}
                <span class="anomaly-time">${time}</span>
              </div>
            </div>
          </a>`;
      });

      setTimeout(() => { indicator.style.color = '#22c55e'; }, 300);
    } catch (e) {
      console.error('Feed refresh failed:', e);
      indicator.style.color = '#ef4444';
    }
  }, 30000); // every 30 seconds
}


// ── Event Log Filter (User Detail Page) ──────────────────────────────────────
function filterEvents(filterValue) {
  const rows = document.querySelectorAll('#event-log .admin-event-row');
  const anomalyTypes = ['warning', 'mfa_trigger', 'mfa_fail', 'session_lock'];

  rows.forEach(row => {
    const type = row.dataset.type;
    if (filterValue === 'all') {
      row.style.display = '';
    } else if (filterValue === 'anomaly') {
      row.style.display = anomalyTypes.includes(type) ? '' : 'none';
    } else {
      row.style.display = type === filterValue ? '' : 'none';
    }
  });
}


// ── Animated Stat Counters ───────────────────────────────────────────────────
function animateCounters() {
  document.querySelectorAll('.stat-value').forEach(el => {
    const target = parseInt(el.textContent, 10);
    if (isNaN(target) || target === 0) return;
    let current = 0;
    const step = Math.max(1, Math.floor(target / 30));
    const interval = setInterval(() => {
      current += step;
      if (current >= target) {
        current = target;
        clearInterval(interval);
      }
      el.textContent = current;
    }, 30);
  });
}

// ── Delete Confirmation Modal ────────────────────────────────────────────────
function showDeleteModal(userId, username) {
  const modal = document.getElementById('delete-user-modal');
  const title = document.getElementById('delete-modal-title');
  const desc = document.getElementById('delete-modal-desc');
  const form = document.getElementById('delete-modal-form');

  if (!modal || !form) {
    console.error('Delete modal elements not found');
    if (confirm(`Permanently delete user "${username}"? This cannot be undone.`)) {
      window.location.href = `/admin/users/${userId}/delete?confirm=1`;
    }
    return;
  }

  if (title) title.textContent = `Delete User "${username}"?`;
  if (desc) desc.innerHTML = `Are you sure you want to permanently delete user <strong>${username}</strong> (ID: #${userId})?`;
  form.action = `/admin/users/${userId}/delete`;

  modal.classList.remove('hidden');
  modal.style.display = 'flex';
}

function hideDeleteModal() {
  const modal = document.getElementById('delete-user-modal');
  if (modal) {
    modal.classList.add('hidden');
    modal.style.display = 'none';
  }
}

// Close modal on escape key or backdrop click
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') hideDeleteModal();
});

document.addEventListener('click', (e) => {
  const modal = document.getElementById('delete-user-modal');
  if (modal && e.target === modal) {
    hideDeleteModal();
  }
});


// Run on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  animateCounters();
});

