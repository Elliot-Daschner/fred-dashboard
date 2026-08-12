let recessionData = null;

async function loadRecessionData() {
    if (!recessionData) {
        recessionData = await (await fetch('/api/recessions')).json();
    }
    return recessionData;
}

function updateSummary(dotId, summaryId, current, previous, label, mode) {
    const change = current - previous;
    const dot = document.getElementById(dotId);
    const summary = document.getElementById(summaryId);
    const direction = change > 0 ? 'Rising' : change < 0 ? 'Falling' : 'Steady';

    // mode: 'bad-when-rising' (unemployment, inflation), 'good-when-rising' (GDP, sentiment),
    // 'neutral' (FFR, treasury - shown yellow regardless of direction)
    let color;
    if (mode === 'neutral') {
        color = 'yellow';
    } else if (mode === 'bad-when-rising') {
        color = change > 0 ? 'red' : 'green';
    } else {
        color = change > 0 ? 'green' : 'red';
    }

    dot.className = `dot ${color}`;
    summary.textContent = `${label}: ${current.toFixed(2)} (${direction})`;
}

function setLastUpdated() {
    const el = document.getElementById('last-updated');
    if (el) {
        el.textContent = 'Last updated: ' + new Date().toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' });
    }
}

function formatDateLabel(dateStr) {
    const [year, month, day] = dateStr.split('-');
    return `${parseInt(month, 10)}/${parseInt(day, 10)}/${year}`;
}

async function fetchAndRender(endpoint, canvasId, statsId, label, color, onSummary) {
    const skeleton = document.getElementById(canvasId.replace('Chart', 'Skeleton'));
    const res = await fetch(endpoint);
    const rawData = await res.json();
    const reversed = rawData.filter(d => d.value !== '.').reverse();

    const card = document.getElementById(canvasId).closest('.card');
    const btnContainer = card.querySelector('.filter-buttons');
    let activeRange = 'All';
    let chartInstance = null;

    const recession = await loadRecessionData();

    function filterData(range) {
        const now = new Date();
        const cutoff = {
            '1M': new Date(now.getFullYear(), now.getMonth() - 1),
            '6M': new Date(now.getFullYear(), now.getMonth() - 6),
            '1Y': new Date(now.getFullYear() - 1, now.getMonth()),
            '5Y': new Date(now.getFullYear() - 5, now.getMonth()),
            '10Y': new Date(now.getFullYear() - 10, now.getMonth()),
            'All': new Date(0)
        }[range];

        return reversed.filter(d => new Date(d.date) >= cutoff);
    }

    function renderChart(range) {
        const filtered = filterData(range);
        const labels = filtered.map(d => d.date);
        const values = filtered.map(d => parseFloat(d.value));

        if (chartInstance) chartInstance.destroy();

        chartInstance = new Chart(document.getElementById(canvasId), {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label,
                    data: values,
                    borderColor: color,
                    tension: 0.3,
                    pointRadius: 2,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            title: (items) => {
                                return 'Date: ' + formatDateLabel(items[0].label);
                            },
                            label: (item) => {
                                return `${label}: ${item.parsed.y.toFixed(2)}`;
                            }
                        },
                        backgroundColor: '#1a2b4a',
                        titleColor: '#aaa',
                        bodyColor: '#fff',
                        padding: 10,
                        cornerRadius: 8,
                        displayColors: false
                    },
                    annotation: {
                        annotations: recession
                            .filter(r => new Date(r.start) >= (filterData(range)[0] ? new Date(filterData(range)[0].date) : new Date(0)))
                            .map((r, i) => ({
                                type: 'box',
                                xMin: r.start,
                                xMax: r.end,
                                backgroundColor: 'rgba(200, 200, 200, 0.25)',
                                borderWidth: 0,
                                label: { display: false }
                            }))
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            maxTicksLimit: 6,
                            callback: function(value) {
                                return formatDateLabel(this.getLabelForValue(value));
                            }
                        }
                    },
                    y: { grid: { color: 'rgba(0,0,0,0.04)' } }
                }
            }
        });
    }

    // Summary stats from full dataset
    const current = parseFloat(reversed[reversed.length - 1].value);
    const previous = parseFloat(reversed[reversed.length - 2].value);
    const change = (current - previous).toFixed(2);
    const isUp = change >= 0;
    document.getElementById(statsId).innerHTML = `
        <div class="stat-item">
            <span class="stat-label">Current</span>
            <span class="stat-value">${current.toFixed(2)}</span>
        </div>
        <div class="stat-item">
            <span class="stat-label">Change</span>
            <span class="stat-change ${isUp ? 'up' : 'down'}">
                ${isUp ? '▲' : '▼'} ${Math.abs(change)}
            </span>
        </div>
    `;

    if (onSummary) onSummary(current, previous);

    // Filter button logic
    btnContainer.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            btnContainer.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeRange = btn.dataset.range;
            renderChart(btn.dataset.range);
        });
    });

    // Set All as default active
    btnContainer.querySelector('[data-range="All"]').classList.add('active');
    renderChart('All');

    // Export button
    const exportBtn = document.getElementById(canvasId.replace('Chart', 'Export'));
    if (exportBtn) {
        exportBtn.addEventListener('click', () => {
            const filtered = filterData(activeRange);
            const csv = ['Date,Value', ...filtered.map(d => `${d.date},${d.value}`)].join('\n');
            const blob = new Blob([csv], { type: 'text/csv' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${label.replace(/ /g, '_')}_${activeRange}.csv`;
            a.click();
            URL.revokeObjectURL(url);
        });
    }
    if (skeleton) skeleton.style.display = 'none';
}
