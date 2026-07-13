import React from 'react';
import '../App.css';

function SummaryCards({ data, metrics }) {
  const actuals = data?.historical_actuals || [];
  const total = actuals.reduce((a, b) => a + b, 0);
  const avg = actuals.length ? (total / actuals.length).toFixed(1) : '—';
  const modelCount = data?.forecasts ? Object.keys(data.forecasts).length : 0;
  const lastVal = actuals.length ? actuals[actuals.length - 1] : '—';

  let bestModel = null;
  if (metrics) {
    bestModel = Object.entries(metrics).reduce((best, [name, m]) =>
      m.MAPE != null && (!best || m.MAPE < best.mape) ? { name, mape: m.MAPE } : best, null);
  }

  const cards = [
    { label: 'Total Sales', value: total.toLocaleString(), color: '#667eea' },
    { label: 'Avg Daily Sales', value: avg, color: '#764ba2' },
    { label: 'Last Sale', value: lastVal, color: '#f093fb' },
    { label: 'Best Model', value: bestModel ? `${bestModel.name} (${bestModel.mape.toFixed(1)}% MAPE)` : '—', color: '#4ecdc4' },
  ];

  return (
    <div className="summary-cards">
      {cards.map(c => (
        <div key={c.label} className="stat-card" style={{ borderTop: `3px solid ${c.color}` }}>
          <div className="stat-label">{c.label}</div>
          <div className="stat-value">{c.value}</div>
        </div>
      ))}
    </div>
  );
}

export default SummaryCards;
