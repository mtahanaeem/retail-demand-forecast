import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts';
import '../App.css';

const COLORS = { SARIMA: '#FF6B6B', Prophet: '#4ECDC4', XGBoost: '#45B7D1', Baseline: '#96CEB4' };
const TICK = { fontSize: 11, fill: '#8899aa' };

function safe(v) { return v != null && isFinite(v) ? v : null; }

function ForecastChart({ data }) {
  if (!data || !data.historical_actuals) {
    return <div className="no-data">No forecast data available.</div>;
  }

  const actuals = data.historical_actuals;
  const forecasts = data.forecasts || {};
  const showLast = Math.min(90, actuals.length);
  const hist = actuals.slice(-showLast);
  const fLen = Object.values(forecasts).reduce((m, v) => Math.max(m, v?.length || 0), 0);
  if (!fLen) return <div className="no-data">No forecast data available.</div>;

  const lastActual = safe(hist[hist.length - 1]);
  const chartData = [];

  hist.forEach((v, i) => chartData.push({ t: i - showLast, actual: safe(v) }));

  Object.keys(forecasts).forEach(model => {
    chartData[chartData.length - 1][model] = lastActual;
  });

  for (let i = 0; i < fLen; i++) {
    const entry = { t: i + 1 };
    Object.entries(forecasts).forEach(([model, vals]) => {
      if (i < vals.length) entry[model] = safe(vals[i]);
    });
    chartData.push(entry);
  }

  return (
    <div className="card">
      <h3 className="card-title">Forecast vs Actual</h3>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData} margin={{ top: 5, right: 20, left: 5, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3e" />
          <XAxis dataKey="t" tick={TICK} stroke="#3a3a5e" domain={['dataMin', 'dataMax']}
            label={{ value: 'Days from forecast start', position: 'insideBottom', offset: -5, style: { fill: '#8899aa', fontSize: 11 } }} />
          <YAxis tick={TICK} stroke="#3a3a5e"
            label={{ value: 'Sales', angle: -90, position: 'insideLeft', offset: 10, style: { fill: '#8899aa', fontSize: 11 } }} />
          <Tooltip contentStyle={{ background: '#1e1e32', border: '1px solid #3a3a5e', borderRadius: 6, fontSize: 12 }}
            itemStyle={{ color: '#ccc' }} labelStyle={{ color: '#fff' }}
            formatter={(value, name) => [typeof value === 'number' ? value.toFixed(1) : '—', name]}
            labelFormatter={v => `${v > 0 ? '+' : ''}${v} days`} />
          <Legend wrapperStyle={{ fontSize: 12, color: '#ccc' }} />
          <ReferenceLine x={0} stroke="#555" strokeDasharray="5 5"
            label={{ value: 'Forecast Start', position: 'top', style: { fill: '#8899aa', fontSize: 11 } }} />
          <Line type="monotone" dataKey="actual" stroke="#e8e8e8" strokeWidth={2} dot={false} activeDot={{ r: 4 }} name="Actual" />
          {Object.keys(forecasts).filter(m => forecasts[m]?.some(v => v != null)).map(model => (
            <Line key={model} type="monotone" dataKey={model} stroke={COLORS[model] || '#95A5A6'}
              strokeWidth={1.5} dot={false} opacity={0.85} connectNulls={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export default ForecastChart;
