import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import '../App.css';

function MetricsComparison({ metrics }) {
  if (!metrics || !Object.keys(metrics).length) {
    return <div className="no-data">No metrics available.</div>;
  }

  const chartData = Object.entries(metrics).map(([model, m]) => ({
    model, MAPE: m.MAPE || 0, RMSE: m.RMSE || 0, MAE: m.MAE || 0,
  }));

  const best = chartData.reduce((a, b) => (b.MAPE < a.MAPE ? b : a), chartData[0]);
  const FONT = { fontSize: 11, fill: '#8899aa' };

  return (
    <div className="card">
      <h3 className="card-title">Model Performance</h3>

      {best && (
        <div className="best-badge">
          Best: <strong>{best.model}</strong> &mdash; MAPE {best.MAPE.toFixed(1)}%
        </div>
      )}

      <ResponsiveContainer width="100%" height={240}>
        <BarChart data={chartData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2a3e" />
          <XAxis dataKey="model" tick={FONT} stroke="#3a3a5e" />
          <YAxis tick={FONT} stroke="#3a3a5e" />
          <Tooltip contentStyle={{ background: '#1e1e32', border: '1px solid #3a3a5e', borderRadius: 6, fontSize: 12 }} itemStyle={{ color: '#ccc' }} labelStyle={{ color: '#fff' }} formatter={(value, name) => [typeof value === 'number' ? value.toFixed(2) : value, name]} />
          <Legend wrapperStyle={{ fontSize: 11, color: '#ccc' }} />
          <Bar dataKey="MAPE" fill="#FF6B6B" radius={[4, 4, 0, 0]} />
          <Bar dataKey="RMSE" fill="#4ECDC4" radius={[4, 4, 0, 0]} />
          <Bar dataKey="MAE" fill="#45B7D1" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>

      <table className="metrics-table">
        <thead>
          <tr><th>Model</th><th>MAPE (%)</th><th>RMSE</th><th>MAE</th></tr>
        </thead>
        <tbody>
          {chartData.map(row => (
            <tr key={row.model} className={row.model === best.model ? 'best-row' : ''}>
              <td><strong>{row.model}</strong></td>
              <td>{row.MAPE.toFixed(2)}%</td>
              <td>{row.RMSE.toFixed(2)}</td>
              <td>{row.MAE.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default MetricsComparison;
