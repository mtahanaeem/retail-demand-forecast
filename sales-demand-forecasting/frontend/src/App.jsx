import React from 'react';
import { useState, useEffect } from 'react';
import axios from 'axios';
import ForecastChart from './components/ForecastChart';
import SummaryCards from './components/SummaryCards';
import MetricsComparison from './components/MetricsComparison';
import './App.css';

const API_BASE = '';

function StoreSelector({ onSelect }) {
  const [stores, setStores] = useState([]);
  const [products, setProducts] = useState([]);
  const [store, setStore] = useState('');
  const [product, setProduct] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.all([
      axios.get(`${API_BASE}/stores`),
      axios.get(`${API_BASE}/product-families`),
    ]).then(axios.spread((s, p) => {
      setStores(s.data.stores);
      setProducts(p.data.product_families);
    })).catch(() => {}).finally(() => setLoading(false));
  }, []);

  const submit = e => {
    e.preventDefault();
    if (store && product) onSelect(store, product);
  };

  return (
    <form className="selector" onSubmit={submit}>
      <div className="field">
        <label>Store</label>
        <select value={store} onChange={e => setStore(e.target.value)} disabled={loading}>
          <option value="">{loading ? 'Loading...' : 'Select store'}</option>
          {stores.map(s => <option key={s} value={s}>Store {s}</option>)}
        </select>
      </div>
      <div className="field">
        <label>Product Family</label>
        <select value={product} onChange={e => setProduct(e.target.value)} disabled={loading}>
          <option value="">Select product</option>
          {products.map(p => <option key={p} value={p}>{p}</option>)}
        </select>
      </div>
      <button type="submit" disabled={!store || !product || loading}>
        Load Dashboard
      </button>
    </form>
  );
}

function App() {
  const [data, setData] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSelect = async (storeId, productFamily) => {
    setLoading(true);
    setError(null);
    setData(null);
    setMetrics(null);
    try {
      const [fRes, mRes] = await Promise.all([
        axios.post(`${API_BASE}/forecast`, { store_id: +storeId, product_family: productFamily, periods: 30 }),
        axios.post(`${API_BASE}/metrics`, { store_id: +storeId, product_family: productFamily, n_test_periods: 12 }),
      ]);
      setData(fRes.data.data);
      setMetrics(mRes.data.results);
    } catch (err) {
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>Retail Demand Forecast</h1>
        <p>Time series forecasting with multiple models</p>
      </header>

      <main className="main">
        <StoreSelector onSelect={handleSelect} />

        {loading && <div className="loading">Computing forecasts...</div>}
        {error && <div className="error">{error}</div>}

        {(data || metrics) && (
          <>
            <SummaryCards data={data} metrics={metrics} />
            <div className="grid">
              <div className="grid-main">
                <ForecastChart data={data} />
              </div>
              <div className="grid-side">
                <MetricsComparison metrics={metrics} />
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
