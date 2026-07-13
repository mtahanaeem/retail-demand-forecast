const API_BASE_URL = 'http://localhost:8000';

export async function fetchStores() {
  const res = await fetch(`${API_BASE_URL}/stores`);
  return res.json();
}

export async function fetchProductFamilies() {
  const res = await fetch(`${API_BASE_URL}/product-families`);
  return res.json();
}

export async function fetchForecast(storeId, productFamily, periods = 30) {
  const res = await fetch(`${API_BASE_URL}/forecast`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ store_id: storeId, product_family: productFamily, periods, use_backtest: false }),
  });
  return res.json();
}

export async function fetchMetrics(storeId, productFamily, nTestPeriods = 12) {
  const res = await fetch(`${API_BASE_URL}/metrics`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ store_id: storeId, product_family: productFamily, n_test_periods: nTestPeriods }),
  });
  return res.json();
}
