import { useEffect, useState } from 'react';
import apiClient from './apiClient';

const inputStyle = {
  width: '100%',
  boxSizing: 'border-box',
  padding: '0.45rem 0.55rem',
  borderRadius: '8px',
  border: '1px solid rgba(148, 163, 184, 0.25)',
  backgroundColor: 'rgba(15, 23, 42, 0.55)',
  color: '#e2e8f0',
  fontSize: '0.74rem',
  marginTop: '0.3rem',
};

const labelStyle = { fontSize: '0.7rem', color: '#94a3b8', marginTop: '0.45rem' };

function LlmProviderManager({ providers, onProvidersChanged }) {
  const [open, setOpen] = useState(false);
  const [presets, setPresets] = useState([]);
  const [selectedPreset, setSelectedPreset] = useState('custom');
  const [label, setLabel] = useState('');
  const [baseUrl, setBaseUrl] = useState('');
  const [model, setModel] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);

  const userProviders = (Array.isArray(providers) ? providers : []).filter((item) => item.user_defined);

  useEffect(() => {
    if (!open || presets.length > 0) return;
    apiClient
      .get('/llm_provider_presets')
      .then((response) => setPresets(response.data.presets || []))
      .catch(() => setStatus('Could not load provider presets.'));
  }, [open, presets.length]);

  const applyPreset = (presetId) => {
    setSelectedPreset(presetId);
    const preset = presets.find((item) => item.preset === presetId);
    if (!preset) return;
    setLabel(preset.preset === 'custom' ? '' : preset.label);
    setBaseUrl(preset.base_url || '');
    setModel(preset.model || '');
    setStatus(preset.hint || '');
  };

  const saveProvider = async () => {
    if (!label.trim() || !baseUrl.trim() || !model.trim()) {
      setStatus('Label, base URL, and model are all required.');
      return;
    }
    setBusy(true);
    setStatus('Saving provider...');
    try {
      const response = await apiClient.post('/llm_providers', {
        label: label.trim(),
        base_url: baseUrl.trim(),
        model: model.trim(),
        api_key: apiKey.trim() || null,
        preset: selectedPreset,
      });
      const savedId = response.data?.provider?.id;
      onProvidersChanged?.();
      // available comes from the backend catalog: true only when a key is stored
      // (typed now, or kept from a previous save on a keyless update)
      const savedEntry = (response.data?.llms || []).find((item) => item.id === savedId);
      if (savedId && savedEntry?.available) {
        setStatus(`Saved ${savedId}. Testing connection...`);
        const test = await apiClient.post(`/llm_providers/${savedId}/test`);
        if (test.data.ok) {
          setStatus(`${savedId}: connection ok in ${test.data.latency_sec}s (${test.data.model}).`);
        } else {
          setStatus(`${savedId}: saved, but test failed — ${test.data.error || `HTTP ${test.data.status_code}`}`);
        }
      } else if (savedId) {
        setStatus(`Saved ${savedId}. Add an API key to enable it.`);
      }
      setApiKey('');
    } catch (error) {
      const detail = error?.response?.data?.detail || error?.message || 'Failed to save provider.';
      setStatus(String(detail));
    } finally {
      setBusy(false);
    }
  };

  const removeProvider = async (providerId) => {
    setBusy(true);
    try {
      await apiClient.delete(`/llm_providers/${providerId}`);
      setStatus(`Removed ${providerId}.`);
      onProvidersChanged?.();
    } catch (error) {
      const detail = error?.response?.data?.detail || error?.message || 'Failed to remove provider.';
      setStatus(String(detail));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{ marginTop: '0.55rem' }}>
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        style={{
          background: 'transparent',
          border: 'none',
          color: '#7dd3fc',
          fontSize: '0.72rem',
          fontWeight: 700,
          cursor: 'pointer',
          padding: 0,
        }}
      >
        {open ? '▾ Manage providers' : '▸ Add / manage providers'}
      </button>
      {open && (
        <div
          style={{
            marginTop: '0.5rem',
            padding: '0.6rem 0.65rem',
            borderRadius: '10px',
            backgroundColor: 'rgba(15, 23, 42, 0.5)',
            border: '1px solid rgba(148, 163, 184, 0.12)',
          }}
        >
          <div style={labelStyle}>Preset</div>
          <select
            value={selectedPreset}
            onChange={(event) => applyPreset(event.target.value)}
            style={inputStyle}
            disabled={busy}
          >
            {presets.length === 0 && <option value="custom">Loading presets...</option>}
            {presets.map((preset) => (
              <option key={preset.preset} value={preset.preset}>
                {preset.label}
              </option>
            ))}
          </select>
          <div style={labelStyle}>Label</div>
          <input style={inputStyle} value={label} onChange={(event) => setLabel(event.target.value)} placeholder="My Anthropic" disabled={busy} />
          <div style={labelStyle}>Base URL (OpenAI-compatible)</div>
          <input style={inputStyle} value={baseUrl} onChange={(event) => setBaseUrl(event.target.value)} placeholder="https://api.example.com/v1" disabled={busy} />
          <div style={labelStyle}>Model</div>
          <input style={inputStyle} value={model} onChange={(event) => setModel(event.target.value)} placeholder="model-name" disabled={busy} />
          <div style={labelStyle}>API key (stored server-side only)</div>
          <input style={inputStyle} type="password" value={apiKey} onChange={(event) => setApiKey(event.target.value)} placeholder="sk-..." disabled={busy} />
          <button
            type="button"
            onClick={saveProvider}
            disabled={busy}
            style={{
              marginTop: '0.6rem',
              width: '100%',
              padding: '0.5rem 0.7rem',
              borderRadius: '8px',
              border: 'none',
              backgroundColor: '#0ea5e9',
              color: '#f0f9ff',
              fontWeight: 700,
              fontSize: '0.74rem',
              cursor: busy ? 'not-allowed' : 'pointer',
            }}
          >
            {busy ? 'Working...' : 'Save & Test'}
          </button>
          {userProviders.length > 0 && (
            <div style={{ marginTop: '0.6rem' }}>
              <div style={{ fontSize: '0.7rem', fontWeight: 700, color: '#cbd5e1' }}>Your providers</div>
              {userProviders.map((item) => (
                <div
                  key={item.id}
                  style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '0.3rem' }}
                >
                  <span style={{ fontSize: '0.72rem', color: '#e2e8f0' }}>
                    {item.label} <span style={{ color: '#94a3b8' }}>({item.model})</span>
                  </span>
                  <button
                    type="button"
                    onClick={() => removeProvider(item.id)}
                    disabled={busy}
                    style={{
                      background: 'transparent',
                      border: '1px solid rgba(248, 113, 113, 0.35)',
                      color: '#fca5a5',
                      borderRadius: '6px',
                      fontSize: '0.68rem',
                      padding: '0.15rem 0.5rem',
                      cursor: busy ? 'not-allowed' : 'pointer',
                    }}
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
          )}
          {status && (
            <div style={{ marginTop: '0.5rem', color: '#94a3b8', fontSize: '0.7rem', lineHeight: 1.4 }}>{status}</div>
          )}
        </div>
      )}
    </div>
  );
}

export default LlmProviderManager;
