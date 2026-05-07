const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();
const configuredWsBaseUrl = import.meta.env.VITE_WS_BASE_URL?.trim();

export function buildWebSocketUrl(path, params = {}) {
  const wsOrigin = configuredWsBaseUrl
    || (configuredApiBaseUrl
      ? configuredApiBaseUrl.replace(/^http/i, 'ws')
      : `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}`);

  const url = new URL(path, wsOrigin.endsWith('/') ? wsOrigin : `${wsOrigin}/`);
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      url.searchParams.set(key, String(value));
    }
  });
  return url.toString();
}
