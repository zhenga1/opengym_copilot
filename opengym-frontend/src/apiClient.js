import axios from 'axios';

const configuredBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim();

const apiClient = axios.create({
  baseURL: configuredBaseUrl || undefined,
});

export default apiClient;
