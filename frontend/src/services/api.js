import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout
});

// Add request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('API Response Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export const sepsisAPI = {
  // Enhanced health check
  healthCheck: async () => {
    const response = await api.get('/health');
    return response.data;
  },

  // Enhanced single patient prediction
  predictSepsis: async (patientData) => {
    const response = await api.post('/predict', patientData);
    return response.data;
  },

  // Enhanced batch prediction
  batchPredict: async (patientsData) => {
    const response = await api.post('/predict_batch', patientsData);
    return response.data;
  },

  // Enhanced feature importance with descriptions
  getFeatureImportance: async () => {
    const response = await api.get('/feature_importance');
    return response.data;
  },

  // Get enhanced model information
  getModelInfo: async () => {
    const response = await api.get('/model_info');
    return response.data;
  },

  // Validate model readiness
  validateModel: async () => {
    const response = await api.get('/validate_model');
    return response.data;
  },

  // Test API connectivity
  testConnection: async () => {
    try {
      const response = await api.get('/');
      return { status: 'online', data: response.data };
    } catch (error) {
      return { status: 'offline', error: error.message };
    }
  }
};

export default api;