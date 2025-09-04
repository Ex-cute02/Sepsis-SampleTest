import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const sepsisAPI = {
  // Health check
  healthCheck: async () => {
    const response = await api.get('/health');
    return response.data;
  },

  // Single patient prediction
  predictSepsis: async (patientData) => {
    const response = await api.post('/predict', patientData);
    return response.data;
  },

  // Batch prediction
  batchPredict: async (patientsData) => {
    const response = await api.post('/batch_predict', patientsData);
    return response.data;
  },

  // Get feature importance
  getFeatureImportance: async () => {
    const response = await api.get('/feature_importance');
    return response.data;
  },
};

export default api;