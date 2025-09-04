
// SepsisPredictor.tsx - React component for sepsis prediction
import React, { useState } from 'react';
import axios from 'axios';

interface PatientData {
  age: number;
  gender: number;
  heart_rate: number;
  systolic_bp: number;
  temperature: number;
  respiratory_rate: number;
  wbc_count: number;
  lactate: number;
  sofa_score: number;
}

interface PredictionResult {
  survival_probability: number;
  mortality_probability: number;
  risk_level: string;
  prediction: string;
  shap_explanations: Record<string, {
    value: number;
    shap_contribution: number;
    impact: string;
  }>;
}

const SepsisPredictor: React.FC = () => {
  const [patientData, setPatientData] = useState<PatientData>({
    age: 65,
    gender: 1,
    heart_rate: 85,
    systolic_bp: 120,
    temperature: 37.0,
    respiratory_rate: 16,
    wbc_count: 8.0,
    lactate: 1.5,
    sofa_score: 2
  });

  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>('');

  const handleInputChange = (field: keyof PatientData, value: string) => {
    setPatientData(prev => ({
      ...prev,
      [field]: parseFloat(value) || 0
    }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError('');

    try {
      const response = await axios.post<PredictionResult>(
        'http://localhost:8000/predict',
        patientData
      );
      setPrediction(response.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'low': return 'text-green-600 bg-green-100';
      case 'moderate': return 'text-yellow-600 bg-yellow-100';
      case 'high': return 'text-orange-600 bg-orange-100';
      case 'critical': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white shadow-lg rounded-lg">
      <h1 className="text-3xl font-bold text-center mb-6 text-blue-800">
        Sepsis Survival Predictor
      </h1>

      {/* Patient Input Form */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium mb-1">Age</label>
          <input
            type="number"
            value={patientData.age}
            onChange={(e) => handleInputChange('age', e.target.value)}
            className="w-full p-2 border rounded-md"
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Gender</label>
          <select
            value={patientData.gender}
            onChange={(e) => handleInputChange('gender', e.target.value)}
            className="w-full p-2 border rounded-md"
          >
            <option value={0}>Female</option>
            <option value={1}>Male</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Heart Rate</label>
          <input
            type="number"
            value={patientData.heart_rate}
            onChange={(e) => handleInputChange('heart_rate', e.target.value)}
            className="w-full p-2 border rounded-md"
            placeholder="beats/min"
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Systolic BP</label>
          <input
            type="number"
            value={patientData.systolic_bp}
            onChange={(e) => handleInputChange('systolic_bp', e.target.value)}
            className="w-full p-2 border rounded-md"
            placeholder="mmHg"
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Temperature</label>
          <input
            type="number"
            step="0.1"
            value={patientData.temperature}
            onChange={(e) => handleInputChange('temperature', e.target.value)}
            className="w-full p-2 border rounded-md"
            placeholder="°C"
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Respiratory Rate</label>
          <input
            type="number"
            value={patientData.respiratory_rate}
            onChange={(e) => handleInputChange('respiratory_rate', e.target.value)}
            className="w-full p-2 border rounded-md"
            placeholder="breaths/min"
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">WBC Count</label>
          <input
            type="number"
            step="0.1"
            value={patientData.wbc_count}
            onChange={(e) => handleInputChange('wbc_count', e.target.value)}
            className="w-full p-2 border rounded-md"
            placeholder="×10³/μL"
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Lactate</label>
          <input
            type="number"
            step="0.1"
            value={patientData.lactate}
            onChange={(e) => handleInputChange('lactate', e.target.value)}
            className="w-full p-2 border rounded-md"
            placeholder="mmol/L"
          />
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">SOFA Score</label>
          <input
            type="number"
            value={patientData.sofa_score}
            onChange={(e) => handleInputChange('sofa_score', e.target.value)}
            className="w-full p-2 border rounded-md"
            placeholder="0-24"
          />
        </div>
      </div>

      {/* Predict Button */}
      <div className="text-center mb-6">
        <button
          onClick={handlePredict}
          disabled={loading}
          className="bg-blue-600 text-white px-8 py-3 rounded-lg hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? 'Predicting...' : 'Predict Sepsis Risk'}
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
          {error}
        </div>
      )}

      {/* Prediction Results */}
      {prediction && (
        <div className="bg-gray-50 p-6 rounded-lg">
          <h2 className="text-2xl font-bold mb-4">Prediction Results</h2>

          {/* Risk Summary */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="bg-white p-4 rounded-lg text-center">
              <h3 className="text-lg font-semibold">Survival Probability</h3>
              <p className="text-3xl font-bold text-green-600">
                {(prediction.survival_probability * 100).toFixed(1)}%
              </p>
            </div>

            <div className="bg-white p-4 rounded-lg text-center">
              <h3 className="text-lg font-semibold">Mortality Risk</h3>
              <p className="text-3xl font-bold text-red-600">
                {(prediction.mortality_probability * 100).toFixed(1)}%
              </p>
            </div>

            <div className="bg-white p-4 rounded-lg text-center">
              <h3 className="text-lg font-semibold">Risk Level</h3>
              <p className={`text-2xl font-bold px-3 py-1 rounded-full ${getRiskColor(prediction.risk_level)}`}>
                {prediction.risk_level.toUpperCase()}
              </p>
            </div>
          </div>

          {/* SHAP Explanations */}
          <div className="bg-white p-4 rounded-lg">
            <h3 className="text-lg font-semibold mb-3">Feature Contributions (SHAP Analysis)</h3>
            <div className="space-y-2">
              {Object.entries(prediction.shap_explanations).map(([feature, explanation]) => (
                <div key={feature} className="flex justify-between items-center p-2 border-b">
                  <span className="font-medium">{feature.replace('_', ' ').toUpperCase()}</span>
                  <div className="text-right">
                    <span className="text-gray-600">Value: {explanation.value.toFixed(2)}</span>
                    <span className={`ml-4 px-2 py-1 rounded text-sm ${
                      explanation.impact === 'increases_survival' 
                        ? 'bg-green-100 text-green-800' 
                        : 'bg-red-100 text-red-800'
                    }`}>
                      {explanation.shap_contribution > 0 ? '+' : ''}{explanation.shap_contribution.toFixed(3)}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Information Footer */}
      <div className="mt-6 text-center text-gray-600 text-sm">
        <p>This is a demonstration model. Always consult medical professionals for clinical decisions.</p>
      </div>
    </div>
  );
};

export default SepsisPredictor;
