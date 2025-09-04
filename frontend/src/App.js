import React, { useState } from 'react';
import Header from './components/Header';
import PatientForm from './components/PatientForm';
import PredictionResult from './components/PredictionResult';
import FeatureImportance from './components/FeatureImportance';
import { sepsisAPI } from './services/api';
import { AlertCircle, BarChart3, User } from 'lucide-react';

function App() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('predict');

  const handlePrediction = async (patientData) => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const result = await sepsisAPI.predictSepsis(patientData);
      setPrediction(result);
    } catch (err) {
      setError(
        err.response?.data?.detail || 
        'Failed to get prediction. Please check if the API server is running.'
      );
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  const tabs = [
    { id: 'predict', label: 'Patient Prediction', icon: User },
    { id: 'importance', label: 'Feature Importance', icon: BarChart3 }
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      <Header />
      
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Tab Navigation */}
        <div className="mb-8">
          <div className="border-b border-gray-200">
            <nav className="-mb-px flex space-x-8">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`py-2 px-1 border-b-2 font-medium text-sm flex items-center ${
                      activeTab === tab.id
                        ? 'border-primary-500 text-primary-600'
                        : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                    }`}
                  >
                    <Icon className="w-4 h-4 mr-2" />
                    {tab.label}
                  </button>
                );
              })}
            </nav>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'predict' && (
          <div className="space-y-8">
            {/* Error Display */}
            {error && (
              <div className="card border-red-200 bg-red-50">
                <div className="flex items-center">
                  <AlertCircle className="w-5 h-5 text-red-500 mr-2" />
                  <div>
                    <h3 className="text-red-800 font-medium">Prediction Error</h3>
                    <p className="text-red-700 text-sm mt-1">{error}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Patient Form */}
            <PatientForm onSubmit={handlePrediction} loading={loading} />

            {/* Prediction Results */}
            {prediction && <PredictionResult result={prediction} />}

            {/* Instructions */}
            {!prediction && !loading && (
              <div className="card bg-blue-50 border-blue-200">
                <h3 className="text-lg font-semibold text-blue-800 mb-3">
                  How to Use This System
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-blue-700">
                  <div>
                    <h4 className="font-medium mb-2">1. Enter Patient Data</h4>
                    <ul className="text-sm space-y-1">
                      <li>• Fill in all required clinical parameters</li>
                      <li>• Use the "Load Sample Data" button for testing</li>
                      <li>• Ensure all values are within normal ranges</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-medium mb-2">2. Get AI Prediction</h4>
                    <ul className="text-sm space-y-1">
                      <li>• Click "Predict Sepsis Risk" to analyze</li>
                      <li>• View survival probability and risk level</li>
                      <li>• Review SHAP explanations for transparency</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-medium mb-2">3. Clinical Decision Support</h4>
                    <ul className="text-sm space-y-1">
                      <li>• Review automated clinical recommendations</li>
                      <li>• Understand which factors drive the prediction</li>
                      <li>• Use as support tool alongside clinical judgment</li>
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-medium mb-2">4. Model Transparency</h4>
                    <ul className="text-sm space-y-1">
                      <li>• SHAP values explain individual predictions</li>
                      <li>• Feature importance shows global model behavior</li>
                      <li>• All predictions include confidence levels</li>
                    </ul>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'importance' && (
          <FeatureImportance />
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-gray-600">
            <p className="text-sm">
              Sepsis Prediction System - AI-powered clinical decision support tool
            </p>
            <p className="text-xs mt-1">
              Built with XGBoost ML model and SHAP interpretability • For educational and research purposes
            </p>
            <div className="mt-2 text-xs text-gray-500">
              <strong>Disclaimer:</strong> This tool is for clinical decision support only. 
              Always consult qualified healthcare professionals for medical decisions.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;