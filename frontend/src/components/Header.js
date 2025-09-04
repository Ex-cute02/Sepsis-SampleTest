import React, { useState, useEffect } from 'react';
import { Activity, Shield, AlertCircle, CheckCircle } from 'lucide-react';
import { sepsisAPI } from '../services/api';

const Header = () => {
  const [apiStatus, setApiStatus] = useState('checking');

  useEffect(() => {
    checkApiHealth();
  }, []);

  const checkApiHealth = async () => {
    try {
      await sepsisAPI.healthCheck();
      setApiStatus('healthy');
    } catch (error) {
      setApiStatus('error');
    }
  };

  const getStatusIcon = () => {
    switch (apiStatus) {
      case 'healthy':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'error':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      default:
        return <div className="w-5 h-5 border-2 border-gray-300 border-t-blue-500 rounded-full animate-spin" />;
    }
  };

  const getStatusText = () => {
    switch (apiStatus) {
      case 'healthy':
        return 'API Online';
      case 'error':
        return 'API Offline';
      default:
        return 'Checking...';
    }
  };

  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center py-4">
          <div className="flex items-center">
            <div className="flex items-center">
              <Shield className="w-8 h-8 text-primary-600 mr-3" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">
                  Sepsis Prediction System
                </h1>
                <p className="text-sm text-gray-600">
                  AI-powered clinical decision support
                </p>
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              {getStatusIcon()}
              <span className={`text-sm font-medium ${
                apiStatus === 'healthy' ? 'text-green-600' : 
                apiStatus === 'error' ? 'text-red-600' : 'text-gray-600'
              }`}>
                {getStatusText()}
              </span>
            </div>
            
            <div className="flex items-center space-x-2 px-3 py-1 bg-primary-50 rounded-full">
              <Activity className="w-4 h-4 text-primary-600" />
              <span className="text-sm font-medium text-primary-700">
                XGBoost + SHAP
              </span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;