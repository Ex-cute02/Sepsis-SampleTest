import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { TrendingUp, Info } from 'lucide-react';
import { sepsisAPI } from '../services/api';

const FeatureImportance = () => {
  const [importance, setImportance] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchFeatureImportance();
  }, []);

  const fetchFeatureImportance = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await sepsisAPI.getFeatureImportance();
      
      // Transform data for chart
      const chartData = Object.entries(data.feature_importance)
        .map(([feature, value]) => ({
          feature: formatFeatureName(feature),
          importance: value,
          percentage: (value * 100).toFixed(1)
        }))
        .sort((a, b) => b.importance - a.importance);
      
      setImportance(chartData);
    } catch (err) {
      setError('Failed to load feature importance data');
      console.error('Error fetching feature importance:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatFeatureName = (name) => {
    const nameMap = {
      'age': 'Age',
      'gender': 'Gender',
      'heart_rate': 'Heart Rate',
      'systolic_bp': 'Systolic BP',
      'temperature': 'Temperature',
      'respiratory_rate': 'Respiratory Rate',
      'wbc_count': 'WBC Count',
      'lactate': 'Lactate',
      'sofa_score': 'SOFA Score'
    };
    return nameMap[name] || name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  const getFeatureDescription = (feature) => {
    const descriptions = {
      'Age': 'Patient age in years - older patients typically have higher sepsis risk',
      'Gender': 'Patient gender - may influence sepsis outcomes',
      'Heart Rate': 'Heart rate in beats per minute - tachycardia is a sepsis indicator',
      'Systolic BP': 'Systolic blood pressure - hypotension indicates severe sepsis',
      'Temperature': 'Body temperature - fever or hypothermia can indicate sepsis',
      'Respiratory Rate': 'Breathing rate - tachypnea is an early sepsis sign',
      'WBC Count': 'White blood cell count - elevated or low counts suggest infection',
      'Lactate': 'Blood lactate level - elevated lactate indicates tissue hypoperfusion',
      'SOFA Score': 'Sequential Organ Failure Assessment - measures organ dysfunction'
    };
    return descriptions[feature] || 'Clinical parameter used in sepsis prediction';
  };

  if (loading) {
    return (
      <div className="card">
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
          <span className="ml-2 text-gray-600">Loading feature importance...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card">
        <div className="text-center py-8">
          <div className="text-red-500 mb-2">
            <Info className="w-8 h-8 mx-auto" />
          </div>
          <p className="text-red-600">{error}</p>
          <button 
            onClick={fetchFeatureImportance}
            className="btn-primary mt-4"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold text-gray-800 flex items-center">
          <TrendingUp className="mr-2 text-primary-600" />
          Model Feature Importance
        </h2>
        <button 
          onClick={fetchFeatureImportance}
          className="btn-secondary text-sm"
          disabled={loading}
        >
          Refresh
        </button>
      </div>

      <p className="text-gray-600 mb-6">
        This chart shows how important each clinical parameter is for the AI model's sepsis predictions. 
        Higher values indicate features that have more influence on the model's decisions.
      </p>

      {importance && (
        <>
          <div className="mb-6">
            <ResponsiveContainer width="100%" height={400}>
              <BarChart
                data={importance}
                margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="feature" 
                  angle={-45}
                  textAnchor="end"
                  height={100}
                  fontSize={12}
                />
                <YAxis 
                  label={{ value: 'Importance', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip 
                  formatter={(value) => [`${(value * 100).toFixed(1)}%`, 'Importance']}
                  labelFormatter={(label) => `Feature: ${label}`}
                />
                <Bar 
                  dataKey="importance" 
                  fill="#3b82f6"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {importance.map((item, index) => (
              <div 
                key={item.feature}
                className="p-4 border rounded-lg hover:shadow-md transition-shadow"
              >
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold text-gray-800">{item.feature}</h3>
                  <span className="text-sm font-medium text-primary-600">
                    {item.percentage}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
                  <div 
                    className="bg-primary-600 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${item.percentage}%` }}
                  ></div>
                </div>
                <p className="text-xs text-gray-600">
                  {getFeatureDescription(item.feature)}
                </p>
              </div>
            ))}
          </div>

          <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <h3 className="font-semibold text-blue-800 mb-2">Understanding Feature Importance</h3>
            <ul className="text-blue-700 text-sm space-y-1">
              <li>• Higher importance values indicate features that more strongly influence predictions</li>
              <li>• The model uses all features together to make final predictions</li>
              <li>• Feature importance is calculated using the XGBoost algorithm's built-in method</li>
              <li>• These values represent global importance across all training data</li>
            </ul>
          </div>
        </>
      )}
    </div>
  );
};

export default FeatureImportance;