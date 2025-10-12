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
      console.log('Feature importance API response:', data);
      
      // Validate API response structure
      if (!data || !data.feature_importance || !Array.isArray(data.feature_importance)) {
        throw new Error('Invalid API response format');
      }
      
      // Transform data for chart - API returns array format
      const chartData = data.feature_importance
        .map((item) => ({
          feature: formatFeatureName(item.feature),
          originalFeature: item.feature,
          importance: item.importance,
          percentage: (item.importance * 100).toFixed(1),
          description: item.description
        }))
        .sort((a, b) => b.importance - a.importance);
      
      console.log('Transformed chart data:', chartData);
      setImportance(chartData);
    } catch (err) {
      setError(`Failed to load feature importance data: ${err.message}`);
      console.error('Error fetching feature importance:', err);
    } finally {
      setLoading(false);
    }
  };

  const formatFeatureName = (name) => {
    const nameMap = {
      'Age': 'Age',
      'Gender': 'Gender', 
      'HR': 'Heart Rate',
      'SBP': 'Systolic BP',
      'DBP': 'Diastolic BP',
      'Temp': 'Temperature',
      'Resp': 'Respiratory Rate',
      'O2Sat': 'Oxygen Saturation',
      'MAP': 'Mean Arterial Pressure',
      'Glucose': 'Blood Glucose',
      'ICULOS': 'ICU Length of Stay',
      'HR_abnormal': 'HR Abnormal Flag',
      'Temp_abnormal': 'Temperature Abnormal Flag',
      'Age_elderly': 'Elderly Age Flag',
      'Age_category': 'Age Category',
      'ICULOS_long': 'Long ICU Stay Flag',
      'Temp_missing': 'Temperature Missing Flag',
      'Glucose_missing': 'Glucose Missing Flag',
      'HR_SBP_ratio': 'HR/SBP Ratio'
    };
    return nameMap[name] || name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
  };

  const getFeatureDescription = (item) => {
    // Use API-provided description if available, otherwise use formatted feature name
    return item.description || `Clinical parameter: ${item.originalFeature}`;
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
          <h3 className="text-lg font-semibold text-red-600 mb-2">Feature Importance Error</h3>
          <p className="text-red-600 mb-4">{error}</p>
          <div className="text-sm text-gray-600 mb-4">
            <p>Possible causes:</p>
            <ul className="list-disc list-inside mt-2">
              <li>API server not running on http://localhost:8000</li>
              <li>Model not loaded properly</li>
              <li>Network connectivity issues</li>
            </ul>
          </div>
          <button 
            onClick={fetchFeatureImportance}
            className="btn-primary mt-4"
            disabled={loading}
          >
            {loading ? 'Retrying...' : 'Retry'}
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

      {importance && importance.length > 0 ? (
        <>
          <div className="mb-6 bg-white border rounded-lg p-4">
            <h4 className="text-lg font-semibold mb-4 text-center">Feature Importance Distribution</h4>
            <div className="text-sm text-gray-600 text-center mb-4">
              Showing {importance.length} features ranked by importance to the AI model
            </div>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart
                data={importance}
                margin={{ top: 20, right: 30, left: 20, bottom: 100 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis 
                  dataKey="feature" 
                  angle={-45}
                  textAnchor="end"
                  height={100}
                  fontSize={11}
                  interval={0}
                />
                <YAxis 
                  label={{ value: 'Importance Score', angle: -90, position: 'insideLeft' }}
                  fontSize={11}
                />
                <Tooltip 
                  formatter={(value) => [`${(value * 100).toFixed(1)}%`, 'Importance']}
                  labelFormatter={(label) => `Feature: ${label}`}
                  contentStyle={{
                    backgroundColor: 'white',
                    border: '1px solid #ccc',
                    borderRadius: '8px',
                    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                  }}
                />
                <Bar 
                  dataKey="importance" 
                  fill="#3b82f6"
                  radius={[4, 4, 0, 0]}
                  stroke="#2563eb"
                  strokeWidth={1}
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
                  {getFeatureDescription(item)}
                </p>
              </div>
            ))}
          </div>

          {/* Summary Statistics */}
          <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 bg-green-50 border border-green-200 rounded-lg text-center">
              <div className="text-2xl font-bold text-green-700">{importance.length}</div>
              <div className="text-sm text-green-600">Total Features</div>
            </div>
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg text-center">
              <div className="text-2xl font-bold text-blue-700">
                {importance.slice(0, 5).reduce((sum, item) => sum + item.importance, 0).toFixed(3)}
              </div>
              <div className="text-sm text-blue-600">Top 5 Combined Importance</div>
            </div>
            <div className="p-4 bg-purple-50 border border-purple-200 rounded-lg text-center">
              <div className="text-2xl font-bold text-purple-700">
                {importance[0]?.percentage || '0'}%
              </div>
              <div className="text-sm text-purple-600">Highest Single Feature</div>
            </div>
          </div>

          <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <h3 className="font-semibold text-blue-800 mb-2">Understanding Feature Importance</h3>
            <ul className="text-blue-700 text-sm space-y-1">
              <li>• Higher importance values indicate features that more strongly influence predictions</li>
              <li>• The model uses all features together to make final predictions</li>
              <li>• Feature importance is calculated using the XGBoost algorithm's built-in method</li>
              <li>• These values represent global importance across all training data</li>
              <li>• Missing value flags and engineered features may have high importance</li>
            </ul>
          </div>
        </>
      ) : (
        !loading && !error && (
          <div className="text-center py-8">
            <div className="text-gray-400 mb-4">
              <TrendingUp className="w-12 h-12 mx-auto" />
            </div>
            <h3 className="text-lg font-semibold text-gray-600 mb-2">No Feature Importance Data</h3>
            <p className="text-gray-500 mb-4">
              Feature importance data is not available. This could be because:
            </p>
            <ul className="text-sm text-gray-500 list-disc list-inside mb-4">
              <li>The model doesn't support feature importance</li>
              <li>The data hasn't loaded yet</li>
              <li>There was an error loading the data</li>
            </ul>
            <button 
              onClick={fetchFeatureImportance}
              className="btn-primary"
            >
              Try Loading Data
            </button>
          </div>
        )
      )}
    </div>
  );
};

export default FeatureImportance;