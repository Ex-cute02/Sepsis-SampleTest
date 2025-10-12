import React, { useState } from 'react';
import { AlertTriangle, CheckCircle, XCircle, TrendingUp, TrendingDown, Info, BarChart3, Target, Waves } from 'lucide-react';
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import SHAPWaterfallPlot from './SHAPWaterfallPlot';
import SHAPForcePlot from './SHAPForcePlot';

const PredictionResult = ({ result }) => {
  const [activeVisualization, setActiveVisualization] = useState('bar');
  
  if (!result) return null;

  // Handle both old and new API response formats
  const isEnhancedResponse = result.clinical_alerts || result.recommendations;

  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'low': return 'text-green-600 bg-green-50 border-green-200';
      case 'moderate': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'high': return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'critical': return 'text-red-600 bg-red-50 border-red-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  const getRiskIcon = (riskLevel) => {
    switch (riskLevel) {
      case 'low': return <CheckCircle className="w-6 h-6" />;
      case 'moderate': return <Info className="w-6 h-6" />;
      case 'high': return <AlertTriangle className="w-6 h-6" />;
      case 'critical': return <XCircle className="w-6 h-6" />;
      default: return <Info className="w-6 h-6" />;
    }
  };

  // Prepare data for pie chart
  const pieData = [
    { name: 'Survival', value: result.survival_probability * 100, color: '#22c55e' },
    { name: 'Mortality', value: result.mortality_probability * 100, color: '#ef4444' }
  ];

  // Prepare SHAP data for bar chart
  const shapData = Object.entries(result.shap_explanations || {})
    .map(([feature, data]) => ({
      feature: feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      value: data.shap_contribution,
      impact: data.impact,
      patientValue: data.value
    }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  const formatFeatureName = (name) => {
    const nameMap = {
      'Age': 'Age',
      'Gender': 'Gender',
      'Heart Rate': 'Heart Rate',
      'Systolic Bp': 'Systolic BP',
      'Temperature': 'Temperature',
      'Respiratory Rate': 'Respiratory Rate',
      'Wbc Count': 'WBC Count',
      'Lactate': 'Lactate',
      'Sofa Score': 'SOFA Score'
    };
    return nameMap[name] || name;
  };

  return (
    <div className="space-y-6">
      {/* Main Prediction Result */}
      <div className="card">
        <div className="text-center">
          <div className={`inline-flex items-center px-6 py-3 rounded-full border-2 ${getRiskColor(result.risk_level)} mb-4`}>
            {getRiskIcon(result.risk_level)}
            <span className="ml-2 text-xl font-bold capitalize">{result.risk_level} Risk</span>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
            {/* Probability Chart */}
            <div>
              <h3 className="text-lg font-semibold mb-4">Prediction Probabilities</h3>
              <ResponsiveContainer width="100%" height={200}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip formatter={(value) => `${value.toFixed(1)}%`} />
                </PieChart>
              </ResponsiveContainer>
              <div className="flex justify-center space-x-4 mt-2">
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
                  <span className="text-sm">Survival: {(result.survival_probability * 100).toFixed(1)}%</span>
                </div>
                <div className="flex items-center">
                  <div className="w-3 h-3 bg-red-500 rounded-full mr-2"></div>
                  <span className="text-sm">Mortality: {(result.mortality_probability * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>

            {/* Key Metrics */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">Key Metrics</h3>
              <div className="space-y-3">
                <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                  <span className="font-medium">Prediction:</span>
                  <span className={`font-bold ${result.prediction === 'survived' ? 'text-green-600' : 'text-red-600'}`}>
                    {result.prediction === 'survived' ? 'Likely to Survive' : 'High Risk'}
                  </span>
                </div>
                <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                  <span className="font-medium">Risk Level:</span>
                  <span className={`font-bold capitalize ${result.risk_level === 'low' ? 'text-green-600' : 
                    result.risk_level === 'moderate' ? 'text-yellow-600' : 
                    result.risk_level === 'high' ? 'text-orange-600' : 'text-red-600'}`}>
                    {result.risk_level}
                  </span>
                </div>
                <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                  <span className="font-medium">Survival Probability:</span>
                  <span className="font-bold text-green-600">
                    {(result.survival_probability * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                  <span className="font-medium">Mortality Risk:</span>
                  <span className="font-bold text-red-600">
                    {(result.mortality_probability * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* SHAP Explanations */}
      {result.shap_explanations && (
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold flex items-center">
              <TrendingUp className="mr-2 text-primary-600" />
              AI Model Explanations (SHAP Values)
            </h3>
            
            {/* Visualization Toggle */}
            <div className="flex bg-gray-100 rounded-lg p-1">
              <button
                onClick={() => setActiveVisualization('bar')}
                className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                  activeVisualization === 'bar' 
                    ? 'bg-white text-blue-600 shadow-sm' 
                    : 'text-gray-600 hover:text-gray-800'
                }`}
              >
                <BarChart3 className="w-4 h-4 inline mr-1" />
                Bar Chart
              </button>
              <button
                onClick={() => setActiveVisualization('waterfall')}
                className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                  activeVisualization === 'waterfall' 
                    ? 'bg-white text-blue-600 shadow-sm' 
                    : 'text-gray-600 hover:text-gray-800'
                }`}
              >
                <Waves className="w-4 h-4 inline mr-1" />
                Waterfall
              </button>
              <button
                onClick={() => setActiveVisualization('force')}
                className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                  activeVisualization === 'force' 
                    ? 'bg-white text-blue-600 shadow-sm' 
                    : 'text-gray-600 hover:text-gray-800'
                }`}
              >
                <Target className="w-4 h-4 inline mr-1" />
                Force Plot
              </button>
            </div>
          </div>
          
          {/* Visualization Description */}
          <div className="mb-6">
            {activeVisualization === 'bar' && (
              <p className="text-gray-600 text-sm">
                This bar chart shows how each patient factor influences the prediction. 
                Positive values increase survival probability, negative values increase mortality risk.
              </p>
            )}
            {activeVisualization === 'waterfall' && (
              <p className="text-gray-600 text-sm">
                The waterfall chart shows step-by-step how we get from the base survival rate to the final prediction,
                with each feature adding or subtracting probability.
              </p>
            )}
            {activeVisualization === 'force' && (
              <p className="text-gray-600 text-sm">
                The force plot visualizes the "tug-of-war" between features pushing toward survival versus mortality,
                showing the balance of forces that determine the final prediction.
              </p>
            )}
          </div>
          
          {/* Dynamic Visualization Content */}
          {activeVisualization === 'bar' && (
            <ResponsiveContainer width="100%" height={400}>
              <BarChart
                data={shapData}
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
                <YAxis />
                <Tooltip 
                  formatter={(value, name, props) => [
                    `${value.toFixed(4)}`,
                    'SHAP Value'
                  ]}
                  labelFormatter={(label) => `${formatFeatureName(label)}`}
                  content={({ active, payload, label }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-white p-3 border rounded-lg shadow-lg">
                          <p className="font-semibold">{formatFeatureName(label)}</p>
                          <p className="text-sm">Patient Value: {data.patientValue}</p>
                          <p className="text-sm">SHAP Value: {data.value.toFixed(4)}</p>
                          <p className={`text-sm font-medium ${data.value > 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {data.value > 0 ? 'Increases Survival' : 'Increases Risk'}
                          </p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Bar 
                  dataKey="value" 
                  fill={(entry) => entry.value > 0 ? '#22c55e' : '#ef4444'}
                  name="SHAP Value"
                >
                  {shapData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#22c55e' : '#ef4444'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}

          {activeVisualization === 'waterfall' && (
            <SHAPWaterfallPlot 
              shapData={shapData}
              baseProbability={0.5} // You can make this dynamic based on your model's base rate
              finalProbability={result.survival_probability}
            />
          )}

          {activeVisualization === 'force' && (
            <SHAPForcePlot 
              shapData={shapData}
              baseProbability={0.5} // You can make this dynamic based on your model's base rate
              finalProbability={result.survival_probability}
            />
          )}

          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-green-50 rounded-lg border border-green-200">
              <div className="flex items-center mb-2">
                <TrendingUp className="w-5 h-5 text-green-600 mr-2" />
                <h4 className="font-semibold text-green-800">Factors Increasing Survival</h4>
              </div>
              <ul className="space-y-1 text-sm">
                {shapData.filter(item => item.value > 0).slice(0, 3).map((item, index) => (
                  <li key={index} className="text-green-700">
                    • {formatFeatureName(item.feature)}: {item.patientValue}
                  </li>
                ))}
              </ul>
            </div>
            
            <div className="p-4 bg-red-50 rounded-lg border border-red-200">
              <div className="flex items-center mb-2">
                <TrendingDown className="w-5 h-5 text-red-600 mr-2" />
                <h4 className="font-semibold text-red-800">Risk Factors</h4>
              </div>
              <ul className="space-y-1 text-sm">
                {shapData.filter(item => item.value < 0).slice(0, 3).map((item, index) => (
                  <li key={index} className="text-red-700">
                    • {formatFeatureName(item.feature)}: {item.patientValue}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Enhanced Clinical Alerts */}
      {isEnhancedResponse && result.clinical_alerts && result.clinical_alerts.length > 0 && (
        <div className="card">
          <h3 className="text-xl font-bold mb-4 flex items-center">
            <AlertTriangle className="mr-2 text-red-500" />
            Clinical Alerts
          </h3>
          <div className="space-y-2">
            {result.clinical_alerts.map((alert, index) => {
              const isCritical = alert.includes('🔴') || alert.includes('🚨');
              const isModerate = alert.includes('🟡') || alert.includes('⚠️');
              
              return (
                <div 
                  key={index}
                  className={`p-3 rounded-lg border ${
                    isCritical ? 'bg-red-50 border-red-200 text-red-800' :
                    isModerate ? 'bg-yellow-50 border-yellow-200 text-yellow-800' :
                    'bg-blue-50 border-blue-200 text-blue-800'
                  }`}
                >
                  <div className="flex items-start">
                    {isCritical ? <XCircle className="w-4 h-4 mt-0.5 mr-2 flex-shrink-0" /> :
                     isModerate ? <AlertTriangle className="w-4 h-4 mt-0.5 mr-2 flex-shrink-0" /> :
                     <Info className="w-4 h-4 mt-0.5 mr-2 flex-shrink-0" />}
                    <span className="text-sm">{alert.replace(/🔴|🟡|🚨|⚠️/g, '').trim()}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Enhanced Clinical Recommendations */}
      <div className="card">
        <h3 className="text-xl font-bold mb-4 flex items-center">
          <CheckCircle className="mr-2 text-green-500" />
          Clinical Recommendations
        </h3>
        
        {/* Enhanced API Recommendations */}
        {isEnhancedResponse && result.recommendations && result.recommendations.length > 0 ? (
          <div className="space-y-2">
            {result.recommendations.map((recommendation, index) => (
              <div key={index} className="p-3 bg-green-50 border border-green-200 rounded-lg">
                <div className="flex items-start">
                  <CheckCircle className="w-4 h-4 mt-0.5 mr-2 text-green-600 flex-shrink-0" />
                  <span className="text-sm text-green-800">{recommendation}</span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          /* Fallback recommendations based on risk level */
          <div className="space-y-3">
            {result.risk_level === 'critical' && (
              <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                <h4 className="font-semibold text-red-800 mb-2">Critical Risk - Immediate Action Required</h4>
                <ul className="text-red-700 text-sm space-y-1">
                  <li>• Consider ICU admission</li>
                  <li>• Initiate sepsis protocol immediately</li>
                  <li>• Frequent vital sign monitoring</li>
                  <li>• Consider vasopressor support</li>
                </ul>
              </div>
            )}
            
            {result.risk_level === 'high' && (
              <div className="p-4 bg-orange-50 border border-orange-200 rounded-lg">
                <h4 className="font-semibold text-orange-800 mb-2">High Risk - Close Monitoring</h4>
                <ul className="text-orange-700 text-sm space-y-1">
                  <li>• Enhanced monitoring protocols</li>
                  <li>• Consider early intervention</li>
                  <li>• Regular reassessment</li>
                  <li>• Prepare for potential escalation</li>
                </ul>
              </div>
            )}
            
            {result.risk_level === 'moderate' && (
              <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                <h4 className="font-semibold text-yellow-800 mb-2">Moderate Risk - Standard Care</h4>
                <ul className="text-yellow-700 text-sm space-y-1">
                  <li>• Continue standard monitoring</li>
                  <li>• Regular vital sign checks</li>
                  <li>• Monitor for deterioration</li>
                  <li>• Follow institutional protocols</li>
                </ul>
              </div>
            )}
            
            {result.risk_level === 'low' && (
              <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                <h4 className="font-semibold text-green-800 mb-2">Low Risk - Routine Care</h4>
                <ul className="text-green-700 text-sm space-y-1">
                  <li>• Routine monitoring sufficient</li>
                  <li>• Standard care protocols</li>
                  <li>• Continue current treatment</li>
                  <li>• Regular reassessment as scheduled</li>
                </ul>
              </div>
            )}
          </div>
        )}
        
        {/* Enhanced API Metadata */}
        {isEnhancedResponse && (
          <div className="mt-4 p-3 bg-gray-50 border border-gray-200 rounded-lg">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              {result.patient_id && (
                <div>
                  <span className="font-medium text-gray-600">Patient ID:</span>
                  <div className="text-gray-800">{result.patient_id}</div>
                </div>
              )}
              {result.confidence && (
                <div>
                  <span className="font-medium text-gray-600">Confidence:</span>
                  <div className="text-gray-800">{(result.confidence * 100).toFixed(1)}%</div>
                </div>
              )}
              {result.risk_score && (
                <div>
                  <span className="font-medium text-gray-600">Risk Score:</span>
                  <div className="text-gray-800">{result.risk_score}/4</div>
                </div>
              )}
              {result.processing_time_ms && (
                <div>
                  <span className="font-medium text-gray-600">Processing:</span>
                  <div className="text-gray-800">{result.processing_time_ms.toFixed(1)}ms</div>
                </div>
              )}
            </div>
            {result.model_version && (
              <div className="mt-2 text-xs text-gray-500">
                Model: {result.model_version} • Timestamp: {result.timestamp ? new Date(result.timestamp).toLocaleString() : 'N/A'}
              </div>
            )}
          </div>
        )}
        
        <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <p className="text-blue-800 text-sm">
            <strong>Disclaimer:</strong> This AI prediction is a clinical decision support tool. 
            Always use clinical judgment and follow institutional protocols. 
            This tool should not replace clinical assessment by qualified healthcare professionals.
          </p>
        </div>
      </div>
    </div>
  );
};

export default PredictionResult;