import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { TrendingUp, TrendingDown, ArrowRight } from 'lucide-react';

const SHAPWaterfallPlot = ({ shapData, baseProbability = 0.5, finalProbability }) => {
  if (!shapData || shapData.length === 0) return null;

  // Prepare waterfall data
  const waterfallData = [];
  let cumulativeValue = baseProbability;

  // Add base rate
  waterfallData.push({
    feature: 'Base Rate',
    value: baseProbability,
    cumulative: cumulativeValue,
    type: 'base',
    contribution: 0,
    displayValue: `${(baseProbability * 100).toFixed(1)}%`
  });

  // Sort SHAP values by absolute magnitude for better visualization
  const sortedShapData = [...shapData].sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  // Add each SHAP contribution
  sortedShapData.forEach((item, index) => {
    const previousCumulative = cumulativeValue;
    cumulativeValue += item.value;
    
    waterfallData.push({
      feature: item.feature,
      value: item.value,
      cumulative: cumulativeValue,
      type: item.value > 0 ? 'positive' : 'negative',
      contribution: item.value,
      patientValue: item.patientValue,
      displayValue: `${item.value > 0 ? '+' : ''}${(item.value * 100).toFixed(1)}%`,
      previousCumulative: previousCumulative
    });
  });

  // Add final prediction
  waterfallData.push({
    feature: 'Final Prediction',
    value: finalProbability || cumulativeValue,
    cumulative: finalProbability || cumulativeValue,
    type: 'final',
    contribution: 0,
    displayValue: `${((finalProbability || cumulativeValue) * 100).toFixed(1)}%`
  });

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white p-4 border rounded-lg shadow-lg max-w-xs">
          <p className="font-semibold text-gray-800">{label}</p>
          {data.patientValue !== undefined && (
            <p className="text-sm text-gray-600">Patient Value: {data.patientValue}</p>
          )}
          {data.contribution !== 0 && (
            <p className={`text-sm font-medium ${data.contribution > 0 ? 'text-green-600' : 'text-red-600'}`}>
              Contribution: {data.displayValue}
            </p>
          )}
          <p className="text-sm text-blue-600">
            Cumulative: {(data.cumulative * 100).toFixed(1)}%
          </p>
          {data.type === 'positive' && (
            <p className="text-xs text-green-600 mt-1">↑ Increases survival probability</p>
          )}
          {data.type === 'negative' && (
            <p className="text-xs text-red-600 mt-1">↓ Decreases survival probability</p>
          )}
        </div>
      );
    }
    return null;
  };

  const getBarColor = (type) => {
    switch (type) {
      case 'base': return '#6b7280';
      case 'positive': return '#22c55e';
      case 'negative': return '#ef4444';
      case 'final': return '#3b82f6';
      default: return '#6b7280';
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold flex items-center">
          <TrendingUp className="mr-2 text-blue-600" />
          SHAP Waterfall Plot
        </h3>
        <div className="text-sm text-gray-600">
          Base: {(baseProbability * 100).toFixed(1)}% → Final: {((finalProbability || cumulativeValue) * 100).toFixed(1)}%
        </div>
      </div>

      <p className="text-gray-600 text-sm">
        This waterfall chart shows how each feature contributes step-by-step to the final prediction, 
        starting from the base survival rate.
      </p>

      <ResponsiveContainer width="100%" height={400}>
        <BarChart
          data={waterfallData}
          margin={{ top: 20, right: 30, left: 20, bottom: 80 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis 
            dataKey="feature" 
            angle={-45}
            textAnchor="end"
            height={100}
            fontSize={11}
            interval={0}
          />
          <YAxis 
            domain={[0, 1]}
            tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
            label={{ value: 'Survival Probability', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={0.5} stroke="#9ca3af" strokeDasharray="5 5" />
          <Bar 
            dataKey="cumulative" 
            fill={(entry) => getBarColor(entry.type)}
            radius={[2, 2, 0, 0]}
          >
            {waterfallData.map((entry, index) => (
              <Bar key={`bar-${index}`} fill={getBarColor(entry.type)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Step-by-step breakdown */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h4 className="font-semibold text-gray-800 mb-3">Step-by-Step Breakdown</h4>
        <div className="space-y-2">
          {waterfallData.map((item, index) => (
            <div key={index} className="flex items-center justify-between text-sm">
              <div className="flex items-center">
                <div 
                  className="w-3 h-3 rounded mr-2"
                  style={{ backgroundColor: getBarColor(item.type) }}
                ></div>
                <span className="font-medium">{item.feature}</span>
                {item.patientValue !== undefined && (
                  <span className="text-gray-500 ml-2">({item.patientValue})</span>
                )}
              </div>
              <div className="flex items-center">
                {item.contribution !== 0 && (
                  <>
                    <span className={`font-medium mr-2 ${item.contribution > 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {item.displayValue}
                    </span>
                    <ArrowRight className="w-3 h-3 text-gray-400 mr-2" />
                  </>
                )}
                <span className="font-semibold text-blue-600">
                  {(item.cumulative * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
        <div className="p-3 bg-gray-100 rounded-lg">
          <div className="text-lg font-bold text-gray-700">{(baseProbability * 100).toFixed(1)}%</div>
          <div className="text-sm text-gray-600">Starting Base Rate</div>
        </div>
        <div className="p-3 bg-blue-100 rounded-lg">
          <div className="text-lg font-bold text-blue-700">
            {((finalProbability || cumulativeValue) - baseProbability > 0 ? '+' : '')}
            {(((finalProbability || cumulativeValue) - baseProbability) * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-blue-600">Total SHAP Impact</div>
        </div>
        <div className="p-3 bg-blue-100 rounded-lg">
          <div className="text-lg font-bold text-blue-700">
            {((finalProbability || cumulativeValue) * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-blue-600">Final Prediction</div>
        </div>
      </div>
    </div>
  );
};

export default SHAPWaterfallPlot;