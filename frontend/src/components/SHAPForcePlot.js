import React from 'react';
import { ArrowLeft, ArrowRight, Target, TrendingUp, TrendingDown } from 'lucide-react';

const SHAPForcePlot = ({ shapData, baseProbability = 0.5, finalProbability }) => {
  if (!shapData || shapData.length === 0) return null;

  // Separate positive and negative contributions
  const positiveContributions = shapData.filter(item => item.value > 0).sort((a, b) => b.value - a.value);
  const negativeContributions = shapData.filter(item => item.value < 0).sort((a, b) => a.value - b.value);

  const totalPositive = positiveContributions.reduce((sum, item) => sum + item.value, 0);
  const totalNegative = negativeContributions.reduce((sum, item) => sum + item.value, 0);
  const netEffect = totalPositive + totalNegative;
  const finalProb = finalProbability || (baseProbability + netEffect);

  // Calculate force strengths for visualization
  const maxAbsValue = Math.max(Math.abs(totalPositive), Math.abs(totalNegative));
  const positiveStrength = maxAbsValue > 0 ? (Math.abs(totalPositive) / maxAbsValue) * 100 : 0;
  const negativeStrength = maxAbsValue > 0 ? (Math.abs(totalNegative) / maxAbsValue) * 100 : 0;

  const ForceArrow = ({ direction, strength, color, label, features }) => (
    <div className="flex flex-col items-center space-y-2">
      <div className={`text-sm font-semibold ${color}`}>{label}</div>
      <div className="flex items-center space-x-2">
        {direction === 'left' && (
          <ArrowLeft 
            className={`${color} transition-all duration-500`}
            size={Math.max(20, strength * 0.8)}
            strokeWidth={Math.max(2, strength * 0.05)}
          />
        )}
        <div className={`h-2 rounded transition-all duration-500 ${color.replace('text-', 'bg-')}`}
             style={{ width: `${Math.max(20, strength * 2)}px` }}>
        </div>
        {direction === 'right' && (
          <ArrowRight 
            className={`${color} transition-all duration-500`}
            size={Math.max(20, strength * 0.8)}
            strokeWidth={Math.max(2, strength * 0.05)}
          />
        )}
      </div>
      <div className="text-xs text-gray-600">
        {(direction === 'left' ? totalNegative * 100 : totalPositive * 100).toFixed(1)}%
      </div>
      
      {/* Feature list */}
      <div className="space-y-1 text-xs">
        {features.slice(0, 3).map((feature, index) => (
          <div key={index} className="flex items-center justify-between bg-gray-50 px-2 py-1 rounded">
            <span className="font-medium">{feature.feature}</span>
            <span className={`${color} font-semibold`}>
              {(feature.value * 100).toFixed(1)}%
            </span>
          </div>
        ))}
        {features.length > 3 && (
          <div className="text-gray-500 text-center">
            +{features.length - 3} more
          </div>
        )}
      </div>
    </div>
  );

  const PredictionGauge = ({ probability }) => {
    const angle = (probability - 0.5) * 180; // -90 to +90 degrees
    const color = probability > 0.7 ? 'text-green-600' : probability > 0.3 ? 'text-yellow-600' : 'text-red-600';
    
    return (
      <div className="flex flex-col items-center space-y-2">
        <div className="relative w-32 h-16 overflow-hidden">
          {/* Gauge background */}
          <div className="absolute inset-0 border-4 border-gray-200 rounded-t-full"></div>
          
          {/* Gauge sections */}
          <div className="absolute inset-0">
            <div className="absolute left-0 top-0 w-1/3 h-full bg-red-100 rounded-tl-full opacity-50"></div>
            <div className="absolute left-1/3 top-0 w-1/3 h-full bg-yellow-100 opacity-50"></div>
            <div className="absolute right-0 top-0 w-1/3 h-full bg-green-100 rounded-tr-full opacity-50"></div>
          </div>
          
          {/* Needle */}
          <div 
            className="absolute bottom-0 left-1/2 w-0.5 h-12 bg-gray-800 origin-bottom transform -translate-x-0.5 transition-transform duration-1000"
            style={{ transform: `translateX(-50%) rotate(${angle}deg)` }}
          ></div>
          
          {/* Center dot */}
          <div className="absolute bottom-0 left-1/2 w-2 h-2 bg-gray-800 rounded-full transform -translate-x-1/2"></div>
        </div>
        
        <div className="text-center">
          <div className={`text-2xl font-bold ${color}`}>
            {(probability * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600">Survival Probability</div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold flex items-center">
          <Target className="mr-2 text-purple-600" />
          SHAP Force Plot
        </h3>
        <div className="text-sm text-gray-600">
          Base: {(baseProbability * 100).toFixed(1)}% → Final: {(finalProb * 100).toFixed(1)}%
        </div>
      </div>

      <p className="text-gray-600 text-sm">
        This force plot shows the "tug-of-war" between features pushing toward survival (right) 
        and features pushing toward mortality risk (left).
      </p>

      {/* Main Force Visualization */}
      <div className="bg-gradient-to-r from-red-50 via-gray-50 to-green-50 rounded-lg p-6">
        <div className="grid grid-cols-3 gap-8 items-center">
          {/* Negative Forces (Left) */}
          <div className="flex justify-center">
            {negativeContributions.length > 0 ? (
              <ForceArrow 
                direction="left"
                strength={negativeStrength}
                color="text-red-600"
                label="Risk Factors"
                features={negativeContributions}
              />
            ) : (
              <div className="text-center text-gray-400">
                <TrendingDown className="mx-auto mb-2" size={24} />
                <div className="text-sm">No Risk Factors</div>
              </div>
            )}
          </div>

          {/* Center Gauge */}
          <div className="flex justify-center">
            <PredictionGauge probability={finalProb} />
          </div>

          {/* Positive Forces (Right) */}
          <div className="flex justify-center">
            {positiveContributions.length > 0 ? (
              <ForceArrow 
                direction="right"
                strength={positiveStrength}
                color="text-green-600"
                label="Survival Factors"
                features={positiveContributions}
              />
            ) : (
              <div className="text-center text-gray-400">
                <TrendingUp className="mx-auto mb-2" size={24} />
                <div className="text-sm">No Survival Factors</div>
              </div>
            )}
          </div>
        </div>

        {/* Force Balance Indicator */}
        <div className="mt-6 flex items-center justify-center">
          <div className="flex items-center space-x-4">
            <div className="text-red-600 font-semibold">
              ← Risk ({Math.abs(totalNegative * 100).toFixed(1)}%)
            </div>
            <div className="w-px h-8 bg-gray-300"></div>
            <div className="text-center">
              <div className="text-sm text-gray-600">Net Effect</div>
              <div className={`font-bold ${netEffect > 0 ? 'text-green-600' : 'text-red-600'}`}>
                {netEffect > 0 ? '+' : ''}{(netEffect * 100).toFixed(1)}%
              </div>
            </div>
            <div className="w-px h-8 bg-gray-300"></div>
            <div className="text-green-600 font-semibold">
              Survival ({(totalPositive * 100).toFixed(1)}%) →
            </div>
          </div>
        </div>
      </div>

      {/* Detailed Feature Breakdown */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Risk Factors Detail */}
        <div className="bg-red-50 rounded-lg p-4 border border-red-200">
          <h4 className="font-semibold text-red-800 mb-3 flex items-center">
            <TrendingDown className="mr-2" size={16} />
            Risk Factors ({negativeContributions.length})
          </h4>
          {negativeContributions.length > 0 ? (
            <div className="space-y-2">
              {negativeContributions.map((item, index) => (
                <div key={index} className="flex items-center justify-between bg-white p-2 rounded">
                  <div>
                    <div className="font-medium text-gray-800">{item.feature}</div>
                    <div className="text-sm text-gray-600">Value: {item.patientValue}</div>
                  </div>
                  <div className="text-red-600 font-semibold">
                    {(item.value * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
              <div className="border-t pt-2 mt-2">
                <div className="flex justify-between font-semibold text-red-800">
                  <span>Total Risk Impact:</span>
                  <span>{(totalNegative * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-red-700 text-center py-4">
              No significant risk factors identified
            </div>
          )}
        </div>

        {/* Survival Factors Detail */}
        <div className="bg-green-50 rounded-lg p-4 border border-green-200">
          <h4 className="font-semibold text-green-800 mb-3 flex items-center">
            <TrendingUp className="mr-2" size={16} />
            Survival Factors ({positiveContributions.length})
          </h4>
          {positiveContributions.length > 0 ? (
            <div className="space-y-2">
              {positiveContributions.map((item, index) => (
                <div key={index} className="flex items-center justify-between bg-white p-2 rounded">
                  <div>
                    <div className="font-medium text-gray-800">{item.feature}</div>
                    <div className="text-sm text-gray-600">Value: {item.patientValue}</div>
                  </div>
                  <div className="text-green-600 font-semibold">
                    +{(item.value * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
              <div className="border-t pt-2 mt-2">
                <div className="flex justify-between font-semibold text-green-800">
                  <span>Total Survival Impact:</span>
                  <span>+{(totalPositive * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-green-700 text-center py-4">
              No significant survival factors identified
            </div>
          )}
        </div>
      </div>

      {/* Summary Statistics */}
      <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
        <h4 className="font-semibold text-blue-800 mb-3">Force Analysis Summary</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          <div>
            <div className="text-lg font-bold text-blue-700">{(baseProbability * 100).toFixed(1)}%</div>
            <div className="text-sm text-blue-600">Base Rate</div>
          </div>
          <div>
            <div className="text-lg font-bold text-red-600">{(totalNegative * 100).toFixed(1)}%</div>
            <div className="text-sm text-red-600">Risk Force</div>
          </div>
          <div>
            <div className="text-lg font-bold text-green-600">+{(totalPositive * 100).toFixed(1)}%</div>
            <div className="text-sm text-green-600">Survival Force</div>
          </div>
          <div>
            <div className="text-lg font-bold text-blue-700">{(finalProb * 100).toFixed(1)}%</div>
            <div className="text-sm text-blue-600">Final Result</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SHAPForcePlot;