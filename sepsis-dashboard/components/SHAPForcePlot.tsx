"use client"

import React from 'react';
import { ArrowLeft, ArrowRight, Target, TrendingUp, TrendingDown } from 'lucide-react';

interface SHAPData {
  feature: string;
  value: number;
  patientValue: number;
}

interface SHAPForcePlotProps {
  shapData: SHAPData[];
  baseProbability?: number;
  finalProbability?: number;
}

const SHAPForcePlot: React.FC<SHAPForcePlotProps> = ({ 
  shapData, 
  baseProbability = 0.5, 
  finalProbability 
}) => {
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

  const ForceArrow = ({ 
    direction, 
    strength, 
    colorClass, 
    label, 
    features 
  }: {
    direction: 'left' | 'right';
    strength: number;
    colorClass: string;
    label: string;
    features: SHAPData[];
  }) => (
    <div className="flex flex-col items-center space-y-2">
      <div className={`text-sm font-semibold ${colorClass}`}>{label}</div>
      <div className="flex items-center space-x-2">
        {direction === 'left' && (
          <ArrowLeft 
            className={`${colorClass} transition-all duration-500`}
            size={Math.max(20, strength * 0.8)}
            strokeWidth={Math.max(2, strength * 0.05)}
          />
        )}
        <div 
          className={`h-2 rounded transition-all duration-500 ${colorClass.replace('text-', 'bg-')}`}
          style={{ width: `${Math.max(20, strength * 2)}px` }}
        />
        {direction === 'right' && (
          <ArrowRight 
            className={`${colorClass} transition-all duration-500`}
            size={Math.max(20, strength * 0.8)}
            strokeWidth={Math.max(2, strength * 0.05)}
          />
        )}
      </div>
      <div className="text-xs text-muted-foreground">
        {(direction === 'left' ? totalNegative * 100 : totalPositive * 100).toFixed(1)}%
      </div>
      
      {/* Feature list */}
      <div className="space-y-1 text-xs">
        {features.slice(0, 3).map((feature, index) => (
          <div key={index} className="flex items-center justify-between bg-muted/30 px-2 py-1 rounded">
            <span className="font-medium">{feature.feature}</span>
            <span className={`${colorClass} font-semibold`}>
              {(feature.value * 100).toFixed(1)}%
            </span>
          </div>
        ))}
        {features.length > 3 && (
          <div className="text-muted-foreground text-center">
            +{features.length - 3} more
          </div>
        )}
      </div>
    </div>
  );

  const PredictionGauge = ({ probability }: { probability: number }) => {
    const angle = (probability - 0.5) * 180; // -90 to +90 degrees
    const colorClass = probability > 0.7 ? 'text-success' : probability > 0.3 ? 'text-warning' : 'text-destructive';
    
    return (
      <div className="flex flex-col items-center space-y-2">
        <div className="relative w-32 h-16 overflow-hidden">
          {/* Gauge background */}
          <div className="absolute inset-0 border-4 border-border rounded-t-full"></div>
          
          {/* Gauge sections */}
          <div className="absolute inset-0">
            <div className="absolute left-0 top-0 w-1/3 h-full bg-destructive/20 rounded-tl-full"></div>
            <div className="absolute left-1/3 top-0 w-1/3 h-full bg-warning/20"></div>
            <div className="absolute right-0 top-0 w-1/3 h-full bg-success/20 rounded-tr-full"></div>
          </div>
          
          {/* Needle */}
          <div 
            className="absolute bottom-0 left-1/2 w-0.5 h-12 bg-foreground origin-bottom transform -translate-x-0.5 transition-transform duration-1000"
            style={{ transform: `translateX(-50%) rotate(${angle}deg)` }}
          />
          
          {/* Center dot */}
          <div className="absolute bottom-0 left-1/2 w-2 h-2 bg-foreground rounded-full transform -translate-x-1/2" />
        </div>
        
        <div className="text-center">
          <div className={`text-2xl font-bold ${colorClass}`}>
            {(probability * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-muted-foreground">Survival Probability</div>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold flex items-center">
          <Target className="mr-2 text-primary" />
          SHAP Force Plot
        </h3>
        <div className="text-sm text-muted-foreground">
          Base: {(baseProbability * 100).toFixed(1)}% → Final: {(finalProb * 100).toFixed(1)}%
        </div>
      </div>

      <p className="text-muted-foreground text-sm">
        This force plot shows the "tug-of-war" between features pushing toward survival (right) 
        and features pushing toward mortality risk (left).
      </p>

      {/* Main Force Visualization */}
      <div className="bg-gradient-to-r from-destructive/10 via-muted/20 to-success/10 rounded-lg p-6">
        <div className="grid grid-cols-3 gap-8 items-center">
          {/* Negative Forces (Left) */}
          <div className="flex justify-center">
            {negativeContributions.length > 0 ? (
              <ForceArrow 
                direction="left"
                strength={negativeStrength}
                colorClass="text-destructive"
                label="Risk Factors"
                features={negativeContributions}
              />
            ) : (
              <div className="text-center text-muted-foreground">
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
                colorClass="text-success"
                label="Survival Factors"
                features={positiveContributions}
              />
            ) : (
              <div className="text-center text-muted-foreground">
                <TrendingUp className="mx-auto mb-2" size={24} />
                <div className="text-sm">No Survival Factors</div>
              </div>
            )}
          </div>
        </div>

        {/* Force Balance Indicator */}
        <div className="mt-6 flex items-center justify-center">
          <div className="flex items-center space-x-4">
            <div className="text-destructive font-semibold">
              ← Risk ({Math.abs(totalNegative * 100).toFixed(1)}%)
            </div>
            <div className="w-px h-8 bg-border" />
            <div className="text-center">
              <div className="text-sm text-muted-foreground">Net Effect</div>
              <div className={`font-bold ${netEffect > 0 ? 'text-success' : 'text-destructive'}`}>
                {netEffect > 0 ? '+' : ''}{(netEffect * 100).toFixed(1)}%
              </div>
            </div>
            <div className="w-px h-8 bg-border" />
            <div className="text-success font-semibold">
              Survival ({(totalPositive * 100).toFixed(1)}%) →
            </div>
          </div>
        </div>
      </div>

      {/* Detailed Feature Breakdown */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Risk Factors Detail */}
        <div className="bg-destructive/5 rounded-lg p-4 border border-destructive/20">
          <h4 className="font-semibold text-destructive mb-3 flex items-center">
            <TrendingDown className="mr-2" size={16} />
            Risk Factors ({negativeContributions.length})
          </h4>
          {negativeContributions.length > 0 ? (
            <div className="space-y-2">
              {negativeContributions.map((item, index) => (
                <div key={index} className="flex items-center justify-between bg-card p-2 rounded">
                  <div>
                    <div className="font-medium text-foreground">{item.feature}</div>
                    <div className="text-sm text-muted-foreground">Value: {item.patientValue}</div>
                  </div>
                  <div className="text-destructive font-semibold">
                    {(item.value * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
              <div className="border-t pt-2 mt-2">
                <div className="flex justify-between font-semibold text-destructive">
                  <span>Total Risk Impact:</span>
                  <span>{(totalNegative * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-destructive text-center py-4">
              No significant risk factors identified
            </div>
          )}
        </div>

        {/* Survival Factors Detail */}
        <div className="bg-success/5 rounded-lg p-4 border border-success/20">
          <h4 className="font-semibold text-success mb-3 flex items-center">
            <TrendingUp className="mr-2" size={16} />
            Survival Factors ({positiveContributions.length})
          </h4>
          {positiveContributions.length > 0 ? (
            <div className="space-y-2">
              {positiveContributions.map((item, index) => (
                <div key={index} className="flex items-center justify-between bg-card p-2 rounded">
                  <div>
                    <div className="font-medium text-foreground">{item.feature}</div>
                    <div className="text-sm text-muted-foreground">Value: {item.patientValue}</div>
                  </div>
                  <div className="text-success font-semibold">
                    +{(item.value * 100).toFixed(1)}%
                  </div>
                </div>
              ))}
              <div className="border-t pt-2 mt-2">
                <div className="flex justify-between font-semibold text-success">
                  <span>Total Survival Impact:</span>
                  <span>+{(totalPositive * 100).toFixed(1)}%</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="text-success text-center py-4">
              No significant survival factors identified
            </div>
          )}
        </div>
      </div>

      {/* Summary Statistics */}
      <div className="bg-primary/5 rounded-lg p-4 border border-primary/20">
        <h4 className="font-semibold text-primary mb-3">Force Analysis Summary</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          <div>
            <div className="text-lg font-bold text-primary">{(baseProbability * 100).toFixed(1)}%</div>
            <div className="text-sm text-primary">Base Rate</div>
          </div>
          <div>
            <div className="text-lg font-bold text-destructive">{(totalNegative * 100).toFixed(1)}%</div>
            <div className="text-sm text-destructive">Risk Force</div>
          </div>
          <div>
            <div className="text-lg font-bold text-success">+{(totalPositive * 100).toFixed(1)}%</div>
            <div className="text-sm text-success">Survival Force</div>
          </div>
          <div>
            <div className="text-lg font-bold text-primary">{(finalProb * 100).toFixed(1)}%</div>
            <div className="text-sm text-primary">Final Result</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SHAPForcePlot;