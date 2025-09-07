"use client"

import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { TrendingUp, ArrowRight } from 'lucide-react';

interface SHAPData {
  feature: string;
  value: number;
  patientValue: number;
}

interface WaterfallData {
  feature: string;
  value: number;
  cumulative: number;
  type: 'base' | 'positive' | 'negative' | 'final';
  contribution: number;
  patientValue?: number;
  displayValue: string;
  previousCumulative?: number;
}

interface SHAPWaterfallPlotProps {
  shapData: SHAPData[];
  baseProbability?: number;
  finalProbability?: number;
}

const SHAPWaterfallPlot: React.FC<SHAPWaterfallPlotProps> = ({ 
  shapData, 
  baseProbability = 0.5, 
  finalProbability 
}) => {
  if (!shapData || shapData.length === 0) return null;

  // Prepare waterfall data
  const waterfallData: WaterfallData[] = [];
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
  sortedShapData.forEach((item) => {
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

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-card p-4 border rounded-lg shadow-lg max-w-xs">
          <p className="font-semibold text-foreground">{label}</p>
          {data.patientValue !== undefined && (
            <p className="text-sm text-muted-foreground">Patient Value: {data.patientValue}</p>
          )}
          {data.contribution !== 0 && (
            <p className={`text-sm font-medium ${data.contribution > 0 ? 'text-success' : 'text-destructive'}`}>
              Contribution: {data.displayValue}
            </p>
          )}
          <p className="text-sm text-primary">
            Cumulative: {(data.cumulative * 100).toFixed(1)}%
          </p>
          {data.type === 'positive' && (
            <p className="text-xs text-success mt-1">↑ Increases survival probability</p>
          )}
          {data.type === 'negative' && (
            <p className="text-xs text-destructive mt-1">↓ Decreases survival probability</p>
          )}
        </div>
      );
    }
    return null;
  };

  const getBarColor = (type: string) => {
    switch (type) {
      case 'base': return 'hsl(var(--muted-foreground))';
      case 'positive': return 'hsl(var(--success))';
      case 'negative': return 'hsl(var(--destructive))';
      case 'final': return 'hsl(var(--primary))';
      default: return 'hsl(var(--muted-foreground))';
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold flex items-center">
          <TrendingUp className="mr-2 text-primary" />
          SHAP Waterfall Plot
        </h3>
        <div className="text-sm text-muted-foreground">
          Base: {(baseProbability * 100).toFixed(1)}% → Final: {((finalProbability || cumulativeValue) * 100).toFixed(1)}%
        </div>
      </div>

      <p className="text-muted-foreground text-sm">
        This waterfall chart shows how each feature contributes step-by-step to the final prediction, 
        starting from the base survival rate.
      </p>

      <ResponsiveContainer width="100%" height={400}>
        <BarChart
          data={waterfallData}
          margin={{ top: 20, right: 30, left: 20, bottom: 80 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
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
          <ReferenceLine y={0.5} stroke="hsl(var(--muted-foreground))" strokeDasharray="5 5" />
          <Bar 
            dataKey="cumulative" 
            radius={[2, 2, 0, 0]}
          >
            {waterfallData.map((entry, index) => (
              <Bar key={`bar-${index}`} fill={getBarColor(entry.type)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Step-by-step breakdown */}
      <div className="bg-muted/50 rounded-lg p-4">
        <h4 className="font-semibold text-foreground mb-3">Step-by-Step Breakdown</h4>
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
                  <span className="text-muted-foreground ml-2">({item.patientValue})</span>
                )}
              </div>
              <div className="flex items-center">
                {item.contribution !== 0 && (
                  <>
                    <span className={`font-medium mr-2 ${item.contribution > 0 ? 'text-success' : 'text-destructive'}`}>
                      {item.displayValue}
                    </span>
                    <ArrowRight className="w-3 h-3 text-muted-foreground mr-2" />
                  </>
                )}
                <span className="font-semibold text-primary">
                  {(item.cumulative * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
        <div className="p-3 bg-muted/30 rounded-lg">
          <div className="text-lg font-bold text-muted-foreground">{(baseProbability * 100).toFixed(1)}%</div>
          <div className="text-sm text-muted-foreground">Starting Base Rate</div>
        </div>
        <div className="p-3 bg-primary/10 rounded-lg">
          <div className="text-lg font-bold text-primary">
            {((finalProbability || cumulativeValue) - baseProbability > 0 ? '+' : '')}
            {(((finalProbability || cumulativeValue) - baseProbability) * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-primary">Total SHAP Impact</div>
        </div>
        <div className="p-3 bg-primary/10 rounded-lg">
          <div className="text-lg font-bold text-primary">
            {((finalProbability || cumulativeValue) * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-primary">Final Prediction</div>
        </div>
      </div>
    </div>
  );
};

export default SHAPWaterfallPlot;