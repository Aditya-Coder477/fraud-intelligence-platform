"use client";

import { Explanation, FeatureContribution } from "@/lib/api";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Sparkles, TrendingUp, TrendingDown, AlertCircle, Loader2 } from "lucide-react";

interface ExplanationPanelProps {
  explanation: Explanation | null;
  loading?: boolean;
  accountId?: string;
}

function DirectionIcon({ direction }: { direction: string }) {
  return direction === "positive"
    ? <TrendingUp className="h-3.5 w-3.5 text-red-400 flex-shrink-0" />
    : <TrendingDown className="h-3.5 w-3.5 text-emerald-400 flex-shrink-0" />;
}

function FeatureBar({ feature, maxImportance }: { feature: FeatureContribution; maxImportance: number }) {
  const pct = maxImportance > 0 ? (feature.importance / maxImportance) * 100 : 0;
  const isPositive = feature.direction === "positive";
  const barColor = isPositive
    ? "from-red-500/80 to-rose-500"
    : "from-emerald-500/80 to-teal-500";

  return (
    <div className="group">
      <div className="flex items-start gap-2 mb-1">
        <DirectionIcon direction={feature.direction} />
        <div className="flex-1 min-w-0">
          <div className="flex justify-between items-center">
            <span className="text-xs font-mono text-zinc-300 truncate max-w-[180px]" title={feature.feature}>
              {feature.feature}
            </span>
            <div className="flex items-center gap-2 flex-shrink-0">
              {feature.pct_of_total != null && (
                <span className="text-[10px] text-muted-foreground">{feature.pct_of_total.toFixed(1)}%</span>
              )}
              <span className={`text-[10px] font-medium ${isPositive ? "text-red-400" : "text-emerald-400"}`}>
                {feature.importance.toFixed(3)}
              </span>
            </div>
          </div>
          {/* Progress bar */}
          <div className="w-full bg-slate-800 rounded-full h-1.5 mt-1">
            <div
              className={`h-1.5 rounded-full bg-gradient-to-r ${barColor} transition-all duration-500`}
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      </div>
      {/* Hover tooltip-like description */}
      <p className="text-[11px] text-muted-foreground pl-5 leading-snug mb-1 group-hover:text-zinc-400 transition-colors">
        {feature.description}
      </p>
    </div>
  );
}

export function ExplanationPanel({ explanation, loading, accountId }: ExplanationPanelProps) {
  if (loading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-purple-400" /> Why Was This Flagged?
          </CardTitle>
        </CardHeader>
        <CardContent className="flex items-center justify-center h-40">
          <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }

  if (!explanation) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-purple-400" /> Why Was This Flagged?
          </CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col items-center justify-center h-40 gap-2 text-muted-foreground">
          <AlertCircle className="h-6 w-6" />
          <p className="text-sm">No explanation available for this account.</p>
        </CardContent>
      </Card>
    );
  }

  const maxImportance = Math.max(...explanation.top_features.map(f => f.importance), 0.001);
  const positiveFeatures = explanation.top_features.filter(f => f.direction === "positive");
  const negativeFeatures = explanation.top_features.filter(f => f.direction === "negative");

  return (
    <div className="space-y-4">
      {/* ── Main explanation card ────────────────────────────────── */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center gap-2 text-lg">
            <Sparkles className="h-5 w-5 text-purple-400" />
            Why Was This Flagged?
          </CardTitle>
          <CardDescription>
            Top risk drivers from the ensemble model (SHAP feature importance)
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Legend */}
          <div className="flex gap-4 text-xs text-muted-foreground">
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-1.5 rounded-full bg-gradient-to-r from-red-500 to-rose-500 inline-block" />
              Increases risk
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-1.5 rounded-full bg-gradient-to-r from-emerald-500 to-teal-500 inline-block" />
              Reduces risk
            </span>
          </div>

          {/* Features */}
          <div className="space-y-2.5">
            {positiveFeatures.length > 0 && (
              <>
                <p className="text-[10px] font-semibold uppercase tracking-widest text-red-400/70">
                  Risk-Increasing Signals
                </p>
                {positiveFeatures.map(f => (
                  <FeatureBar key={f.feature} feature={f} maxImportance={maxImportance} />
                ))}
              </>
            )}

            {negativeFeatures.length > 0 && (
              <>
                <p className="text-[10px] font-semibold uppercase tracking-widest text-emerald-400/70 mt-3">
                  Mitigating Signals
                </p>
                {negativeFeatures.map(f => (
                  <FeatureBar key={f.feature} feature={f} maxImportance={maxImportance} />
                ))}
              </>
            )}
          </div>

          {/* Confidence */}
          {explanation.confidence != null && (
            <div className="flex justify-between text-xs text-muted-foreground border-t border-border/50 pt-3 mt-3">
              <span>Model Confidence</span>
              <span className="font-medium text-zinc-200">
                {(explanation.confidence * 100).toFixed(1)}% fraud probability
              </span>
            </div>
          )}
        </CardContent>
      </Card>

      {/* ── Narrative summary card ───────────────────────────────── */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Model Narrative</CardTitle>
          <CardDescription>Analyst-ready explanation generated from model outputs</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-sm leading-relaxed text-zinc-300 bg-slate-900/50 p-4 rounded-lg border border-slate-800 font-mono">
            {explanation.overall_summary || explanation.summary}
          </p>

          {/* Reason codes */}
          {explanation.reason_codes?.length > 0 && (
            <div>
              <p className="text-xs text-muted-foreground uppercase tracking-wide mb-2">Reason Codes</p>
              <div className="flex flex-wrap gap-1.5">
                {explanation.reason_codes.map(rc => (
                  <Badge key={rc} variant="secondary" className="text-[10px] font-mono">
                    {rc}
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
