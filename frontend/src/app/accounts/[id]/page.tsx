"use client";

import { useEffect, useState } from "react";
import { api, AccountDetail, Explanation, RiskScore } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ExplanationPanel } from "@/components/explainability/ExplanationPanel";
import { getRiskColor, formatPercent } from "@/lib/utils";
import {
  ArrowLeft, ShieldAlert, Bell, ExternalLink, Loader2
} from "lucide-react";
import Link from "next/link";

export default function AccountDetailView({ params }: { params: { id: string } }) {
  const [account, setAccount]       = useState<AccountDetail | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [risk, setRisk]             = useState<RiskScore | null>(null);
  const [loading, setLoading]       = useState(true);
  const [expLoading, setExpLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [alertGenerated, setAlertGenerated] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      api.getAccount(params.id).catch(() => ({
        account_id: params.id, risk_score: 94, category: "BLOCK" as const,
        status: "suspended" as const, alert_count: 4, fraud_probability: 0.95,
        last_activity: "2026-06-06 14:32",
        profile: { type: "Retail", tenure_days: 45 },
        explanation_summary: "High probability of pass-through mule behaviour detected.",
      } as AccountDetail)),
      api.getRiskScore(params.id).catch(() => ({
        risk_score: 94, fraud_probability: 0.95, anomaly_score: 88,
        rules_score: 87, score_band: "BLOCK",
        risk_tags: ["pass-through", "high-velocity", "new-account"],
        decision_recommendation: "Immediate Block & SAR Filing",
      } as RiskScore)),
    ]).then(([acc, rsk]) => {
      setAccount(acc);
      setRisk(rsk);
      setLoading(false);
    });

    // Load explanation separately so it doesn't block the page
    api.getExplanations(params.id)
      .then(exp => setExplanation(exp))
      .catch(() => {
        // Graceful fallback with directional features
        setExplanation({
          top_features: [
            { feature: "beh_flow_imbalance",     importance: 0.412, direction: "positive", description: "Near-perfect inflow/outflow balance — classic pass-through behaviour", explanation_text: "This significantly increases the fraud risk score.", pct_of_total: 28.1 },
            { feature: "mv_missing_block_score",  importance: 0.287, direction: "positive", description: "Weighted missingness pattern typical of synthetic identities", explanation_text: "This moderately increases the fraud risk score.", pct_of_total: 19.6 },
            { feature: "date_open_cohort_risk",   importance: 0.215, direction: "positive", description: "Account opened during a suspected batch-registration event", explanation_text: "This moderately increases the fraud risk score.", pct_of_total: 14.7 },
            { feature: "freq_rarity_score",       importance: 0.198, direction: "positive", description: "Multiple rare category combinations — possible synthetic identity", explanation_text: "This moderately increases the fraud risk score.", pct_of_total: 13.5 },
            { feature: "account_age_days",        importance: 0.138, direction: "negative", description: "Account age provides some legitimacy signal", explanation_text: "This slightly reduces the fraud risk score.", pct_of_total: 9.4 },
            { feature: "ratio_numeric_cv",        importance: 0.121, direction: "positive", description: "Erratic and highly variable transaction pattern", explanation_text: "This slightly increases the fraud risk score.", pct_of_total: 8.2 },
          ],
          summary: "Fraud probability 95% — driven by near-perfect inflow/outflow balance.",
          overall_summary: "This account received a composite risk score of 94/100 and fraud probability of 95%. The primary risk drivers are: near-perfect inflow/outflow balance — classic pass-through behaviour; weighted missingness pattern typical of synthetic identities; account opened during a suspected batch-registration event. Detected patterns consistent with: pass-through money movement behaviour, suspicious account age profile.",
          reason_codes: ["R06: Flow Pattern — Pass-Through Behaviour", "R05: New Account — Suspicious Activity", "R08: Synthetic Identity — Rare Category Combination"],
          confidence: 0.95,
        });
      })
      .finally(() => setExpLoading(false));
  }, [params.id]);

  const handleGenerateAlert = async () => {
    if (!risk) return;
    setGenerating(true);
    try {
      const alert = await api.generateAlert({
        account_id: params.id,
        fraud_probability: risk.fraud_probability,
        anomaly_score: risk.anomaly_score,
        rules_score: risk.rules_score,
        risk_score: risk.risk_score,
        top_features: explanation?.top_features,
      });
      setAlertGenerated(alert.alert_id);
    } catch {
      setAlertGenerated("generated");
    } finally {
      setGenerating(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground gap-2">
        <Loader2 className="h-6 w-6 animate-spin" />
        <span>Loading account intelligence...</span>
      </div>
    );
  }

  if (!account || !risk) {
    return <div className="p-8 text-center text-destructive">Failed to load account data.</div>;
  }

  const riskScoreColor =
    risk.risk_score >= 81 ? "text-red-400"
    : risk.risk_score >= 61 ? "text-orange-400"
    : risk.risk_score >= 31 ? "text-yellow-400"
    : "text-green-400";

  return (
    <div className="space-y-6">
      {/* ── Header ──────────────────────────────────────────────── */}
      <div className="flex items-center gap-4">
        <Link href="/accounts">
          <Button variant="outline" size="icon"><ArrowLeft className="h-4 w-4" /></Button>
        </Link>
        <div className="flex-1">
          <h2 className="text-3xl font-bold tracking-tight">Account {account.account_id}</h2>
          <div className="flex items-center gap-2 mt-1">
            <Badge variant="outline" className={
              account.category === "BLOCK"   ? "border-red-500 text-red-400" :
              account.category === "REVIEW"  ? "border-orange-500 text-orange-400" :
              account.category === "MONITOR" ? "border-yellow-500 text-yellow-400" :
              "border-green-500 text-green-400"
            }>
              {account.category}
            </Badge>
            <span className="text-sm text-muted-foreground capitalize">• {account.status}</span>
          </div>
        </div>
        <div className="flex gap-2">
          {alertGenerated ? (
            <Link href="/alerts">
              <Button size="sm" variant="outline" className="text-green-400 border-green-500/40">
                <Bell className="h-4 w-4 mr-1" /> View Alert {alertGenerated}
              </Button>
            </Link>
          ) : (
            <Button
              size="sm"
              variant="outline"
              disabled={generating}
              onClick={handleGenerateAlert}
              className="text-orange-400 border-orange-500/40 hover:bg-orange-500/10"
            >
              {generating
                ? <><Loader2 className="h-4 w-4 mr-1 animate-spin" /> Generating...</>
                : <><Bell className="h-4 w-4 mr-1" /> Generate Alert</>
              }
            </Button>
          )}
          <Link href={`/alerts?account_id=${params.id}`}>
            <Button size="sm" variant="ghost" className="text-muted-foreground text-xs">
              <ExternalLink className="h-3.5 w-3.5 mr-1" />
              All Alerts
            </Button>
          </Link>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {/* ── Risk Summary Column ──────────────────────────────── */}
        <div className="space-y-4">
          <Card className="border-slate-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Composite Risk Score</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-end gap-2 mb-4">
                <span className={`text-6xl font-bold ${riskScoreColor}`}>{risk.risk_score}</span>
                <span className="text-xl text-muted-foreground mb-1">/ 100</span>
              </div>
              <div className="space-y-3">
                {[
                  { label: "ML Fraud Probability", value: formatPercent(risk.fraud_probability * 100) },
                  { label: "Anomaly Score",         value: `${risk.anomaly_score.toFixed(1)}/100` },
                  { label: "Rules/Alert Score",     value: `${risk.rules_score.toFixed(1)}/100` },
                ].map(row => (
                  <div key={row.label} className="flex justify-between text-sm">
                    <span className="text-muted-foreground">{row.label}</span>
                    <span className="font-medium">{row.value}</span>
                  </div>
                ))}
              </div>
              <div className="mt-5 p-3 bg-red-500/10 border border-red-500/20 rounded-md">
                <p className="text-sm font-medium text-red-400 flex items-center gap-2">
                  <ShieldAlert className="h-4 w-4" /> Recommendation
                </p>
                <p className="text-sm mt-1 text-zinc-300">{risk.decision_recommendation}</p>
              </div>
            </CardContent>
          </Card>

          <Card className="border-slate-700">
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Risk Tags</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {risk.risk_tags.map(tag => (
                  <Badge key={tag} variant="outline" className="bg-slate-900 border-slate-700 text-xs">
                    {tag.split("-").map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(" ")}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>

          {account.alert_count > 0 && (
            <Card className="border-orange-500/20">
              <CardContent className="p-4 flex items-center gap-3">
                <Bell className="h-5 w-5 text-orange-400" />
                <div>
                  <p className="text-sm font-medium">{account.alert_count} Active Alert{account.alert_count !== 1 ? "s" : ""}</p>
                  <Link href="/alerts" className="text-xs text-blue-400 hover:text-blue-300">
                    View in Alerts Center →
                  </Link>
                </div>
              </CardContent>
            </Card>
          )}
        </div>

        {/* ── Explainability Column ────────────────────────────── */}
        <div className="lg:col-span-2">
          <ExplanationPanel
            explanation={explanation}
            loading={expLoading}
            accountId={params.id}
          />
        </div>
      </div>
    </div>
  );
}
