"use client";

import { useEffect, useState, useCallback } from "react";
import { api, Alert } from "@/lib/api";
import { AlertCard } from "@/components/alerts/AlertCard";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  ShieldAlert, AlertTriangle, CheckCircle2, Filter,
  RefreshCw, Activity, TrendingUp, Bell
} from "lucide-react";

// ── Stats bar data helper ──────────────────────────────────────────────────
function getStats(alerts: Alert[]) {
  return {
    total: alerts.length,
    critical: alerts.filter(a => a.severity === "CRITICAL").length,
    open: alerts.filter(a => a.status === "OPEN").length,
    escalated: alerts.filter(a => a.status === "ESCALATED").length,
  };
}

const SEVERITIES = ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"] as const;
const STATUSES   = ["ALL", "OPEN", "ACKNOWLEDGED", "ESCALATED", "RESOLVED"] as const;

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [severityFilter, setSeverityFilter] = useState<string>("ALL");
  const [statusFilter, setStatusFilter]     = useState<string>("ALL");
  const [search, setSearch]                 = useState("");

  const fetchAlerts = useCallback(async (silent = false) => {
    if (!silent) setLoading(true);
    else setRefreshing(true);

    try {
      const res = await api.getAlerts({
        severity: severityFilter !== "ALL" ? severityFilter : undefined,
        status: statusFilter !== "ALL" ? statusFilter : undefined,
        search: search || undefined,
        page_size: 100,
      });
      setAlerts(res.data);
    } catch {
      // Fallback mock data so the page is still useful during development
      setAlerts([
        {
          alert_id: "ALT-A1B2C3", account_id: "ACC-8921",
          alert_type: "CONVERGENT", date: new Date().toISOString(),
          severity: "CRITICAL", status: "OPEN",
          description: "Convergent evidence from 3 detection systems: ML fraud probability (94%), anomaly score (88/100), rule engine score (87/100). Primary drivers: near-zero net balance; high transaction velocity; new account with suspicious activity.",
          reason_codes: ["R04: Convergent Evidence", "R01: ML Model — High Fraud Probability", "R02: Anomaly Detection", "R06: Pass-Through Pattern"],
          contributing_factors: [
            { factor: "ML Fraud Probability", value: "94.0%", weight: 0.94 },
            { factor: "Anomaly Detection Score", value: "88.0/100", weight: 0.88 },
            { factor: "Rule-Based Alert Score", value: "87.0/100", weight: 0.87 },
          ],
          recommended_action: "Immediately block account, freeze pending transactions, file Suspicious Activity Report (SAR), and escalate to senior AML officer.",
        },
        {
          alert_id: "ALT-D4E5F6", account_id: "ACC-3342",
          alert_type: "MODEL_SCORE", date: new Date(Date.now() - 3600000).toISOString(),
          severity: "HIGH", status: "ESCALATED",
          description: "ML ensemble model assigned a fraud probability of 78% to this account. Primary drivers: erratic transaction distribution; pass-through balance pattern.",
          reason_codes: ["R01: ML Model — High Fraud Probability", "R07: Transaction Velocity"],
          contributing_factors: [
            { factor: "ML Fraud Probability", value: "78.0%", weight: 0.78 },
            { factor: "Anomaly Detection Score", value: "62.0/100", weight: 0.62 },
          ],
          recommended_action: "Place account on enhanced monitoring hold. Analyst review required within 4 hours.",
        },
        {
          alert_id: "ALT-G7H8I9", account_id: "ACC-1092",
          alert_type: "ANOMALY", date: new Date(Date.now() - 86400000).toISOString(),
          severity: "MEDIUM", status: "ACKNOWLEDGED",
          description: "Anomaly detection flagged this account with a score of 67/100. Primary drivers: unusual channel access pattern.",
          reason_codes: ["R02: Anomaly Detection — Statistical Outlier"],
          contributing_factors: [
            { factor: "Anomaly Detection Score", value: "67.0/100", weight: 0.67 },
          ],
          recommended_action: "Add to enhanced monitoring queue. Schedule analyst review within 24 hours.",
        },
      ]);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [severityFilter, statusFilter, search]);

  useEffect(() => { fetchAlerts(); }, [fetchAlerts]);

  const handleAction = async (id: string, action: 'acknowledge' | 'escalate' | 'resolve') => {
    const statusMap = {
      acknowledge: "ACKNOWLEDGED",
      escalate: "ESCALATED",
      resolve: "RESOLVED",
    } as const;

    // Optimistic update
    setAlerts(prev =>
      prev.map(a => a.alert_id === id ? { ...a, status: statusMap[action] } : a)
    );

    try {
      if (action === "acknowledge") await api.acknowledgeAlert(id);
      else if (action === "escalate")  await api.escalateAlert(id);
      else                             await api.resolveAlert(id);
    } catch {
      // Revert on failure — refetch
      fetchAlerts(true);
    }
  };

  const stats = getStats(alerts);

  return (
    <div className="space-y-6">
      {/* ── Page header ─────────────────────────────────────────── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-3xl font-bold tracking-tight flex items-center gap-2">
            <Bell className="h-7 w-7 text-orange-400" /> Alerts Center
          </h2>
          <p className="text-muted-foreground mt-0.5">
            Real-time fraud intelligence alerts — triage, escalate, and resolve
          </p>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={() => fetchAlerts(true)}
          disabled={refreshing}
          className="flex items-center gap-2"
        >
          <RefreshCw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {/* ── Stats bar ───────────────────────────────────────────── */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        {[
          { label: "Total Alerts",      value: stats.total,    icon: <Activity className="h-5 w-5 text-slate-400" />,   color: "text-white" },
          { label: "Critical",          value: stats.critical,  icon: <ShieldAlert className="h-5 w-5 text-red-400" />,  color: "text-red-400" },
          { label: "Open",              value: stats.open,      icon: <AlertTriangle className="h-5 w-5 text-orange-400" />, color: "text-orange-400" },
          { label: "Escalated",         value: stats.escalated, icon: <TrendingUp className="h-5 w-5 text-purple-400" />, color: "text-purple-400" },
        ].map(s => (
          <Card key={s.label} className="border-slate-800">
            <CardContent className="flex items-center gap-3 p-4">
              {s.icon}
              <div>
                <p className={`text-2xl font-bold ${s.color}`}>{s.value}</p>
                <p className="text-xs text-muted-foreground">{s.label}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* ── Filters ─────────────────────────────────────────────── */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <input
            type="text"
            placeholder="Search by account ID…"
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="w-full bg-slate-900 border border-slate-700 rounded-md px-3 py-1.5 text-sm text-zinc-200 placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-slate-500"
          />
        </div>
        <div className="flex gap-1.5 flex-wrap">
          {SEVERITIES.map(s => (
            <button
              key={s}
              onClick={() => setSeverityFilter(s)}
              className={`text-xs px-3 py-1.5 rounded-md border transition-colors ${
                severityFilter === s
                  ? "bg-slate-700 border-slate-500 text-white"
                  : "border-slate-700 text-muted-foreground hover:border-slate-500"
              }`}
            >
              {s}
            </button>
          ))}
        </div>
        <div className="flex gap-1.5 flex-wrap">
          {STATUSES.map(s => (
            <button
              key={s}
              onClick={() => setStatusFilter(s)}
              className={`text-xs px-3 py-1.5 rounded-md border transition-colors ${
                statusFilter === s
                  ? "bg-slate-700 border-slate-500 text-white"
                  : "border-slate-700 text-muted-foreground hover:border-slate-500"
              }`}
            >
              {s === "ALL" ? "All Status" : s.charAt(0) + s.slice(1).toLowerCase()}
            </button>
          ))}
        </div>
      </div>

      {/* ── Alert list ──────────────────────────────────────────── */}
      <div className="space-y-3">
        {loading ? (
          Array.from({ length: 3 }).map((_, i) => (
            <Card key={i} className="border-slate-800 animate-pulse">
              <CardContent className="p-4 h-20 flex items-center gap-3">
                <div className="w-2.5 h-2.5 rounded-full bg-slate-700" />
                <div className="flex-1 space-y-2">
                  <div className="h-3 bg-slate-700 rounded w-1/3" />
                  <div className="h-3 bg-slate-800 rounded w-2/3" />
                </div>
              </CardContent>
            </Card>
          ))
        ) : alerts.length === 0 ? (
          <Card className="border-slate-800">
            <CardContent className="flex flex-col items-center justify-center py-16 text-muted-foreground gap-2">
              <CheckCircle2 className="h-8 w-8 text-green-500/50" />
              <p className="text-sm">No alerts match your current filters.</p>
            </CardContent>
          </Card>
        ) : (
          alerts.map(alert => (
            <AlertCard key={alert.alert_id} alert={alert} onAction={handleAction} />
          ))
        )}
      </div>
    </div>
  );
}
