"use client";

import { useState } from "react";
import { Alert, ContributingFactor } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  ShieldAlert, AlertTriangle, Info, CheckCircle2,
  ChevronDown, ChevronUp, ExternalLink, Clock
} from "lucide-react";
import Link from "next/link";

interface AlertCardProps {
  alert: Alert;
  onAction: (id: string, action: 'acknowledge' | 'escalate' | 'resolve') => Promise<void>;
}

const SEVERITY_CONFIG = {
  CRITICAL: {
    badge: "border-red-500 bg-red-500/10 text-red-400",
    card: "border-red-500/30 shadow-red-500/10",
    icon: <ShieldAlert className="h-4 w-4 text-red-400" />,
    pulse: "bg-red-500",
  },
  HIGH: {
    badge: "border-orange-500 bg-orange-500/10 text-orange-400",
    card: "border-orange-500/20",
    icon: <AlertTriangle className="h-4 w-4 text-orange-400" />,
    pulse: "bg-orange-500",
  },
  MEDIUM: {
    badge: "border-yellow-500 bg-yellow-500/10 text-yellow-400",
    card: "border-yellow-500/20",
    icon: <AlertTriangle className="h-4 w-4 text-yellow-400" />,
    pulse: "bg-yellow-500",
  },
  LOW: {
    badge: "border-blue-500 bg-blue-500/10 text-blue-400",
    card: "border-blue-500/20",
    icon: <Info className="h-4 w-4 text-blue-400" />,
    pulse: "bg-blue-500",
  },
};

const STATUS_CONFIG = {
  OPEN:         { label: "Open",         class: "bg-slate-700 text-slate-200" },
  ACKNOWLEDGED: { label: "Acknowledged", class: "bg-blue-900/50 text-blue-300" },
  ESCALATED:    { label: "Escalated",    class: "bg-red-900/50 text-red-300" },
  RESOLVED:     { label: "Resolved",     class: "bg-green-900/50 text-green-300" },
};

const ALERT_TYPE_LABEL: Record<string, string> = {
  MODEL_SCORE: "ML Model",
  ANOMALY:     "Anomaly",
  RULE:        "Rule Engine",
  CONVERGENT:  "Convergent Evidence",
};

export function AlertCard({ alert, onAction }: AlertCardProps) {
  const [expanded, setExpanded] = useState(false);
  const [loading, setLoading] = useState<string | null>(null);

  const cfg = SEVERITY_CONFIG[alert.severity] || SEVERITY_CONFIG.LOW;
  const statusCfg = STATUS_CONFIG[alert.status] || STATUS_CONFIG.OPEN;

  const handle = async (action: 'acknowledge' | 'escalate' | 'resolve') => {
    setLoading(action);
    await onAction(alert.alert_id, action);
    setLoading(null);
  };

  const dateStr = alert.date
    ? new Date(alert.date).toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" })
    : "—";

  return (
    <Card className={`border ${cfg.card} transition-all duration-200 hover:shadow-md`}>
      <CardContent className="p-0">
        {/* ── Header Row ─────────────────────────────────────────── */}
        <div
          className="flex items-start gap-3 p-4 cursor-pointer select-none"
          onClick={() => setExpanded(!expanded)}
        >
          {/* Severity dot */}
          <div className="mt-1 flex-shrink-0 relative">
            <span className={`block w-2.5 h-2.5 rounded-full ${cfg.pulse}`} />
            {alert.status === "OPEN" && (
              <span className={`absolute inset-0 rounded-full ${cfg.pulse} animate-ping opacity-50`} />
            )}
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex flex-wrap items-center gap-2 mb-1">
              {/* Severity badge */}
              <Badge variant="outline" className={`text-xs px-2 py-0.5 ${cfg.badge}`}>
                {cfg.icon}
                <span className="ml-1">{alert.severity}</span>
              </Badge>
              {/* Alert type */}
              <span className="text-[10px] font-mono text-muted-foreground bg-slate-800 px-2 py-0.5 rounded-full">
                {ALERT_TYPE_LABEL[alert.alert_type] || alert.alert_type}
              </span>
              {/* Status */}
              <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium ${statusCfg.class}`}>
                {statusCfg.label}
              </span>
              <span className="text-xs text-muted-foreground font-mono ml-auto hidden sm:block">
                {alert.alert_id}
              </span>
            </div>

            {/* Description */}
            <p className="text-sm text-zinc-200 leading-snug line-clamp-2">{alert.description}</p>

            <div className="flex items-center gap-3 mt-1.5 text-xs text-muted-foreground">
              <Link
                href={`/accounts/${alert.account_id}`}
                onClick={e => e.stopPropagation()}
                className="text-blue-400 hover:text-blue-300 flex items-center gap-1"
              >
                {alert.account_id} <ExternalLink className="h-3 w-3" />
              </Link>
              <span className="flex items-center gap-1">
                <Clock className="h-3 w-3" /> {dateStr}
              </span>
            </div>
          </div>

          <button className="text-muted-foreground flex-shrink-0 mt-1">
            {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
          </button>
        </div>

        {/* ── Expanded Detail ─────────────────────────────────────── */}
        {expanded && (
          <div className="border-t border-border/50 px-4 pb-4 pt-3 space-y-4 animate-in slide-in-from-top-2 duration-200">
            {/* Contributing factors */}
            {alert.contributing_factors?.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                  Contributing Factors
                </p>
                <div className="space-y-2">
                  {alert.contributing_factors.map((f: ContributingFactor, i: number) => (
                    <div key={i} className="flex items-center gap-3">
                      <div className="flex-1">
                        <div className="flex justify-between text-xs mb-0.5">
                          <span className="text-zinc-300 font-medium">{f.factor}</span>
                          <span className="text-muted-foreground font-mono">{f.value}</span>
                        </div>
                        <div className="w-full bg-slate-800 rounded-full h-1.5">
                          <div
                            className="h-1.5 rounded-full bg-gradient-to-r from-orange-500 to-red-500 transition-all"
                            style={{ width: `${Math.min(f.weight * 100, 100)}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Reason codes */}
            {alert.reason_codes?.length > 0 && (
              <div>
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-2">
                  Reason Codes
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {alert.reason_codes.map((rc: string, i: number) => (
                    <Badge key={i} variant="secondary" className="text-[10px] font-mono">
                      {rc}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {/* Recommended action */}
            {alert.recommended_action && (
              <div className="p-3 bg-slate-900/60 border border-slate-700/50 rounded-lg">
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">
                  Recommended Action
                </p>
                <p className="text-sm text-zinc-200">{alert.recommended_action}</p>
              </div>
            )}

            {/* Action buttons */}
            <div className="flex gap-2 pt-1">
              {alert.status === "OPEN" && (
                <Button
                  size="sm"
                  variant="outline"
                  disabled={loading === "acknowledge"}
                  onClick={() => handle("acknowledge")}
                  className="text-blue-400 border-blue-500/40 hover:bg-blue-500/10"
                >
                  <CheckCircle2 className="h-3.5 w-3.5 mr-1" />
                  {loading === "acknowledge" ? "..." : "Acknowledge"}
                </Button>
              )}
              {(alert.status === "OPEN" || alert.status === "ACKNOWLEDGED") && (
                <Button
                  size="sm"
                  variant="destructive"
                  disabled={loading === "escalate"}
                  onClick={() => handle("escalate")}
                >
                  <ShieldAlert className="h-3.5 w-3.5 mr-1" />
                  {loading === "escalate" ? "..." : "Escalate"}
                </Button>
              )}
              {alert.status !== "RESOLVED" && (
                <Button
                  size="sm"
                  variant="secondary"
                  disabled={loading === "resolve"}
                  onClick={() => handle("resolve")}
                >
                  {loading === "resolve" ? "..." : "Resolve"}
                </Button>
              )}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
