"use client";

import { useEffect, useState } from "react";
import { api, DashboardSummary } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Activity, AlertTriangle, ShieldAlert, Users } from "lucide-react";
import { formatCurrency } from "@/lib/utils";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, LineChart, Line } from "recharts";
import { Badge } from "@/components/ui/badge";

export default function DashboardPage() {
  const [data, setData] = useState<DashboardSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Attempt to fetch from real API, fallback to dummy data for development visualization
    api.getDashboardSummary()
      .then(setData)
      .catch((err) => {
        console.warn("Backend not available, using development mock data.", err);
        // Development Mock Data
        setData({
          total_transactions: 1254300,
          suspicious_accounts: 342,
          high_risk_alerts: 89,
          fraud_probability_distribution: [
            { bin: "0-10%", count: 8000 },
            { bin: "10-30%", count: 1200 },
            { bin: "30-60%", count: 400 },
            { bin: "60-80%", count: 150 },
            { bin: "80-100%", count: 85 },
          ],
          trend_data: [
            { date: "01/01", flag_count: 12 },
            { date: "01/02", flag_count: 15 },
            { date: "01/03", flag_count: 9 },
            { date: "01/04", flag_count: 22 },
            { date: "01/05", flag_count: 18 },
            { date: "01/06", flag_count: 31 },
            { date: "01/07", flag_count: 28 },
          ],
          recent_flagged_cases: [
            { account_id: "ACC-8921", risk_score: 94, category: "BLOCK", status: "suspended", alert_count: 4, fraud_probability: 0.95, last_activity: "2026-06-06" },
            { account_id: "ACC-3342", risk_score: 88, category: "BLOCK", status: "active", alert_count: 2, fraud_probability: 0.89, last_activity: "2026-06-06" },
            { account_id: "ACC-1092", risk_score: 76, category: "REVIEW", status: "investigating", alert_count: 1, fraud_probability: 0.75, last_activity: "2026-06-05" },
            { account_id: "ACC-5521", risk_score: 65, category: "REVIEW", status: "active", alert_count: 3, fraud_probability: 0.68, last_activity: "2026-06-05" },
          ]
        });
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8 text-center text-muted-foreground animate-pulse">Loading intelligence data...</div>;
  if (!data) return <div className="p-8 text-center text-destructive">Failed to load dashboard data.</div>;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Executive Overview</h2>
        <p className="text-muted-foreground">Real-time fraud intelligence and system performance.</p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Transactions</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{data.total_transactions.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">+20.1% from last month</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Suspicious Accounts</CardTitle>
            <Users className="h-4 w-4 text-orange-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{data.suspicious_accounts.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">Accounts flagged across all tiers</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Critical Alerts</CardTitle>
            <ShieldAlert className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{data.high_risk_alerts}</div>
            <p className="text-xs text-red-500 font-medium">Requires immediate action</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">System Status</CardTitle>
            <Activity className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-500">Operational</div>
            <p className="text-xs text-muted-foreground">Model latency: 45ms</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
        <Card className="col-span-4">
          <CardHeader>
            <CardTitle>Suspicious Activity Trend (7 Days)</CardTitle>
          </CardHeader>
          <CardContent className="pl-2">
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data.trend_data}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                  <XAxis dataKey="date" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
                  <YAxis stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
                  <RechartsTooltip 
                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }}
                    itemStyle={{ color: '#38bdf8' }}
                  />
                  <Line type="monotone" dataKey="flag_count" stroke="#38bdf8" strokeWidth={3} dot={{ r: 4, fill: '#38bdf8' }} activeDot={{ r: 6 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
        <Card className="col-span-3">
          <CardHeader>
            <CardTitle>Fraud Probability Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data.fraud_probability_distribution} layout="vertical" margin={{ top: 0, right: 0, bottom: 0, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#333" horizontal={true} vertical={false} />
                  <XAxis type="number" stroke="#888888" fontSize={12} hide />
                  <YAxis dataKey="bin" type="category" stroke="#888888" fontSize={12} tickLine={false} axisLine={false} />
                  <RechartsTooltip cursor={{fill: '#1e293b'}} contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b' }} />
                  <Bar dataKey="count" fill="#8b5cf6" radius={[0, 4, 4, 0]} barSize={20} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Recent Priority Cases</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {data.recent_flagged_cases.map((account) => (
              <div key={account.account_id} className="flex items-center justify-between p-4 border rounded-lg bg-card/50 hover:bg-muted/50 transition-colors">
                <div className="flex items-center gap-4">
                  <div className={`p-2 rounded-full ${account.category === 'BLOCK' ? 'bg-red-500/20 text-red-500' : 'bg-orange-500/20 text-orange-500'}`}>
                    <AlertTriangle className="h-5 w-5" />
                  </div>
                  <div>
                    <p className="font-medium">{account.account_id}</p>
                    <p className="text-sm text-muted-foreground">Last activity: {account.last_activity}</p>
                  </div>
                </div>
                <div className="flex items-center gap-6">
                    <div className="text-right hidden sm:block">
                        <p className="text-sm text-muted-foreground">Alerts</p>
                        <p className="font-medium">{account.alert_count}</p>
                    </div>
                    <div className="text-right">
                        <p className="text-sm text-muted-foreground">Risk Score</p>
                        <p className={`font-bold text-lg ${account.risk_score >= 80 ? 'text-red-500' : 'text-orange-500'}`}>{account.risk_score}</p>
                    </div>
                  <Badge variant={account.category.toLowerCase() as any} className="ml-4 w-24 justify-center">
                    {account.category}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
