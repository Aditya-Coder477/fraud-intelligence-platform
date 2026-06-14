"use client";

import { useEffect, useState } from "react";
import { api, Account } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { formatPercent } from "@/lib/utils";
import { Search, Filter, ArrowRight } from "lucide-react";
import Link from "next/link";

export default function AccountsPage() {
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Attempt to fetch from real API, fallback to dummy data for development visualization
    api.getAccounts()
      .then(res => setAccounts(res.data))
      .catch((err) => {
        console.warn("Backend not available, using development mock data.", err);
        setAccounts([
            { account_id: "ACC-8921", risk_score: 94, category: "BLOCK", status: "suspended", alert_count: 4, fraud_probability: 0.95, last_activity: "2026-06-06 14:32" },
            { account_id: "ACC-3342", risk_score: 88, category: "BLOCK", status: "active", alert_count: 2, fraud_probability: 0.89, last_activity: "2026-06-06 11:15" },
            { account_id: "ACC-1092", risk_score: 76, category: "REVIEW", status: "investigating", alert_count: 1, fraud_probability: 0.75, last_activity: "2026-06-05 09:44" },
            { account_id: "ACC-5521", risk_score: 65, category: "REVIEW", status: "active", alert_count: 3, fraud_probability: 0.68, last_activity: "2026-06-05 16:20" },
            { account_id: "ACC-9982", risk_score: 52, category: "MONITOR", status: "active", alert_count: 0, fraud_probability: 0.45, last_activity: "2026-06-04 10:05" },
            { account_id: "ACC-4411", risk_score: 41, category: "MONITOR", status: "active", alert_count: 1, fraud_probability: 0.38, last_activity: "2026-06-03 14:50" },
            { account_id: "ACC-2210", risk_score: 15, category: "SAFE", status: "active", alert_count: 0, fraud_probability: 0.05, last_activity: "2026-06-02 08:30" },
        ]);
      })
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Suspicious Accounts</h2>
          <p className="text-muted-foreground">Monitor and investigate flagged accounts.</p>
        </div>
        <Button variant="outline" className="gap-2">
          <Filter className="h-4 w-4" /> Filters
        </Button>
      </div>

      <Card>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Account ID</TableHead>
                <TableHead>Risk Score</TableHead>
                <TableHead>Category</TableHead>
                <TableHead>Fraud Prob.</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Alerts</TableHead>
                <TableHead>Last Activity</TableHead>
                <TableHead className="text-right">Action</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {loading ? (
                <TableRow>
                  <TableCell colSpan={8} className="text-center py-8 text-muted-foreground animate-pulse">Loading accounts...</TableCell>
                </TableRow>
              ) : accounts.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={8} className="text-center py-8 text-muted-foreground">No accounts found.</TableCell>
                </TableRow>
              ) : (
                accounts.map((account) => (
                  <TableRow key={account.account_id}>
                    <TableCell className="font-medium">{account.account_id}</TableCell>
                    <TableCell>
                        <span className={`font-bold ${account.risk_score >= 80 ? 'text-red-500' : account.risk_score >= 60 ? 'text-orange-500' : 'text-foreground'}`}>
                            {account.risk_score}
                        </span>
                    </TableCell>
                    <TableCell>
                      <Badge variant={account.category.toLowerCase() as any}>
                        {account.category}
                      </Badge>
                    </TableCell>
                    <TableCell>{formatPercent(account.fraud_probability * 100)}</TableCell>
                    <TableCell className="capitalize">{account.status}</TableCell>
                    <TableCell>{account.alert_count}</TableCell>
                    <TableCell className="text-muted-foreground">{account.last_activity}</TableCell>
                    <TableCell className="text-right">
                      <Link href={`/accounts/${account.account_id}`}>
                        <Button variant="ghost" size="sm" className="gap-1">
                          Investigate <ArrowRight className="h-4 w-4" />
                        </Button>
                      </Link>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
