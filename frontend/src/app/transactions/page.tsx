"use client";

import { useEffect, useState } from "react";
import { api, Transaction } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { formatCurrency } from "@/lib/utils";

export default function TransactionsPage() {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Attempt to fetch from real API, fallback to dummy data
    // Usually you'd fetch all suspicious transactions, for now using a mock account ID
    api.getTransactions('ALL')
      .then(res => setTransactions(res.data))
      .catch((err) => {
        console.warn("Backend not available, using development mock data.", err);
        setTransactions([
          { transaction_id: "TXN-882910", date: "2026-06-06 14:30:00", amount: 15000, type: "Wire Transfer", counterparty: "Global Finance Ltd", status: "PENDING" },
          { transaction_id: "TXN-882909", date: "2026-06-06 13:15:22", amount: 250, type: "Card Payment", counterparty: "Retail Store", status: "COMPLETED" },
          { transaction_id: "TXN-882908", date: "2026-06-06 11:05:10", amount: 8900, type: "ACH Transfer", counterparty: "Crypto Exchange", status: "FLAGGED" },
          { transaction_id: "TXN-882907", date: "2026-06-05 18:45:00", amount: 45000, type: "Wire Transfer", counterparty: "Offshore Account", status: "BLOCKED" },
          { transaction_id: "TXN-882906", date: "2026-06-05 09:20:00", amount: 1200, type: "P2P Transfer", counterparty: "John Doe", status: "COMPLETED" },
        ]);
      })
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-3xl font-bold tracking-tight">Transaction Monitoring</h2>
          <p className="text-muted-foreground">Monitor high-value and flagged transactions globally.</p>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-3 mb-6">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Daily Volume (Flagged)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">$142,500</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Blocked Transactions</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-500">12</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium">Pending Reviews</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-orange-500">45</div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Recent Transactions</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Transaction ID</TableHead>
                <TableHead>Date & Time</TableHead>
                <TableHead>Type</TableHead>
                <TableHead>Counterparty</TableHead>
                <TableHead className="text-right">Amount</TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {loading ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-8 text-muted-foreground animate-pulse">Loading transactions...</TableCell>
                </TableRow>
              ) : transactions.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-8 text-muted-foreground">No transactions found.</TableCell>
                </TableRow>
              ) : (
                transactions.map((txn) => (
                  <TableRow key={txn.transaction_id}>
                    <TableCell className="font-mono text-xs">{txn.transaction_id}</TableCell>
                    <TableCell className="text-sm">{txn.date}</TableCell>
                    <TableCell className="text-sm">{txn.type}</TableCell>
                    <TableCell className="font-medium">{txn.counterparty}</TableCell>
                    <TableCell className="text-right font-medium">{formatCurrency(txn.amount)}</TableCell>
                    <TableCell>
                      <Badge variant={
                          txn.status === 'BLOCKED' ? 'destructive' : 
                          txn.status === 'FLAGGED' ? 'outline' : 
                          txn.status === 'PENDING' ? 'secondary' : 'default'
                      } className={txn.status === 'FLAGGED' ? 'border-orange-500 text-orange-500' : ''}>
                        {txn.status}
                      </Badge>
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
