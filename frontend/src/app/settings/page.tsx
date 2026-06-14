"use client";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";

export default function SettingsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold tracking-tight">Settings</h2>
        <p className="text-muted-foreground">Manage your dashboard preferences and system configurations.</p>
      </div>

      <div className="grid gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Model Configuration</CardTitle>
            <CardDescription>Adjust thresholds for the Fraud Detection model.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex justify-between items-center border-b border-border pb-4">
              <div>
                <p className="font-medium">Auto-Block Threshold</p>
                <p className="text-sm text-muted-foreground">Risk scores above this will automatically block accounts.</p>
              </div>
              <div className="font-bold text-red-500 text-xl">80</div>
            </div>
            <div className="flex justify-between items-center border-b border-border pb-4">
              <div>
                <p className="font-medium">Manual Review Threshold</p>
                <p className="text-sm text-muted-foreground">Risk scores above this will trigger a review alert.</p>
              </div>
              <div className="font-bold text-orange-500 text-xl">60</div>
            </div>
            <Button variant="outline" className="mt-4">Update Thresholds</Button>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>API Integration</CardTitle>
            <CardDescription>Backend connection settings.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <p className="font-medium mb-1">Backend URL</p>
              <code className="text-sm bg-muted px-2 py-1 rounded">http://localhost:8000/api</code>
            </div>
            <div>
              <p className="font-medium mb-1">API Status</p>
              <div className="flex items-center gap-2 text-sm text-green-500">
                <div className="w-2 h-2 rounded-full bg-green-500"></div> Connected
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
