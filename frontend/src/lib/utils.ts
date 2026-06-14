import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatCurrency(value: number) {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
}

export function formatPercent(value: number) {
  return new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  }).format(value / 100);
}

export function getRiskColor(score: number): string {
    if (score >= 81) return "text-risk-block";
    if (score >= 61) return "text-risk-review";
    if (score >= 31) return "text-risk-monitor";
    return "text-risk-safe";
}

export function getRiskBgColor(score: number): string {
    if (score >= 81) return "bg-risk-block/20 text-risk-block";
    if (score >= 61) return "bg-risk-review/20 text-risk-review";
    if (score >= 31) return "bg-risk-monitor/20 text-risk-monitor";
    return "bg-risk-safe/20 text-risk-safe";
}
