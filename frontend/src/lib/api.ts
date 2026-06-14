import axios from 'axios';

const apiClient = axios.create({
  baseURL: '/api',
  headers: { 'Content-Type': 'application/json' },
});

export interface DashboardSummary {
  total_transactions: number;
  suspicious_accounts: number;
  high_risk_alerts: number;
  fraud_probability_distribution: { bin: string; count: number }[];
  trend_data: { date: string; flag_count: number }[];
  recent_flagged_cases: Account[];
}

export interface Account {
  account_id: string;
  risk_score: number;
  fraud_probability: number;
  alert_count: number;
  status: 'active' | 'suspended' | 'investigating' | 'closed';
  last_activity: string;
  category: 'BLOCK' | 'REVIEW' | 'MONITOR' | 'SAFE';
}

export interface AccountDetail extends Account {
  profile: any;
  explanation_summary: string;
}

export interface Transaction {
  transaction_id: string;
  date: string;
  amount: number;
  type: string;
  counterparty: string;
  status: string;
}

export interface FeatureContribution {
  feature: string;
  importance: number;
  direction: 'positive' | 'negative';  // positive = increases risk
  description: string;
  explanation_text: string;
  pct_of_total?: number;
}

export interface Explanation {
  top_features: FeatureContribution[];
  summary: string;
  overall_summary: string;
  reason_codes: string[];
  confidence?: number;
}

export interface RiskScore {
  risk_score: number;
  fraud_probability: number;
  anomaly_score: number;
  rules_score: number;
  score_band: string;
  risk_tags: string[];
  decision_recommendation: string;
}

export interface ContributingFactor {
  factor: string;
  value: string;
  weight: number;
}

export interface Alert {
  alert_id: string;
  account_id: string;
  transaction_id?: string;
  alert_type: 'MODEL_SCORE' | 'ANOMALY' | 'RULE' | 'CONVERGENT';
  date: string;
  severity: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  status: 'OPEN' | 'ACKNOWLEDGED' | 'ESCALATED' | 'RESOLVED';
  description: string;
  reason_codes: string[];
  contributing_factors: ContributingFactor[];
  recommended_action: string;
}

export interface AlertGenerateRequest {
  account_id: string;
  fraud_probability: number;
  anomaly_score: number;
  rules_score: number;
  risk_score: number;
  top_features?: any[];
}

export const api = {
  // Health
  getHealth: () => apiClient.get('/health').then(r => r.data),

  // Dashboard
  getDashboardSummary: (): Promise<DashboardSummary> =>
    apiClient.get('/dashboard/summary').then(r => r.data),

  // Accounts
  getAccounts: (params?: any): Promise<{ data: Account[]; total: number }> =>
    apiClient.get('/accounts', { params }).then(r => r.data),
  getAccount: (id: string): Promise<AccountDetail> =>
    apiClient.get(`/accounts/${id}`).then(r => r.data),

  // Transactions
  getTransactions: (id: string, params?: any): Promise<{ data: Transaction[]; total: number }> =>
    apiClient.get(`/accounts/${id}/transactions`, { params }).then(r => r.data),

  // Explainability
  getExplanations: (id: string): Promise<Explanation> =>
    apiClient.get(`/accounts/${id}/explanations`).then(r => r.data),

  // Risk Scoring
  getRiskScore: (id: string): Promise<RiskScore> =>
    apiClient.get(`/risk/${id}`).then(r => r.data),

  // Prediction
  predict: (payload: any): Promise<any> =>
    apiClient.post('/predict', payload).then(r => r.data),

  // Alerts
  getAlerts: (params?: any): Promise<{ data: Alert[]; total: number }> =>
    apiClient.get('/alerts', { params }).then(r => r.data),
  generateAlert: (payload: AlertGenerateRequest): Promise<Alert> =>
    apiClient.post('/alerts/generate', payload).then(r => r.data),
  acknowledgeAlert: (id: string): Promise<any> =>
    apiClient.post(`/alerts/${id}/acknowledge`).then(r => r.data),
  escalateAlert: (id: string): Promise<any> =>
    apiClient.post(`/alerts/${id}/escalate`).then(r => r.data),
  resolveAlert: (id: string): Promise<any> =>
    apiClient.post(`/alerts/${id}/resolve`).then(r => r.data),

  // Admin
  getModelInfo: (): Promise<any> => apiClient.get('/model/info').then(r => r.data),
};
