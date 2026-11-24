// Dimple, Upload Date: 2026-08-02
// Production API Client for AcousticSpace Frontend — with Task-based Submit
import type { AnalysisResponse, SubmitResponse } from "../types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export interface DemoSample {
  id: string;
  name: string;
  description: string;
  target_class: string;
}

export interface ModelMetricsData {
  created_at_utc: string;
  dataset_rows: number;
  train_rows: number;
  test_rows: number;
  feature_count: number;
  best_model: string;
  holdout_accuracy: number;
  holdout_balanced_accuracy: number;
  holdout_f1_macro: number;
  voice_binary_accuracy: number;
  training_seconds: number;
  comparison: Record<string, { cv_accuracy_mean: number; cv_f1_macro_mean: number; cv_seconds: number }>;
  confusion_matrix: number[][];
  labels: string[];
}

export interface EdaSummaryData {
  total_files: number;
  duplicate_paths: number;
  class_counts: Record<string, number>;
  mix_profile_counts: Record<string, number>;
  unique_utterances: number;
  unique_backgrounds: number;
}

/** Synchronous analysis (kept for backward compatibility). */
export async function analyzeAudio(file: File): Promise<AnalysisResponse> {
  const body = new FormData();
  body.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/analyze`, {
    method: "POST",
    body,
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? "Analysis failed. Please try another audio file.");
  }

  return response.json();
}

/**
 * Submit audio for async task-based analysis.
 * Returns task_id — caller should connect to WS /ws/{task_id} for progress.
 */
export async function submitAnalysis(file: File): Promise<SubmitResponse> {
  const body = new FormData();
  body.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/analyze/submit`, {
    method: "POST",
    body,
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail ?? "Submission failed. Please try another audio file.");
  }

  return response.json();
}

export async function getHealth(): Promise<{ status: string; model_type?: string; artifact_loaded?: boolean }> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) return { status: "offline" };
    return await response.json();
  } catch {
    return { status: "offline" };
  }
}

const FALLBACK_METRICS: ModelMetricsData = {
  created_at_utc: "2026-08-17T18:16:18.271586+00:00",
  dataset_rows: 12000,
  train_rows: 9600,
  test_rows: 2400,
  feature_count: 118,
  best_model: "dual_head_ensemble",
  holdout_accuracy: 0.4325,
  holdout_balanced_accuracy: 0.4325,
  holdout_f1_macro: 0.4318,
  voice_binary_accuracy: 0.5058,
  training_seconds: 40.66,
  comparison: {
    dual_head_ensemble: { cv_accuracy_mean: 0.4325, cv_f1_macro_mean: 0.4318, cv_seconds: 12.5 },
    logistic_regression: { cv_accuracy_mean: 0.4325, cv_f1_macro_mean: 0.4318, cv_seconds: 5.2 },
    extra_trees: { cv_accuracy_mean: 0.3792, cv_f1_macro_mean: 0.3778, cv_seconds: 8.4 },
  },
  confusion_matrix: [
    [255, 112, 168, 65],
    [95, 289, 56, 160],
    [173, 69, 227, 131],
    [56, 179, 98, 267],
  ],
  labels: ["fake_voice_fake_bg", "fake_voice_real_bg", "real_voice_fake_bg", "real_voice_real_bg"],
};

const FALLBACK_EDA: EdaSummaryData = {
  total_files: 12000,
  duplicate_paths: 0,
  class_counts: {
    fake_voice_fake_bg: 3000,
    fake_voice_real_bg: 3000,
    real_voice_fake_bg: 3000,
    real_voice_real_bg: 3000,
  },
  mix_profile_counts: {
    balanced: 4000,
    bg_dominant: 4000,
    voice_dominant: 4000,
  },
  unique_utterances: 2400,
  unique_backgrounds: 800,
};

export async function getModelMetrics(): Promise<ModelMetricsData | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/metrics`);
    if (!response.ok) return FALLBACK_METRICS;
    return await response.json();
  } catch {
    return FALLBACK_METRICS;
  }
}

export async function getEdaSummary(): Promise<EdaSummaryData | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/eda-summary`);
    if (!response.ok) return FALLBACK_EDA;
    return await response.json();
  } catch {
    return FALLBACK_EDA;
  }
}

export async function getDemoSamples(): Promise<DemoSample[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/demo-samples`);
    if (!response.ok) return [];
    return await response.json();
  } catch {
    return [];
  }
}

export async function fetchDemoSampleFile(sampleId: string): Promise<File> {
  const response = await fetch(`${API_BASE_URL}/api/demo-samples/${sampleId}`);
  if (!response.ok) {
    throw new Error(`Failed to load demo sample '${sampleId}'.`);
  }
  const blob = await response.blob();
  return new File([blob], `${sampleId}.wav`, { type: "audio/wav" });
}
