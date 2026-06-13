import type { AnalyzeResponse } from "./types";

export async function analyzeAudio(file: File): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append("audio", file);

  const res = await fetch("/analyze", {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Request failed (${res.status})`);
  }

  return res.json() as Promise<AnalyzeResponse>;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch("/health");
    if (!res.ok) return false;
    const data = await res.json();
    return data.status === "ok";
  } catch {
    return false;
  }
}
