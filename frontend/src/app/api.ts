/**
 * API helpers — use 127.0.0.1 on Windows to avoid localhost → ::1 connection issues.
 */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:7860";

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export async function checkBackendHealth(): Promise<boolean> {
  const urls = [`${API_BASE}/health`, "/health"];
  for (const url of urls) {
    try {
      const response = await fetch(url, { cache: "no-store" });
      if (response.ok) {
        const body = await response.json();
        return body?.status === "healthy" || body?.status === "degraded";
      }
    } catch {
      /* try next */
    }
  }
  return false;
}

export async function waitForBackend(
  maxAttempts = 30,
  intervalMs = 2000
): Promise<boolean> {
  for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
    if (await checkBackendHealth()) return true;
    await sleep(intervalMs);
  }
  return false;
}

export async function postJson<T>(path: string): Promise<T | null> {
  const bases = ["", API_BASE];

  for (const base of bases) {
    try {
      const response = await fetch(`${base}${path}`, {
        method: "POST",
        cache: "no-store",
      });
      if (response.ok) return (await response.json()) as T;
    } catch {
      /* try next base */
    }
  }
  return null;
}

export async function fetchJson<T>(path: string): Promise<T | null> {
  const bases = ["", API_BASE];

  for (const base of bases) {
    try {
      const response = await fetch(`${base}${path}`, { cache: "no-store" });
      if (response.ok) return (await response.json()) as T;
    } catch {
      /* try next base */
    }
  }
  return null;
}
