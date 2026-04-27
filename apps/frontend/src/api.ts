/** API client — thin wrapper around fetch with auth token handling. */

let accessToken: string | null = localStorage.getItem("banusnas_token");

export function setToken(token: string | null) {
  accessToken = token;
  if (token) localStorage.setItem("banusnas_token", token);
  else localStorage.removeItem("banusnas_token");
}

export function getToken() {
  return accessToken;
}

async function request<T>(path: string, init?: RequestInit, _retried?: boolean): Promise<T> {
  const headers: Record<string, string> = {
    ...(init?.headers as Record<string, string>),
  };

  if (accessToken) headers["Authorization"] = `Bearer ${accessToken}`;

  // Don't set Content-Type for FormData (browser handles multipart boundary)
  if (!(init?.body instanceof FormData) && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }

  let res: Response;
  try {
    res = await fetch(path, { ...init, headers });
  } catch {
    // Network error — likely Cloudflare Access session expired (302 cross-origin redirect blocked)
    // Force full page reload so the browser navigates through CF Access re-auth
    window.location.reload();
    throw new Error("Network error — reloading page");
  }

  // If we got redirected to a non-API page (e.g., Cloudflare Access login HTML), reload
  if (res.redirected && !res.url.includes("/api/")) {
    window.location.reload();
    throw new Error("Session expired — reloading page");
  }

  if (res.status === 401) {
    if (!_retried) {
      const refreshed = await tryRefresh();
      if (refreshed) return request<T>(path, init, true);
    }
    setToken(null);
    window.location.href = "/login";
    throw new Error("Unauthorized");
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail || res.statusText);
  }

  if (res.status === 204) return undefined as T;
  return res.json();
}

async function tryRefresh(): Promise<boolean> {
  const refresh = localStorage.getItem("banusnas_refresh");
  if (!refresh) return false;

  try {
    const res = await fetch("/api/auth/refresh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refresh }),
    });
    if (!res.ok) return false;
    const data = await res.json();
    setToken(data.access_token);
    if (data.refresh_token) localStorage.setItem("banusnas_refresh", data.refresh_token);
    return true;
  } catch {
    return false;
  }
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "POST", body: body instanceof FormData ? body : JSON.stringify(body) }),
  put: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "PUT", body: JSON.stringify(body) }),
  patch: <T>(path: string, body?: unknown) =>
    request<T>(path, { method: "PATCH", body: JSON.stringify(body) }),
  delete: <T>(path: string) => request<T>(path, { method: "DELETE" }),
};
