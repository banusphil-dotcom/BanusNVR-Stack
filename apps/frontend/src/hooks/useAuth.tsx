import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { api, setToken } from "../api";

interface User {
  id: string;
  username: string;
  is_admin: boolean;
  theme?: string;
}

interface AuthCtx {
  user: User | null;
  loading: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthCtx>(null!);

function syncTheme(theme: string | undefined) {
  if (!theme) return;
  const stored = localStorage.getItem("banusnas_theme");
  if (stored === theme) return;
  localStorage.setItem("banusnas_theme", theme);
  let resolved: "light" | "dark" = "dark";
  if (theme === "system") {
    resolved = window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  } else if (theme === "light" || theme === "dark") {
    resolved = theme;
  }
  const root = document.documentElement;
  root.classList.remove("light", "dark");
  root.classList.add(resolved);
  root.dataset.theme = resolved;
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem("banusnas_token");
    if (!token) {
      setLoading(false);
      return;
    }
    api
      .get<User>("/api/auth/me")
      .then((u) => { syncTheme(u.theme); setUser(u); })
      .catch(() => setToken(null))
      .finally(() => setLoading(false));
  }, []);

  const login = async (username: string, password: string) => {
    const data = await api.post<{ access_token: string; refresh_token: string }>("/api/auth/login", {
      username,
      password,
    });
    setToken(data.access_token);
    localStorage.setItem("banusnas_refresh", data.refresh_token);
    const me = await api.get<User>("/api/auth/me");
    syncTheme(me.theme);
    setUser(me);
  };

  const register = async (username: string, password: string) => {
    await api.post("/api/auth/register", { username, password });
    await login(username, password);
  };

  const logout = () => {
    setToken(null);
    localStorage.removeItem("banusnas_refresh");
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
