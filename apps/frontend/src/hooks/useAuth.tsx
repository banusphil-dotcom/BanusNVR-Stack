import { createContext, useContext, useEffect, useState, type ReactNode } from "react";
import { api, setToken } from "../api";
import { loginWithPasskey } from "../webauthn";

export type UserRole = "admin" | "operator" | "viewer" | "guest";

export interface User {
  id: string;
  username: string;
  email?: string;
  is_admin: boolean;
  role: UserRole;
  theme?: string;
  must_change_password?: boolean;
  disabled?: boolean;
  last_login_at?: string | null;
  totp_enabled?: boolean;
}

interface PermissionsResponse {
  role: UserRole;
  permissions: string[];
  must_change_password: boolean;
}

interface LoginResponse {
  access_token: string;
  refresh_token: string;
  must_change_password?: boolean;
  step?: "totp" | null;
  temp_token?: string | null;
}

export type LoginResult =
  | { kind: "ok" }
  | { kind: "totp"; tempToken: string };

interface AuthCtx {
  user: User | null;
  permissions: string[];
  loading: boolean;
  mustChangePassword: boolean;
  hasPermission: (perm: string) => boolean;
  hasRole: (...roles: UserRole[]) => boolean;
  login: (username: string, password: string) => Promise<LoginResult>;
  loginTotp: (tempToken: string, code: string) => Promise<void>;
  loginPasskey: (username?: string) => Promise<void>;
  register: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  refreshProfile: () => Promise<void>;
  setMustChangePassword: (v: boolean) => void;
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
  const [permissions, setPermissions] = useState<string[]>([]);
  const [mustChangePassword, setMustChangePassword] = useState(false);
  const [loading, setLoading] = useState(true);

  const loadProfile = async () => {
    const u = await api.get<User>("/api/auth/me");
    syncTheme(u.theme);
    setUser(u);
    try {
      const p = await api.get<PermissionsResponse>("/api/auth/me/permissions");
      setPermissions(p.permissions);
      setMustChangePassword(p.must_change_password);
    } catch {
      setPermissions([]);
    }
  };

  useEffect(() => {
    const token = localStorage.getItem("banusnas_token");
    if (!token) {
      setLoading(false);
      return;
    }
    loadProfile()
      .catch(() => setToken(null))
      .finally(() => setLoading(false));
  }, []);

  const finishLogin = async (data: LoginResponse) => {
    setToken(data.access_token);
    localStorage.setItem("banusnas_refresh", data.refresh_token);
    await loadProfile();
    if (data.must_change_password) setMustChangePassword(true);
  };

  const login = async (username: string, password: string): Promise<LoginResult> => {
    const data = await api.post<LoginResponse>("/api/auth/login", { username, password });
    if (data.step === "totp" && data.temp_token) {
      return { kind: "totp", tempToken: data.temp_token };
    }
    await finishLogin(data);
    return { kind: "ok" };
  };

  const loginTotp = async (tempToken: string, code: string) => {
    const data = await api.post<LoginResponse>("/api/auth/login/totp", {
      temp_token: tempToken,
      token: code,
    });
    await finishLogin(data);
  };

  const loginPasskey = async (username?: string) => {
    const data = await loginWithPasskey(username);
    await finishLogin(data as LoginResponse);
  };

  const register = async (username: string, password: string) => {
    await api.post("/api/auth/register", { username, password });
    await login(username, password);
  };

  const logout = async () => {
    try { await api.post("/api/auth/logout"); } catch { /* ignore */ }
    setToken(null);
    localStorage.removeItem("banusnas_refresh");
    setUser(null);
    setPermissions([]);
    setMustChangePassword(false);
  };

  const hasPermission = (perm: string) => permissions.includes(perm);
  const hasRole = (...roles: UserRole[]) => !!user && roles.includes(user.role);

  return (
    <AuthContext.Provider
      value={{
        user,
        permissions,
        loading,
        mustChangePassword,
        hasPermission,
        hasRole,
        login,
        loginTotp,
        loginPasskey,
        register,
        logout,
        refreshProfile: loadProfile,
        setMustChangePassword,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);
