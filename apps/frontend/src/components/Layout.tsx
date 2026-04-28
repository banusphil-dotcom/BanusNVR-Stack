import { useState, useEffect, useRef, useCallback } from "react";
import { Outlet, useLocation, useNavigate } from "react-router-dom";
import {
  LayoutDashboard,
  Bell,
  Search,
  Users,
  Settings as SettingsIcon,
  Download,
  X,
  RefreshCw,
  Menu,
  Camera,
  UserCog,
  ScrollText,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { useWebSocket } from "../hooks/useWebSocket";
import { useAuth } from "../hooks/useAuth";

const navItems = [
  { to: "/", icon: LayoutDashboard, label: "Live" },
  { to: "/events", icon: Bell, label: "Events" },
  { to: "/cameras", icon: Camera, label: "Cameras" },
  { to: "/profiles", icon: Users, label: "Profiles" },
  { to: "/search", icon: Search, label: "Search" },
  { to: "/settings", icon: SettingsIcon, label: "Settings" },
];

const adminNavItems = [
  { to: "/users", icon: UserCog, label: "Users", permission: "manage_users" },
  { to: "/audit", icon: ScrollText, label: "Audit log", permission: "view_audit_log" },
];

export default function Layout() {
  const { events, connected } = useWebSocket();
  const { user, hasPermission } = useAuth();
  const visibleAdminItems = adminNavItems.filter((i) => hasPermission(i.permission));
  const allNavItems = [...navItems, ...visibleAdminItems];
  const location = useLocation();
  const navigate = useNavigate();
  const deferredPrompt = useRef<BeforeInstallPromptEvent | null>(null);
  const [showInstallBanner, setShowInstallBanner] = useState(false);
  const [showUpdateBanner, setShowUpdateBanner] = useState(false);
  const [fabOpen, setFabOpen] = useState(false);

  // Unread notification count
  const { data: unreadData } = useQuery({
    queryKey: ["unread-count"],
    queryFn: () => api.get<{ count: number }>("/api/notifications/unread-count"),
    refetchInterval: 30_000,
  });
  const unreadCount = unreadData?.count ?? 0;

  // Close FAB when navigating
  useEffect(() => {
    setFabOpen(false);
  }, [location.pathname]);

  // Close FAB on outside click
  useEffect(() => {
    if (!fabOpen) return;
    const handleClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (!target.closest("[data-fab]")) setFabOpen(false);
    };
    document.addEventListener("click", handleClick);
    return () => document.removeEventListener("click", handleClick);
  }, [fabOpen]);

  // Listen for service worker update notifications
  useEffect(() => {
    if (!("serviceWorker" in navigator)) return;

    const onMessage = (event: MessageEvent) => {
      if (event.data?.type === "SW_UPDATED") {
        setShowUpdateBanner(true);
      }
      if (event.data?.type === "NAVIGATE" && event.data.url) {
        navigate(event.data.url);
      }
    };
    navigator.serviceWorker.addEventListener("message", onMessage);

    // Also detect controller change (another tab triggered the update)
    const onControllerChange = () => setShowUpdateBanner(true);
    navigator.serviceWorker.addEventListener("controllerchange", onControllerChange);

    return () => {
      navigator.serviceWorker.removeEventListener("message", onMessage);
      navigator.serviceWorker.removeEventListener("controllerchange", onControllerChange);
    };
  }, []);

  const handleUpdate = useCallback(() => {
    window.location.reload();
  }, []);

  useEffect(() => {
    const handler = (e: Event) => {
      e.preventDefault();
      deferredPrompt.current = e as BeforeInstallPromptEvent;
      const dismissed = sessionStorage.getItem("pwa-install-dismissed");
      if (!dismissed) setShowInstallBanner(true);
    };
    window.addEventListener("beforeinstallprompt", handler);

    const installedHandler = () => {
      setShowInstallBanner(false);
      deferredPrompt.current = null;
    };
    window.addEventListener("appinstalled", installedHandler);

    return () => {
      window.removeEventListener("beforeinstallprompt", handler);
      window.removeEventListener("appinstalled", installedHandler);
    };
  }, []);

  const handleInstall = async () => {
    if (!deferredPrompt.current) return;
    deferredPrompt.current.prompt();
    await deferredPrompt.current.userChoice;
    deferredPrompt.current = null;
    setShowInstallBanner(false);
  };

  const dismissInstall = () => {
    setShowInstallBanner(false);
    sessionStorage.setItem("pwa-install-dismissed", "1");
  };

  return (
    <div className="flex flex-col h-[100dvh] overflow-hidden">
      {/* Top bar */}
      <header className="flex items-center justify-between px-4 py-2 bg-slate-900 border-b border-slate-800 shrink-0 safe-area-pt">
        <button
          onClick={() => navigate("/")}
          className="flex items-center gap-2 hover:opacity-80 transition-opacity"
        >
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center">
            <Camera size={18} className="text-white" />
          </div>
          <span className="font-semibold text-lg hidden sm:block">BanusNVR</span>
        </button>
        <div className="flex items-center gap-3">
          <span
            className={`w-2 h-2 rounded-full ${connected ? "bg-green-500" : "bg-red-500"}`}
            title={connected ? "Connected" : "Disconnected"}
          />
          {/* Bell icon with unread badge */}
          <button
            onClick={() => navigate("/events")}
            className="relative p-1.5 hover:bg-slate-800 rounded-lg transition-colors"
            title="Events"
          >
            <Bell size={20} className="text-slate-400" />
            {unreadCount > 0 && (
              <span className="absolute -top-0.5 -right-0.5 min-w-[18px] h-[18px] flex items-center justify-center rounded-full bg-red-500 text-[10px] font-bold px-1">
                {unreadCount > 99 ? "99+" : unreadCount}
              </span>
            )}
          </button>
          <span className="text-sm text-slate-400 hidden sm:block">{user?.username}</span>
          {user?.role && user.role !== "viewer" && (
            <span
              className={`hidden sm:inline-block text-[10px] px-1.5 py-0.5 rounded font-bold uppercase tracking-wide ${
                user.role === "admin"
                  ? "bg-red-500/15 text-red-300 border border-red-500/30"
                  : user.role === "operator"
                    ? "bg-blue-500/15 text-blue-300 border border-blue-500/30"
                    : "bg-amber-500/15 text-amber-300 border border-amber-500/30"
              }`}
            >
              {user.role}
            </span>
          )}
        </div>
      </header>

      {/* Update Available Banner */}
      {showUpdateBanner && (
        <div className="flex items-center justify-between px-4 py-2 bg-emerald-600 text-white text-sm shrink-0">
          <div className="flex items-center gap-2">
            <RefreshCw size={16} />
            <span>A new version is available</span>
          </div>
          <button
            onClick={handleUpdate}
            className="px-3 py-1 bg-white text-emerald-600 rounded font-medium hover:bg-emerald-50 transition-colors"
          >
            Refresh
          </button>
        </div>
      )}

      {/* PWA Install Banner */}
      {showInstallBanner && !showUpdateBanner && (
        <div className="flex items-center justify-between px-4 py-2 bg-blue-600 text-white text-sm shrink-0">
          <div className="flex items-center gap-2">
            <Download size={16} />
            <span>Install BanusNas NVR for the best experience</span>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={handleInstall}
              className="px-3 py-1 bg-white text-blue-600 rounded font-medium hover:bg-blue-50 transition-colors"
            >
              Install
            </button>
            <button onClick={dismissInstall} className="p-1 hover:bg-blue-500 rounded transition-colors">
              <X size={16} />
            </button>
          </div>
        </div>
      )}

      {/* Content */}
      <main className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden">
        <Outlet context={{ events, connected }} />
      </main>

      {/* Floating Action Button navigation */}
      <div className="fixed bottom-6 right-6 z-50 flex flex-col-reverse items-center gap-3 safe-area-pb" data-fab>
        {/* FAB toggle */}
        <button
          onClick={() => setFabOpen((o) => !o)}
          className={`w-14 h-14 rounded-full bg-blue-600 shadow-lg flex items-center justify-center transition-transform duration-200 ${
            fabOpen ? "rotate-45" : ""
          }`}
        >
          <Menu size={24} className="text-white" />
        </button>

        {/* Cascade items */}
        {fabOpen &&
          allNavItems.map(({ to, icon: Icon, label }, i) => {
            const isActive = location.pathname === to || (to !== "/" && location.pathname.startsWith(to));
            return (
              <button
                key={to}
                onClick={() => {
                  navigate(to);
                  setFabOpen(false);
                }}
                className={`flex items-center gap-2 pl-3 pr-4 py-2 rounded-full shadow-lg transition-all duration-200 ${
                  isActive
                    ? "bg-blue-600 text-white"
                    : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                }`}
                style={{
                  animation: `fab-pop 0.15s ease-out ${i * 0.04}s both`,
                }}
              >
                <Icon size={18} />
                <span className="text-sm font-medium whitespace-nowrap">{label}</span>
              </button>
            );
          })}
      </div>
    </div>
  );
}
