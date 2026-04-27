import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import "./index.css";

// ── Bootstrap theme before React renders to avoid flash ──
(function initTheme() {
  const saved = (localStorage.getItem("banusnas_theme") as "light" | "dark" | "system" | null) || "system";
  let resolved: "light" | "dark" = "dark";
  if (saved === "system") {
    resolved = window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  } else {
    resolved = saved;
  }
  const root = document.documentElement;
  root.classList.remove("light", "dark");
  root.classList.add(resolved);
  root.dataset.theme = resolved;
  // React to OS changes when on "system"
  try {
    window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", (e) => {
      const cur = localStorage.getItem("banusnas_theme") || "system";
      if (cur !== "system") return;
      const next = e.matches ? "light" : "dark";
      root.classList.remove("light", "dark");
      root.classList.add(next);
      root.dataset.theme = next;
    });
  } catch { /* older browsers */ }
})();

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: 1, refetchOnWindowFocus: false, staleTime: 30_000 },
  },
});

// Register service worker for PWA
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").then((reg) => {
      // Check for updates every 60s
      setInterval(() => reg.update(), 60_000);
    });
  });
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
