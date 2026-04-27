/// <reference lib="webworker" />
import { precacheAndRoute, cleanupOutdatedCaches } from "workbox-precaching";
import { registerRoute } from "workbox-routing";
import { NetworkFirst } from "workbox-strategies";
import { ExpirationPlugin } from "workbox-expiration";

export type {};
declare const self: ServiceWorkerGlobalScope;

// Take control immediately on update
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (event) => {
  event.waitUntil(
    self.clients.claim().then(() =>
      self.clients.matchAll({ type: "window" }).then((clients) => {
        for (const client of clients) {
          client.postMessage({ type: "SW_UPDATED" });
        }
      })
    )
  );
});

// Allow the app to explicitly trigger skipWaiting from a waiting SW
self.addEventListener("message", (event) => {
  if (event.data?.type === "SKIP_WAITING") {
    self.skipWaiting();
  }
});

// Cleanup old precache entries
cleanupOutdatedCaches();

// Precache all assets built by Vite
precacheAndRoute(self.__WB_MANIFEST);

// Cache API responses (except recordings and snapshots)
registerRoute(
  ({ url }) =>
    url.pathname.startsWith("/api/") &&
    !url.pathname.startsWith("/api/recordings") &&
    !url.pathname.match(/\/api\/events\/.*\/snapshot/),
  new NetworkFirst({
    cacheName: "api-cache",
    plugins: [new ExpirationPlugin({ maxEntries: 50, maxAgeSeconds: 300 })],
  })
);

// Listen for push notifications
self.addEventListener("push", (event: PushEvent) => {
  let data: { title?: string; body?: string; url?: string; image?: string };
  try {
    data = event.data?.json() ?? {};
  } catch {
    data = {};
  }
  const title = typeof data.title === "string" ? data.title : "BanusNas";
  const options: NotificationOptions = {
    body: typeof data.body === "string" ? data.body : "New event",
    icon: "/pwa-192.png",
    badge: "/pwa-192.png",
    data: typeof data.url === "string" ? data.url : undefined,
  };
  if (typeof data.image === "string") {
    try {
      const imgUrl = new URL(data.image, self.location.origin);
      if (imgUrl.origin === self.location.origin) {
        options.image = imgUrl.href;
      }
    } catch { /* invalid URL — skip image */ }
  }
  event.waitUntil(
    self.registration.showNotification(title, options)
  );
});

self.addEventListener("notificationclick", (event: NotificationEvent) => {
  event.notification.close();
  const url = (event.notification.data as string) || "/";
  event.waitUntil(
    self.clients.matchAll({ type: "window", includeUncontrolled: true }).then((clients) => {
      // Focus existing window if possible
      for (const client of clients) {
        if ("focus" in client) {
          client.focus();
          client.postMessage({ type: "NAVIGATE", url });
          return;
        }
      }
      return self.clients.openWindow(url);
    })
  );
});
