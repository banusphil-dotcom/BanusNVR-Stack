import { Routes, Route, Navigate, useParams } from "react-router-dom";
import { AuthProvider, useAuth } from "./hooks/useAuth";
import Layout from "./components/Layout";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import Events from "./pages/Events";
import Search from "./pages/Search";
import Cameras from "./pages/Cameras";
import Training from "./pages/Training";
import Settings from "./pages/Settings";
import DetectionSettings from "./pages/DetectionSettings";
import Recordings from "./pages/Recordings";
import CameraView from "./pages/CameraView";
import ObjectProfile from "./pages/ObjectProfile";
import DailySummary from "./pages/DailySummary";
import Retrain from "./pages/Retrain";
import RingSetup from "./pages/RingSetup";
import Users from "./pages/Users";
import AuditLog from "./pages/AuditLog";
import ForceChangePasswordModal from "./components/ForceChangePasswordModal";

function RedirectToProfile() {
  const { id } = useParams();
  return <Navigate to={`/profiles/${id}`} replace />;
}

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  if (loading) return <div className="flex items-center justify-center h-screen text-slate-400">Loading...</div>;
  if (!user) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

function AppRoutes() {
  const { user, loading } = useAuth();

  if (loading) return null;

  return (
    <Routes>
      <Route path="/login" element={user ? <Navigate to="/" replace /> : <Login />} />
      <Route
        element={
          <ProtectedRoute>
            <Layout />
          </ProtectedRoute>
        }
      >
        <Route path="/" element={<Dashboard />} />
        <Route path="/events" element={<Events />} />
        <Route path="/search" element={<Search />} />
        <Route path="/cameras" element={<Cameras />} />
        <Route path="/cameras/:id/detection" element={<DetectionSettings />} />
        <Route path="/recordings" element={<Recordings />} />
        <Route path="/camera/:id" element={<CameraView />} />
        <Route path="/profiles" element={<Training />} />
        <Route path="/profiles/:id" element={<ObjectProfile />} />
        <Route path="/profiles/:id/retrain" element={<Retrain />} />
        {/* Legacy redirects */}
        <Route path="/training" element={<Navigate to="/profiles" replace />} />
        <Route path="/training/object/:id" element={<RedirectToProfile />} />
        <Route path="/ring" element={<RingSetup />} />
        <Route path="/notifications" element={<Navigate to="/events" replace />} />
        <Route path="/settings" element={<Settings />} />
        <Route path="/users" element={<Users />} />
        <Route path="/audit" element={<AuditLog />} />
        <Route path="/summary" element={<DailySummary />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <AppRoutes />
      <ForceChangePasswordModal />
    </AuthProvider>
  );
}
