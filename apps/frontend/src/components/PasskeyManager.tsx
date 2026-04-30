import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Fingerprint, Loader2, Trash2, Plus } from "lucide-react";
import { api } from "../api";
import { isWebAuthnSupported, registerPasskey } from "../webauthn";

interface Credential {
  id: number;
  name: string;
  created_at: string;
  last_used_at?: string | null;
  transports: string[];
  is_backup: boolean;
}

export default function PasskeyManager() {
  const qc = useQueryClient();
  const [name, setName] = useState("");
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [supported, setSupported] = useState(true);

  useEffect(() => {
    setSupported(isWebAuthnSupported());
  }, []);

  const { data, isLoading } = useQuery<{ credentials: Credential[] }>({
    queryKey: ["webauthn-credentials"],
    queryFn: () => api.get("/api/webauthn/"),
  });

  const del = useMutation({
    mutationFn: (id: number) => api.delete(`/api/webauthn/${id}`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["webauthn-credentials"] }),
  });

  const handleAdd = async () => {
    setError(null);
    setAdding(true);
    try {
      await registerPasskey(name || "Passkey");
      setName("");
      await qc.invalidateQueries({ queryKey: ["webauthn-credentials"] });
    } catch (e: any) {
      setError(e?.message || "Failed to register passkey");
    } finally {
      setAdding(false);
    }
  };

  if (!supported) {
    return (
      <div className="text-xs text-slate-400">
        Passkeys are not supported in this browser. Use a recent version of Chrome, Edge, Safari, or Firefox.
      </div>
    );
  }

  const creds = data?.credentials ?? [];

  return (
    <div className="space-y-3">
      <div className="flex flex-col sm:flex-row gap-2">
        <input
          className="input flex-1"
          placeholder="Device name (e.g. iPhone, YubiKey)"
          value={name}
          onChange={(e) => setName(e.target.value)}
          maxLength={100}
        />
        <button
          onClick={handleAdd}
          disabled={adding}
          className="btn-primary text-sm flex items-center justify-center gap-1"
        >
          {adding ? <Loader2 size={14} className="animate-spin" /> : <Plus size={14} />}
          Add passkey
        </button>
      </div>
      {error && <p className="text-xs text-red-400">{error}</p>}

      {isLoading ? (
        <p className="text-xs text-slate-500">Loading…</p>
      ) : creds.length === 0 ? (
        <p className="text-xs text-slate-500">
          No passkeys registered yet. Add one to enable biometric / Touch ID / Windows Hello sign-in.
        </p>
      ) : (
        <ul className="divide-y divide-slate-800 border border-slate-800 rounded-lg">
          {creds.map((c) => (
            <li key={c.id} className="flex items-center justify-between p-2">
              <div className="flex items-center gap-2 min-w-0">
                <Fingerprint size={16} className="text-emerald-400 shrink-0" />
                <div className="min-w-0">
                  <div className="text-sm font-medium truncate">{c.name}</div>
                  <div className="text-xs text-slate-500 truncate">
                    Added {new Date(c.created_at).toLocaleDateString()}
                    {c.last_used_at && ` • Last used ${new Date(c.last_used_at).toLocaleDateString()}`}
                    {c.transports.length > 0 && ` • ${c.transports.join(", ")}`}
                  </div>
                </div>
              </div>
              <button
                onClick={() => {
                  if (confirm(`Remove passkey "${c.name}"?`)) del.mutate(c.id);
                }}
                className="p-1 rounded hover:bg-slate-800 text-red-400 hover:text-red-300"
                title="Remove"
              >
                <Trash2 size={14} />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
