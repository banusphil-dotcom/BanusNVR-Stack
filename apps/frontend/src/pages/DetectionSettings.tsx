import { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api, getToken } from "../api";
import {
  ArrowLeft,
  Save,
  ChevronDown,
  ChevronRight,
  Zap,
  Shield,
  SlidersHorizontal,
  EyeOff,
  Trash2,
  Plus,
  X,
} from "lucide-react";

interface IgnoreZone {
  polygon: number[][];
  classes: string[];
  label: string;
  image_width?: number;
  image_height?: number;
}

interface DetectionData {
  camera_id: number;
  camera_name: string;
  detection_enabled: boolean;
  detection_objects: string[];
  detection_confidence: number;
  detection_settings: Record<string, ObjectSettings>;
  ptz_mode: boolean;
}

interface ObjectSettings {
  confidence?: number;
  min_area?: number;
  enhanced?: boolean;
}

interface ClassesData {
  classes: string[];
  categories: Record<string, string[]>;
}

const COCO_CATEGORIES: Record<string, string[]> = {
  People: ["person"],
  Vehicles: ["bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"],
  Animals: ["bird", "cat", "dog", "horse", "sheep", "cow"],
  Accessories: ["backpack", "umbrella", "handbag", "tie", "suitcase"],
  Sports: ["frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket"],
  Kitchen: ["bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl"],
  Food: ["banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake"],
  Furniture: ["chair", "couch", "potted plant", "bed", "dining table", "toilet"],
  Electronics: ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster"],
  Indoor: ["sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"],
};

const ALL_COCO_CLASSES = Object.values(COCO_CATEGORIES).flat();

// Recommended presets for key object types
const PRESETS: Record<string, { confidence: number; min_area: number; enhanced: boolean; tip: string }> = {
  person: { confidence: 0.35, min_area: 3000, enhanced: true, tip: "Lowered threshold + enhanced mode for better detection of distant or partially obscured people" },
  cat: { confidence: 0.30, min_area: 800, enhanced: true, tip: "Cats are often small and fast — low threshold + enhanced multi-crop detection catches them reliably" },
  dog: { confidence: 0.35, min_area: 1500, enhanced: true, tip: "Similar to cats but typically larger, enhanced mode helps with fast-moving dogs" },
  car: { confidence: 0.50, min_area: 5000, enhanced: false, tip: "Vehicles are large and easy to detect at standard confidence" },
  bird: { confidence: 0.30, min_area: 400, enhanced: true, tip: "Birds are small — low threshold and enhanced mode recommended" },
  truck: { confidence: 0.50, min_area: 8000, enhanced: false, tip: "Large vehicles detected reliably at standard settings" },
  bicycle: { confidence: 0.40, min_area: 2000, enhanced: false, tip: "Medium-sized objects, standard settings work well" },
  motorcycle: { confidence: 0.40, min_area: 2500, enhanced: false, tip: "Standard detection settings sufficient" },
};

export default function DetectionSettings() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const qc = useQueryClient();
  const cameraId = parseInt(id || "0");

  const { data: detData, isLoading: loadingDet } = useQuery({
    queryKey: ["detection-settings", cameraId],
    queryFn: () => api.get<DetectionData>(`/api/cameras/${cameraId}/detection-settings`),
    enabled: cameraId > 0,
  });

  const [enabledObjects, setEnabledObjects] = useState<string[]>([]);
  const [globalConf, setGlobalConf] = useState(0.5);
  const [objSettings, setObjSettings] = useState<Record<string, ObjectSettings>>({});
  const [expandedCategory, setExpandedCategory] = useState<string | null>(null);
  const [expandedObject, setExpandedObject] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);

  // ── Ignore zones ──
  const { data: ignoreData, refetch: refetchIgnore } = useQuery({
    queryKey: ["ignore-zones", cameraId],
    queryFn: () => api.get<{ ignore_zones: IgnoreZone[] }>(`/api/cameras/${cameraId}/ignore-zones`),
    enabled: cameraId > 0,
  });
  const [drawingZone, setDrawingZone] = useState(false);
  const [newPoints, setNewPoints] = useState<number[][]>([]);
  const [newZoneClasses, setNewZoneClasses] = useState<string[]>([]);
  const [newZoneLabel, setNewZoneLabel] = useState("");
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const [snapshotUrl, setSnapshotUrl] = useState<string | null>(null);

  // Fetch snapshot with auth header when drawing mode starts
  useEffect(() => {
    if (!drawingZone) return;
    let revoked = false;
    const fetchSnapshot = async () => {
      try {
        const token = getToken();
        const res = await fetch(`/api/cameras/${cameraId}/snapshot?t=${Date.now()}`, {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        });
        if (!res.ok) return;
        const blob = await res.blob();
        if (revoked) return;
        setSnapshotUrl(URL.createObjectURL(blob));
      } catch { /* ignore */ }
    };
    fetchSnapshot();
    return () => { revoked = true; };
  }, [drawingZone, cameraId]);

  const [ignoreError, setIgnoreError] = useState<string | null>(null);
  const saveIgnoreMut = useMutation({
    mutationFn: (zones: IgnoreZone[]) =>
      api.put(`/api/cameras/${cameraId}/ignore-zones`, zones),
    onSuccess: () => {
      setIgnoreError(null);
      refetchIgnore();
      setDrawingZone(false);
      setNewPoints([]);
      setNewZoneClasses([]);
      setNewZoneLabel("");
      if (snapshotUrl) { URL.revokeObjectURL(snapshotUrl); setSnapshotUrl(null); }
    },
    onError: (err: Error) => {
      setIgnoreError(err.message || "Failed to save ignore zone");
    },
  });

  const handleCanvasClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!drawingZone || !canvasRef.current) return;
      const rect = canvasRef.current.getBoundingClientRect();
      const scaleX = canvasRef.current.width / rect.width;
      const scaleY = canvasRef.current.height / rect.height;
      const x = Math.round((e.clientX - rect.left) * scaleX);
      const y = Math.round((e.clientY - rect.top) * scaleY);
      setNewPoints((prev) => [...prev, [x, y]]);
    },
    [drawingZone],
  );

  // Redraw canvas overlay when points change
  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !img.naturalWidth) return;
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Draw existing zones
    for (const zone of ignoreData?.ignore_zones || []) {
      ctx.fillStyle = "rgba(239, 68, 68, 0.2)";
      ctx.strokeStyle = "rgba(239, 68, 68, 0.7)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      zone.polygon.forEach(([px, py], i) => (i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)));
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      if (zone.label) {
        ctx.fillStyle = "rgba(239, 68, 68, 0.9)";
        ctx.font = "14px sans-serif";
        ctx.fillText(zone.label, zone.polygon[0][0] + 4, zone.polygon[0][1] - 4);
      }
    }
    // Draw in-progress polygon
    if (newPoints.length > 0) {
      ctx.fillStyle = "rgba(59, 130, 246, 0.25)";
      ctx.strokeStyle = "rgba(59, 130, 246, 0.9)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      newPoints.forEach(([px, py], i) => (i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)));
      if (newPoints.length > 2) { ctx.closePath(); ctx.fill(); }
      ctx.stroke();
      // Draw vertices
      for (const [px, py] of newPoints) {
        ctx.fillStyle = "rgba(59, 130, 246, 1)";
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }, [newPoints, ignoreData]);

  const addIgnoreZone = () => {
    if (newPoints.length < 3) return;
    const img = imgRef.current;
    const iw = img?.naturalWidth || 0;
    const ih = img?.naturalHeight || 0;
    const existing = ignoreData?.ignore_zones || [];
    saveIgnoreMut.mutate([
      ...existing,
      {
        polygon: newPoints,
        classes: newZoneClasses,
        label: newZoneLabel,
        image_width: iw,
        image_height: ih,
      },
    ]);
  };

  const removeIgnoreZone = (idx: number) => {
    const existing = ignoreData?.ignore_zones || [];
    saveIgnoreMut.mutate(existing.filter((_, i) => i !== idx));
  };

  useEffect(() => {
    if (detData) {
      setEnabledObjects(detData.detection_objects);
      setGlobalConf(detData.detection_confidence);
      setObjSettings(detData.detection_settings || {});
    }
  }, [detData]);

  const saveMut = useMutation({
    mutationFn: (payload: {
      detection_objects: string[];
      detection_confidence: number;
      detection_settings: Record<string, ObjectSettings>;
    }) => api.put(`/api/cameras/${cameraId}/detection-settings`, payload),
    onSuccess: () => {
      setDirty(false);
      qc.invalidateQueries({ queryKey: ["detection-settings", cameraId] });
      qc.invalidateQueries({ queryKey: ["cameras"] });
    },
  });

  const toggleObject = (cls: string) => {
    setDirty(true);
    if (enabledObjects.includes(cls)) {
      setEnabledObjects(enabledObjects.filter((o) => o !== cls));
      const next = { ...objSettings };
      delete next[cls];
      setObjSettings(next);
    } else {
      setEnabledObjects([...enabledObjects, cls]);
      // Apply preset if available, otherwise use global defaults
      const preset = PRESETS[cls];
      if (preset) {
        setObjSettings({
          ...objSettings,
          [cls]: { confidence: preset.confidence, min_area: preset.min_area, enhanced: preset.enhanced },
        });
      }
    }
  };

  const updateObjSetting = (cls: string, key: keyof ObjectSettings, value: number | boolean) => {
    setDirty(true);
    setObjSettings({
      ...objSettings,
      [cls]: { ...objSettings[cls], [key]: value },
    });
  };

  const applyPreset = (cls: string) => {
    const preset = PRESETS[cls];
    if (!preset) return;
    setDirty(true);
    setObjSettings({
      ...objSettings,
      [cls]: { confidence: preset.confidence, min_area: preset.min_area, enhanced: preset.enhanced },
    });
  };

  const handleSave = () => {
    // Only save settings for enabled objects
    const filtered: Record<string, ObjectSettings> = {};
    for (const cls of enabledObjects) {
      if (objSettings[cls]) filtered[cls] = objSettings[cls];
    }
    saveMut.mutate({
      detection_objects: enabledObjects,
      detection_confidence: globalConf,
      detection_settings: filtered,
    });
  };

  if (loadingDet) {
    return <div className="p-4 text-center text-slate-400 py-12">Loading...</div>;
  }

  if (!detData) {
    return <div className="p-4 text-center text-slate-400 py-12">Camera not found</div>;
  }

  const categories = COCO_CATEGORIES;

  return (
    <div className="p-4 space-y-4 pb-24">
      {/* Header */}
      <div className="flex items-center gap-3">
        <button onClick={() => navigate("/settings")} className="p-1.5 hover:bg-slate-800 rounded-lg transition-colors">
          <ArrowLeft size={20} />
        </button>
        <div>
          <h2 className="text-lg font-bold">Detection Settings</h2>
          <p className="text-sm text-slate-400">{detData.camera_name}</p>
        </div>
      </div>

      {/* Global Confidence */}
      <div className="card space-y-3">
        <h3 className="font-semibold text-sm flex items-center gap-2">
          <SlidersHorizontal size={16} className="text-blue-400" /> Global Settings
        </h3>
        <div>
          <label className="label">Default Confidence: {(globalConf * 100).toFixed(0)}%</label>
          <input
            type="range"
            min="0.1"
            max="0.95"
            step="0.05"
            value={globalConf}
            onChange={(e) => { setGlobalConf(parseFloat(e.target.value)); setDirty(true); }}
            className="w-full"
          />
          <p className="text-xs text-slate-500 mt-1">
            Used for objects that don't have per-object confidence set below
          </p>
        </div>
      </div>

      {/* Active Objects — Per-Object Controls */}
      {enabledObjects.length > 0 && (
        <div className="card space-y-3">
          <h3 className="font-semibold text-sm flex items-center gap-2">
            <Shield size={16} className="text-emerald-400" /> Active Objects — Fine Tuning
          </h3>
          <div className="space-y-2">
            {enabledObjects.map((cls) => {
              const s = objSettings[cls] || {};
              const conf = s.confidence ?? globalConf;
              const minArea = s.min_area ?? 0;
              const isExpanded = expandedObject === cls;
              const preset = PRESETS[cls];

              return (
                <div key={cls} className="bg-slate-800 rounded-lg overflow-hidden">
                  <button
                    onClick={() => setExpandedObject(isExpanded ? null : cls)}
                    className="w-full flex items-center justify-between p-3 text-left"
                  >
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-sm capitalize">{cls}</span>
                      <span className="text-xs text-slate-500">{(conf * 100).toFixed(0)}%</span>
                    </div>
                    {isExpanded ? <ChevronDown size={16} className="text-slate-400" /> : <ChevronRight size={16} className="text-slate-400" />}
                  </button>

                  {isExpanded && (
                    <div className="px-3 pb-3 space-y-3 border-t border-slate-700 pt-3">
                      {preset && (
                        <div className="flex items-start gap-2 p-2 bg-blue-900/20 rounded-lg border border-blue-800/30">
                          <Zap size={14} className="text-blue-400 shrink-0 mt-0.5" />
                          <div className="flex-1">
                            <p className="text-xs text-blue-300">{preset.tip}</p>
                            <button
                              onClick={() => applyPreset(cls)}
                              className="text-xs text-blue-400 hover:text-blue-300 font-medium mt-1"
                            >
                              Apply recommended preset
                            </button>
                          </div>
                        </div>
                      )}

                      <div>
                        <label className="text-xs text-slate-400">
                          Confidence: {(conf * 100).toFixed(0)}%
                        </label>
                        <input
                          type="range"
                          min="0.1"
                          max="0.95"
                          step="0.05"
                          value={conf}
                          onChange={(e) => updateObjSetting(cls, "confidence", parseFloat(e.target.value))}
                          className="w-full"
                        />
                        <div className="flex justify-between text-[10px] text-slate-600">
                          <span>More detections</span>
                          <span>Fewer false positives</span>
                        </div>
                      </div>

                      <div>
                        <label className="text-xs text-slate-400">
                          Min Area: {minArea}px²
                        </label>
                        <input
                          type="range"
                          min="0"
                          max="20000"
                          step="500"
                          value={minArea}
                          onChange={(e) => updateObjSetting(cls, "min_area", parseInt(e.target.value))}
                          className="w-full"
                        />
                        <div className="flex justify-between text-[10px] text-slate-600">
                          <span>All sizes</span>
                          <span>Large only</span>
                        </div>
                      </div>

                      <button
                        onClick={() => toggleObject(cls)}
                        className="text-xs text-red-400 hover:text-red-300"
                      >
                        Remove from detection
                      </button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Ignore Zones — suppress detections in specific areas */}
      <div className="card space-y-3">
        <h3 className="font-semibold text-sm flex items-center gap-2">
          <EyeOff size={16} className="text-red-400" /> Ignore Zones
        </h3>
        <p className="text-xs text-slate-500">
          Click on the camera snapshot to draw polygon zones where detections should be suppressed
          (e.g. a statue that gets misidentified).
        </p>

        {/* Existing zones list */}
        {(ignoreData?.ignore_zones || []).map((zone, i) => (
          <div key={i} className="flex items-center justify-between bg-slate-800 rounded-lg p-2.5">
            <div className="text-sm">
              <span className="font-medium">{zone.label || `Zone ${i + 1}`}</span>
              <span className="text-xs text-slate-500 ml-2">
                {zone.polygon.length} points
                {zone.classes.length > 0 ? ` · ${zone.classes.join(", ")}` : " · all classes"}
              </span>
            </div>
            <button
              onClick={() => removeIgnoreZone(i)}
              className="p-1 text-red-400 hover:text-red-300"
            >
              <Trash2 size={14} />
            </button>
          </div>
        ))}

        {/* Drawing UI */}
        {drawingZone ? (
          <div className="space-y-3">
            <div className="relative rounded-lg overflow-hidden border border-slate-700">
              {snapshotUrl ? (
                <img
                  ref={imgRef}
                  src={snapshotUrl}
                  alt="Camera snapshot"
                  className="w-full"
                  onLoad={() => {
                    const canvas = canvasRef.current;
                    const img = imgRef.current;
                    if (canvas && img) {
                      canvas.width = img.naturalWidth;
                      canvas.height = img.naturalHeight;
                    }
                  }}
                />
              ) : (
                <div className="w-full h-48 flex items-center justify-center text-slate-500 text-sm">Loading snapshot...</div>
              )}
              <canvas
                ref={canvasRef}
                onClick={handleCanvasClick}
                className="absolute inset-0 w-full h-full cursor-crosshair"
              />
            </div>
            <p className="text-xs text-slate-400">
              Click to place points. {newPoints.length} point{newPoints.length !== 1 ? "s" : ""} placed
              {newPoints.length < 3 ? ` (need ${3 - newPoints.length} more)` : " — ready to save"}.
            </p>
            <div>
              <label className="text-xs text-slate-400 block mb-1">Label (optional)</label>
              <input
                type="text"
                value={newZoneLabel}
                onChange={(e) => setNewZoneLabel(e.target.value)}
                className="bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-sm w-full"
                placeholder="e.g. Statue area"
              />
            </div>
            <div>
              <label className="text-xs text-slate-400 block mb-1">
                Suppress classes (empty = all)
              </label>
              <div className="flex flex-wrap gap-1.5">
                {enabledObjects.map((cls) => (
                  <button
                    key={cls}
                    onClick={() =>
                      setNewZoneClasses((prev) =>
                        prev.includes(cls) ? prev.filter((c) => c !== cls) : [...prev, cls],
                      )
                    }
                    className={`text-xs px-2 py-1 rounded-full capitalize ${
                      newZoneClasses.includes(cls)
                        ? "bg-red-600 text-white"
                        : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                    }`}
                  >
                    {cls}
                  </button>
                ))}
              </div>
            </div>
            <div className="flex gap-2">
              <button
                onClick={addIgnoreZone}
                disabled={newPoints.length < 3 || saveIgnoreMut.isPending}
                className="btn-primary text-sm px-3 py-1.5 flex items-center gap-1.5 disabled:opacity-40"
              >
                <Save size={14} />
                Save Zone
              </button>
              <button
                onClick={() => setNewPoints([])}
                className="text-xs text-slate-400 hover:text-slate-300 px-2"
              >
                Clear Points
              </button>
              <button
                onClick={() => { setDrawingZone(false); setNewPoints([]); if (snapshotUrl) { URL.revokeObjectURL(snapshotUrl); setSnapshotUrl(null); } }}
                className="text-xs text-slate-400 hover:text-slate-300 px-2 flex items-center gap-1"
              >
                <X size={12} /> Cancel
              </button>
            </div>
            {ignoreError && (
              <p className="text-xs text-red-400">{ignoreError}</p>
            )}
          </div>
        ) : (
          <button
            onClick={() => setDrawingZone(true)}
            className="flex items-center gap-1.5 text-sm text-blue-400 hover:text-blue-300"
          >
            <Plus size={14} /> Add Ignore Zone
          </button>
        )}
      </div>

      {/* Add Objects by Category */}
      <div className="card space-y-3">
        <h3 className="font-semibold text-sm">Available Object Types</h3>
        <p className="text-xs text-slate-500">
          Toggle objects to detect. All {ALL_COCO_CLASSES.length} COCO classes available.
        </p>
        <div className="space-y-1">
          {Object.entries(categories).map(([category, classes]) => {
            const isExpanded = expandedCategory === category;
            const enabledCount = classes.filter((c) => enabledObjects.includes(c)).length;

            return (
              <div key={category}>
                <button
                  onClick={() => setExpandedCategory(isExpanded ? null : category)}
                  className="w-full flex items-center justify-between p-2 rounded hover:bg-slate-800 transition-colors"
                >
                  <div className="flex items-center gap-2">
                    {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                    <span className="text-sm font-medium">{category}</span>
                    <span className="text-xs text-slate-500">({classes.length})</span>
                  </div>
                  {enabledCount > 0 && (
                    <span className="text-xs px-1.5 py-0.5 rounded bg-blue-600/20 text-blue-400">
                      {enabledCount} active
                    </span>
                  )}
                </button>

                {isExpanded && (
                  <div className="flex flex-wrap gap-1.5 pl-7 pb-2 pt-1">
                    {classes.map((cls) => (
                      <button
                        key={cls}
                        onClick={() => toggleObject(cls)}
                        className={`text-xs px-2.5 py-1.5 rounded-full transition-colors capitalize ${
                          enabledObjects.includes(cls)
                            ? "bg-blue-600 text-white"
                            : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                        }`}
                      >
                        {cls}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Sticky Save Bar */}
      {dirty && (
        <div className="fixed bottom-0 left-0 right-0 px-4 py-3 bg-slate-900/95 border-t border-slate-700 backdrop-blur-sm z-40 safe-area-pb">
          <button
            onClick={handleSave}
            disabled={saveMut.isPending}
            className="btn-primary w-full flex items-center justify-center gap-2"
          >
            <Save size={16} />
            {saveMut.isPending ? "Saving..." : "Save Detection Settings"}
          </button>
        </div>
      )}
    </div>
  );
}
