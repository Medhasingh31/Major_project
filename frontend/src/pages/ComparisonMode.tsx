import React, { useState, useEffect, useRef } from 'react';
import { 
  Layers, 
  ArrowRight, 
  HelpCircle,
  Eye,
  Sliders,
  Calendar,
  Grid,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Upload,
  Play,
  AlertTriangle,
  Info
} from 'lucide-react';
import { apiService } from '../services/api';
import { ExtractionConfig } from '../types';

const DEFAULT_CONFIG: ExtractionConfig = {
  threshold: 0.30,
  closingRadius: 6,
  minObjectSize: 32,
  bridgeKernelSize: 5,
  imageSize: 512,
  useModel: true
};

export default function ComparisonMode() {
  // Config & Metadata states
  const [name, setName] = useState('Meridian County Comparison');
  const [studyArea, setStudyArea] = useState('Meridian County');
  const [config, setConfig] = useState<ExtractionConfig>(DEFAULT_CONFIG);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Epoch A (Before)
  const [fileA, setFileA] = useState<File | null>(null);
  const [jobIdA, setJobIdA] = useState('');
  const [yearA, setYearA] = useState('2016');

  // Epoch B (After)
  const [fileB, setFileB] = useState<File | null>(null);
  const [jobIdB, setJobIdB] = useState('');
  const [yearB, setYearB] = useState('2026');

  // Available completed runs from sessionStorage
  const [availableRuns, setAvailableRuns] = useState<{ id: string; name: string; year: string }[]>([]);

  // Processing & result states
  const [isProcessing, setIsProcessing] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [comparisonResult, setComparisonResult] = useState<any>(null);

  // Overlay GeoJSON geometries
  const [addedGeojson, setAddedGeojson] = useState<any>(null);
  const [removedGeojson, setRemovedGeojson] = useState<any>(null);
  const [unchangedGeojson, setUnchangedGeojson] = useState<any>(null);
  const [geojsonLoading, setGeojsonLoading] = useState(false);

  // Viewport navigation
  const [viewMode, setViewMode] = useState<'side-by-side' | 'swipe'>('side-by-side');
  const [swipeOffset, setSwipeOffset] = useState(50);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [isDraggingDivider, setIsDraggingDivider] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [cursorCoords, setCursorCoords] = useState({ x: 0, y: 0, lat: '31.9700° N', lng: '97.2400° W' });

  // Layer toggles
  const [showAdded, setShowAdded] = useState(true);
  const [showRemoved, setShowRemoved] = useState(true);
  const [showUnchanged, setShowUnchanged] = useState(true);
  const [showNodes, setShowNodes] = useState(true);

  const containerRef = useRef<HTMLDivElement>(null);

  // Load available runs on mount
  useEffect(() => {
    const runs = [];
    for (let i = 0; i < sessionStorage.length; i++) {
      const key = sessionStorage.key(i);
      if (key && key.startsWith('analysis_result_')) {
        try {
          const raw = sessionStorage.getItem(key);
          if (raw) {
            const data = JSON.parse(raw);
            runs.push({
              id: data.projectId,
              name: data.projectName || data.projectId,
              year: data.imageYear || '2026'
            });
          }
        } catch {}
      }
    }
    setAvailableRuns(runs);
  }, []);

  // Fetch GeoJSON overlays when a comparison result is loaded
  const loadComparisonOverlays = async (resultData: any) => {
    setGeojsonLoading(true);
    try {
      const fetchJson = async (url: string) => {
        if (!url) return null;
        const res = await fetch(url);
        if (res.ok) return res.json();
        return null;
      };

      const [added, removed, unchanged] = await Promise.all([
        fetchJson(resultData.addedGeojsonUrl),
        fetchJson(resultData.removedGeojsonUrl),
        fetchJson(resultData.unchangedGeojsonUrl)
      ]);

      setAddedGeojson(added);
      setRemovedGeojson(removed);
      setUnchangedGeojson(unchanged);
    } catch (e) {
      console.error('Failed to load comparison overlays:', e);
    } finally {
      setGeojsonLoading(false);
    }
  };

  // Drag-and-drop Handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDropA = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFileA(e.dataTransfer.files[0]);
      setJobIdA('');
    }
  };

  const handleDropB = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFileB(e.dataTransfer.files[0]);
      setJobIdB('');
    }
  };

  // Run Spatial Comparison API call
  const triggerComparison = async () => {
    setIsProcessing(true);
    setErrorMsg(null);
    setComparisonResult(null);
    setAddedGeojson(null);
    setRemovedGeojson(null);
    setUnchangedGeojson(null);

    try {
      const data = await apiService.runComparison(
        fileA,
        fileB,
        jobIdA,
        jobIdB,
        yearA,
        yearB,
        name,
        studyArea,
        config
      );

      setComparisonResult(data);
      await loadComparisonOverlays(data);
    } catch (err: any) {
      console.error('Comparison pipeline failed:', err);
      setErrorMsg(err.message || 'Spatial road comparison failed.');
    } finally {
      setIsProcessing(false);
    }
  };

  // Map Navigation & Pan/Zoom handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('button') || (e.target as HTMLElement).closest('select') || (e.target as HTMLElement).closest('input')) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDraggingDivider && containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      const offset = ((e.clientX - rect.left) / rect.width) * 100;
      setSwipeOffset(Math.max(0, Math.min(100, offset)));
      return;
    }

    if (isDragging) {
      setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    }

    const viewport = (e.target as HTMLElement).closest('.comparison-viewport');
    if (viewport) {
      const rect = viewport.getBoundingClientRect();
      const px = Math.min(1024, Math.max(0, Math.round(((e.clientX - rect.left) / rect.width) * 1024)));
      const py = Math.min(1024, Math.max(0, Math.round(((e.clientY - rect.top) / rect.height) * 1024)));

      const lat = (31.97 - (py / 1024) * 0.02).toFixed(4);
      const lng = (97.24 - (px / 1024) * 0.03).toFixed(4);

      setCursorCoords({ x: px, y: py, lat: `${lat}° N`, lng: `${lng}° W` });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setIsDraggingDivider(false);
  };

  const handleDividerMouseDown = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsDraggingDivider(true);
  };

  const handleZoom = (direction: 'in' | 'out') => {
    setZoom((z) => {
      const step = 0.25;
      return direction === 'in' ? Math.min(3, z + step) : Math.max(0.5, z - step);
    });
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  // SVG Helper
  const getLinePath = (coords: number[][]) => {
    return coords.map((c, i) => `${i === 0 ? 'M' : 'L'} ${c[0]} ${c[1]}`).join(' ');
  };

  // Extraction of SVG paths
  const addedLines = addedGeojson?.features?.filter((f: any) => f.geometry?.type === 'LineString') || [];
  const addedNodes = addedGeojson?.features?.filter((f: any) => f.geometry?.type === 'Point') || [];

  const removedLines = removedGeojson?.features?.filter((f: any) => f.geometry?.type === 'LineString') || [];
  const removedNodes = removedGeojson?.features?.filter((f: any) => f.geometry?.type === 'Point') || [];

  const unchangedLines = unchangedGeojson?.features?.filter((f: any) => f.geometry?.type === 'LineString') || [];
  const unchangedNodes = unchangedGeojson?.features?.filter((f: any) => f.geometry?.type === 'Point') || [];

  return (
    <div className="h-full flex overflow-hidden bg-[#070a0e]">
      {/* Left panel: configurations, uploads, and selectors */}
      <div className="w-80 border-r border-[#1f242c] bg-[#0b0f14]/40 flex flex-col justify-between overflow-y-auto shrink-0 p-5 space-y-6">
        <div className="space-y-5">
          {/* Metadata section */}
          <div className="space-y-2">
            <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider font-mono">1. Metadata</label>
            <input 
              type="text" 
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Comparison Project Name"
              className="w-full text-xs bg-gray-900 border border-[#1f242c] rounded px-3 py-2 text-white outline-none focus:border-emerald-500/50"
            />
            <input 
              type="text" 
              value={studyArea}
              onChange={(e) => setStudyArea(e.target.value)}
              placeholder="Region / Location"
              className="w-full text-xs bg-gray-900 border border-[#1f242c] rounded px-3 py-2 text-white outline-none focus:border-emerald-500/50"
            />
          </div>

          {/* Epoch A Configuration */}
          <div className="space-y-2.5 border-t border-gray-800/60 pt-3">
            <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider font-mono">2. Epoch A (Before/Baseline)</label>
            <div className="space-y-1">
              <select
                value={jobIdA}
                onChange={(e) => {
                  setJobIdA(e.target.value);
                  if (e.target.value) {
                    setFileA(null);
                    const match = availableRuns.find(r => r.id === e.target.value);
                    if (match) setYearA(match.year);
                  }
                }}
                className="w-full bg-gray-900 border border-[#1f242c] rounded p-1.5 text-xs text-white outline-none"
              >
                <option value="">-- Upload New Image --</option>
                {availableRuns.map(run => (
                  <option key={run.id} value={run.id}>{run.name}</option>
                ))}
              </select>
            </div>
            
            {!jobIdA && (
              <label 
                onDragOver={handleDragOver}
                onDrop={handleDropA}
                className="flex flex-col items-center justify-center border border-dashed border-[#1f242c] hover:border-emerald-500/30 rounded-lg p-3 bg-gray-900/35 cursor-pointer text-center"
              >
                <Upload className="h-4 w-4 text-gray-500 mb-1" />
                <span className="text-[10px] text-gray-300 font-medium">
                  {fileA ? fileA.name : 'Select or drop Image A'}
                </span>
                <input 
                  type="file" 
                  accept=".tif,.tiff,.jpg,.jpeg,.png"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      setFileA(e.target.files[0]);
                      setJobIdA('');
                    }
                  }}
                  className="hidden"
                />
              </label>
            )}
            <input 
              type="text" 
              value={yearA}
              onChange={(e) => setYearA(e.target.value)}
              placeholder="Year / Date A"
              className="w-full text-xs bg-gray-900 border border-[#1f242c] rounded px-3 py-1.5 text-white outline-none focus:border-emerald-500/50"
            />
          </div>

          {/* Epoch B Configuration */}
          <div className="space-y-2.5 border-t border-gray-800/60 pt-3">
            <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider font-mono">3. Epoch B (After/Current)</label>
            <div className="space-y-1">
              <select
                value={jobIdB}
                onChange={(e) => {
                  setJobIdB(e.target.value);
                  if (e.target.value) {
                    setFileB(null);
                    const match = availableRuns.find(r => r.id === e.target.value);
                    if (match) setYearB(match.year);
                  }
                }}
                className="w-full bg-gray-900 border border-[#1f242c] rounded p-1.5 text-xs text-white outline-none"
              >
                <option value="">-- Upload New Image --</option>
                {availableRuns.map(run => (
                  <option key={run.id} value={run.id}>{run.name}</option>
                ))}
              </select>
            </div>
            
            {!jobIdB && (
              <label 
                onDragOver={handleDragOver}
                onDrop={handleDropB}
                className="flex flex-col items-center justify-center border border-dashed border-[#1f242c] hover:border-emerald-500/30 rounded-lg p-3 bg-gray-900/35 cursor-pointer text-center"
              >
                <Upload className="h-4 w-4 text-gray-500 mb-1" />
                <span className="text-[10px] text-gray-300 font-medium">
                  {fileB ? fileB.name : 'Select or drop Image B'}
                </span>
                <input 
                  type="file" 
                  accept=".tif,.tiff,.jpg,.jpeg,.png"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      setFileB(e.target.files[0]);
                      setJobIdB('');
                    }
                  }}
                  className="hidden"
                />
              </label>
            )}
            <input 
              type="text" 
              value={yearB}
              onChange={(e) => setYearB(e.target.value)}
              placeholder="Year / Date B"
              className="w-full text-xs bg-gray-900 border border-[#1f242c] rounded px-3 py-1.5 text-white outline-none focus:border-emerald-500/50"
            />
          </div>

          {/* Model parameters weights */}
          <div className="space-y-3 border-t border-gray-800/60 pt-3">
            <div className="flex justify-between items-center">
              <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider font-mono">4. Parameter Weights</label>
              <button 
                onClick={() => setShowAdvanced(!showAdvanced)} 
                className="text-[10px] text-emerald-400 hover:text-emerald-300 underline font-mono"
              >
                {showAdvanced ? 'Collapse' : 'Advanced'}
              </button>
            </div>

            {showAdvanced && (
              <div className="space-y-3 text-xs font-mono">
                <div className="space-y-1">
                  <div className="flex justify-between text-gray-400">
                    <span>Confidence Thresh:</span>
                    <span className="text-emerald-400">{config.threshold}</span>
                  </div>
                  <input 
                    type="range" 
                    min="0.10" 
                    max="0.90" 
                    step="0.05"
                    value={config.threshold}
                    onChange={(e) => setConfig({ ...config, threshold: parseFloat(e.target.value) })}
                    className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                  />
                </div>

                <div className="space-y-1">
                  <div className="flex justify-between text-gray-400">
                    <span>Closing Radius (px):</span>
                    <span className="text-white">{config.closingRadius}</span>
                  </div>
                  <input 
                    type="range" 
                    min="1" 
                    max="15" 
                    value={config.closingRadius}
                    onChange={(e) => setConfig({ ...config, closingRadius: parseInt(e.target.value) })}
                    className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                  />
                </div>

                <div className="space-y-1">
                  <div className="flex justify-between text-gray-400">
                    <span>Min Object Size (px):</span>
                    <span className="text-white">{config.minObjectSize}</span>
                  </div>
                  <input 
                    type="range" 
                    min="8" 
                    max="128" 
                    value={config.minObjectSize}
                    onChange={(e) => setConfig({ ...config, minObjectSize: parseInt(e.target.value) })}
                    className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                  />
                </div>

                <div className="flex items-center justify-between border-t border-gray-800/40 pt-2 text-gray-400">
                  <span>Use AI U-Net Model:</span>
                  <input 
                    type="checkbox" 
                    checked={config.useModel}
                    onChange={(e) => setConfig({ ...config, useModel: e.target.checked })}
                    className="h-3.5 w-3.5 cursor-pointer accent-emerald-500"
                  />
                </div>
              </div>
            )}
          </div>

          {/* Trigger button */}
          <button
            onClick={triggerComparison}
            disabled={isProcessing}
            className={`w-full py-2.5 px-4 rounded text-xs font-mono font-bold uppercase tracking-wider transition-all flex items-center justify-center gap-2 cursor-pointer ${
              isProcessing
                ? 'bg-gray-800 text-gray-500 cursor-not-allowed'
                : 'bg-emerald-500 text-[#070a0e] hover:bg-emerald-400 shadow-md shadow-emerald-500/10'
            }`}
          >
            {isProcessing ? (
              <>
                <div className="h-3 w-3 border-2 border-gray-500 border-t-emerald-400 rounded-full animate-spin" />
                <span>Comparing Epochs...</span>
              </>
            ) : (
              <>
                <Play className="h-3.5 w-3.5 fill-current" />
                <span>Run Spatial Comparison</span>
              </>
            )}
          </button>

          {/* Dynamic Change Statistics card */}
          {comparisonResult && (
            <div className="space-y-2.5 border-t border-gray-800/60 pt-3">
              <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider font-mono">Change Statistics</label>
              <div className="border border-[#1f242c] rounded bg-gray-900/40 p-4 space-y-3 text-xs font-mono">
                <div className="flex justify-between items-center text-gray-400">
                  <span>Total Added Roads:</span>
                  <span className="text-emerald-400 font-bold">+{comparisonResult.addedLengthKm} km</span>
                </div>
                <div className="flex justify-between items-center text-gray-400">
                  <span>Removed/Abandoned:</span>
                  <span className="text-red-400 font-bold">-{comparisonResult.removedLengthKm} km</span>
                </div>
                <div className="flex justify-between items-center text-gray-400">
                  <span>Δ Total Length:</span>
                  <span className={`font-bold ${comparisonResult.deltaLengthKm >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {comparisonResult.deltaLengthKm >= 0 ? '+' : ''}{comparisonResult.deltaLengthKm} km
                  </span>
                </div>
                <div className="flex justify-between items-center text-gray-400">
                  <span>Δ Connectivity:</span>
                  <span className={`font-bold ${comparisonResult.deltaConnectivity >= 0 ? 'text-cyan-400' : 'text-red-400'}`}>
                    {comparisonResult.deltaConnectivity >= 0 ? '+' : ''}{comparisonResult.deltaConnectivity}%
                  </span>
                </div>
                <div className="flex justify-between items-center text-gray-400">
                  <span>Δ Junction Nodes:</span>
                  <span className={`font-bold ${comparisonResult.deltaJunctions >= 0 ? 'text-white' : 'text-red-400'}`}>
                    {comparisonResult.deltaJunctions >= 0 ? '+' : ''}{comparisonResult.deltaJunctions}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>

        <div className="p-4 border-t border-[#1f242c] bg-gray-900/10 text-[10px] text-gray-500 rounded flex gap-2">
          <HelpCircle className="h-4 w-4 shrink-0 text-emerald-400" />
          <span>Spatial diff compares geodetic centerline paths between Epoch A and B.</span>
        </div>
      </div>

      {/* Center canvas content */}
      <div className="flex-1 flex flex-col p-6 space-y-6 relative overflow-hidden min-w-0">
        {errorMsg && (
          <div className="bg-red-500/10 border border-red-500/25 p-4 rounded text-xs text-red-400 font-mono flex items-start gap-2">
            <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
            <div>
              <strong>Comparison Failure:</strong> {errorMsg}
            </div>
          </div>
        )}

        <div 
          className="flex-1 flex flex-col bg-[#0d121a] border border-[#1f242c] rounded overflow-hidden relative select-none"
          ref={containerRef}
        >
          {isProcessing ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center p-8 space-y-4">
              <div className="h-10 w-10 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
              <div className="space-y-1">
                <h4 className="text-xs font-semibold text-white tracking-widest font-mono uppercase">ANALYSIS IN PROGRESS</h4>
                <p className="text-[11px] text-gray-400 max-w-sm leading-relaxed">
                  Extracting dual-epoch centerline skeletons, performing geodetic buffering, and aligning road graphs.
                </p>
              </div>
            </div>
          ) : !comparisonResult ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center p-8 space-y-4">
              <div className="text-gray-600 bg-gray-900/60 p-4 rounded-full border border-[#1f242c] relative">
                <Layers className="h-10 w-10 text-gray-500" />
              </div>
              <div className="space-y-1">
                <h4 className="text-xs font-semibold text-white tracking-widest font-mono uppercase">NO COMPARISON RUN LOADED</h4>
                <p className="text-[11px] text-gray-400 max-w-sm leading-relaxed">
                  Configure Epoch A and Epoch B in the left panel using satellite images or previous analysis results.
                </p>
              </div>
            </div>
          ) : (
            <>
              {/* Toolbar controls */}
              <div className="absolute top-4 left-4 right-4 flex justify-between items-center pointer-events-none z-10">
                {/* Visualizations toggles */}
                <div className="flex gap-2 pointer-events-auto">
                  <div className="flex gap-1 bg-[#0b0f14]/85 border border-[#1f242c] p-1 rounded-md backdrop-blur-sm">
                    <button
                      onClick={() => setViewMode('side-by-side')}
                      className={`px-3 py-1.5 text-[9px] font-mono font-semibold tracking-wider rounded uppercase transition-colors cursor-pointer ${
                        viewMode === 'side-by-side' 
                          ? 'bg-emerald-500/25 text-emerald-400 border border-emerald-500/25' 
                          : 'text-gray-400 hover:text-white hover:bg-gray-800/35 border border-transparent'
                      }`}
                    >
                      Side-by-Side
                    </button>
                    <button
                      onClick={() => setViewMode('swipe')}
                      className={`px-3 py-1.5 text-[9px] font-mono font-semibold tracking-wider rounded uppercase transition-colors cursor-pointer ${
                        viewMode === 'swipe' 
                          ? 'bg-emerald-500/25 text-emerald-400 border border-emerald-500/25' 
                          : 'text-gray-400 hover:text-white hover:bg-gray-800/35 border border-transparent'
                      }`}
                    >
                      Swipe
                    </button>
                  </div>

                  {/* Overlays toggle selection */}
                  <div className="flex gap-1 bg-[#0b0f14]/85 border border-[#1f242c] p-1 rounded-md backdrop-blur-sm">
                    <button
                      onClick={() => setShowAdded(!showAdded)}
                      className={`px-2 py-1.5 text-[9px] font-mono font-semibold tracking-wider rounded uppercase transition-colors cursor-pointer flex items-center gap-1.5 ${
                        showAdded 
                          ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/25' 
                          : 'text-gray-500 hover:text-white border border-transparent'
                      }`}
                    >
                      <div className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                      Added
                    </button>
                    <button
                      onClick={() => setShowRemoved(!showRemoved)}
                      className={`px-2 py-1.5 text-[9px] font-mono font-semibold tracking-wider rounded uppercase transition-colors cursor-pointer flex items-center gap-1.5 ${
                        showRemoved 
                          ? 'bg-red-500/10 text-red-400 border border-red-500/25' 
                          : 'text-gray-500 hover:text-white border border-transparent'
                      }`}
                    >
                      <div className="h-1.5 w-1.5 rounded-full bg-red-400" />
                      Removed
                    </button>
                    <button
                      onClick={() => setShowUnchanged(!showUnchanged)}
                      className={`px-2 py-1.5 text-[9px] font-mono font-semibold tracking-wider rounded uppercase transition-colors cursor-pointer flex items-center gap-1.5 ${
                        showUnchanged 
                          ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/25' 
                          : 'text-gray-500 hover:text-white border border-transparent'
                      }`}
                    >
                      <div className="h-1.5 w-1.5 rounded-full bg-cyan-400" />
                      Unchanged
                    </button>
                    <button
                      onClick={() => setShowNodes(!showNodes)}
                      className={`px-2 py-1.5 text-[9px] font-mono font-semibold tracking-wider rounded uppercase transition-colors cursor-pointer ${
                        showNodes 
                          ? 'bg-cyan-500/10 text-cyan-300 border border-cyan-500/25' 
                          : 'text-gray-500 hover:text-white border border-transparent'
                      }`}
                    >
                      Nodes
                    </button>
                  </div>
                </div>

                {/* Zoom tools */}
                <div className="flex items-center gap-2 pointer-events-auto bg-[#0b0f14]/85 border border-[#1f242c] p-1 rounded-md backdrop-blur-sm text-xs">
                  <button onClick={() => handleZoom('in')} className="p-1 hover:text-white text-gray-400" title="Zoom In"><ZoomIn className="h-3.5 w-3.5" /></button>
                  <button onClick={() => handleZoom('out')} className="p-1 hover:text-white text-gray-400" title="Zoom Out"><ZoomOut className="h-3.5 w-3.5" /></button>
                  <button onClick={resetView} className="p-1 hover:text-white text-gray-400" title="Reset View"><RotateCcw className="h-3.5 w-3.5" /></button>
                </div>
              </div>

              {/* Viewport Canvas wrapper */}
              <div 
                className="flex-1 relative cursor-crosshair overflow-hidden"
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
              >
                {geojsonLoading && (
                  <div className="absolute inset-0 bg-[#0d121a]/70 z-40 flex items-center justify-center font-mono text-xs text-gray-400 gap-2">
                    <div className="h-3 w-3 border border-emerald-500 border-t-transparent rounded-full animate-spin" />
                    Rendering temporal vectors...
                  </div>
                )}

                {viewMode === 'side-by-side' ? (
                  <div className="flex w-full h-full divide-x divide-[#1f242c]">
                    {/* Viewport Epoch A (Left) */}
                    <div className="flex-1 h-full relative overflow-hidden comparison-viewport">
                      <div className="absolute top-4 left-4 z-20 bg-[#0b0f14]/80 border border-[#1f242c] px-3 py-1.5 rounded text-xs font-mono font-semibold text-white">
                        EPOCH A: {yearA} Baseline
                      </div>
                      <div className="absolute inset-0 gis-grid opacity-20 pointer-events-none" />
                      
                      <img 
                        src={comparisonResult.imageAUrl}
                        alt={`Epoch A: ${yearA}`}
                        className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                        style={{
                          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                          transformOrigin: 'center center'
                        }}
                      />

                      {/* SVG overlay Epoch A: Unchanged + Removed */}
                      <svg 
                        className="absolute inset-0 w-full h-full pointer-events-none"
                        viewBox="0 0 1024 1024"
                        style={{
                          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                          transformOrigin: 'center center'
                        }}
                      >
                        {/* Unchanged Roads (Gray/Cyan) */}
                        {showUnchanged && unchangedLines.length > 0 && (
                          <g fill="none" stroke="#64748b" strokeWidth="2.5" strokeLinecap="round" opacity="0.8">
                            {unchangedLines.map((line: any, idx: number) => (
                              <path key={`u-a-${idx}`} d={getLinePath(line.geometry.coordinates)} />
                            ))}
                          </g>
                        )}
                        {/* Removed Roads (Red) */}
                        {showRemoved && removedLines.length > 0 && (
                          <g fill="none" stroke="#ef4444" strokeWidth="3" strokeLinecap="round" opacity="0.9">
                            {removedLines.map((line: any, idx: number) => (
                              <path key={`r-a-${idx}`} d={getLinePath(line.geometry.coordinates)} />
                            ))}
                          </g>
                        )}
                        {/* Junction points */}
                        {showNodes && (
                          <g fill="#22d3ee" stroke="#0b0f14" strokeWidth="0.8" opacity="0.85">
                            {showUnchanged && unchangedNodes.map((node: any, idx: number) => (
                              <circle key={`un-a-${idx}`} cx={node.geometry.coordinates[0]} cy={node.geometry.coordinates[1]} r="4" />
                            ))}
                            {showRemoved && removedNodes.map((node: any, idx: number) => (
                              <circle key={`rn-a-${idx}`} cx={node.geometry.coordinates[0]} cy={node.geometry.coordinates[1]} r="4" fill="#ef4444" />
                            ))}
                          </g>
                        )}
                      </svg>
                    </div>

                    {/* Viewport Epoch B (Right) */}
                    <div className="flex-1 h-full relative overflow-hidden comparison-viewport">
                      <div className="absolute top-4 left-4 z-20 bg-[#0b0f14]/80 border border-[#1f242c] px-3 py-1.5 rounded text-xs font-mono font-semibold text-white">
                        EPOCH B: {yearB} Extraction
                      </div>
                      <div className="absolute inset-0 gis-grid opacity-20 pointer-events-none" />

                      <img 
                        src={comparisonResult.imageBUrl}
                        alt={`Epoch B: ${yearB}`}
                        className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                        style={{
                          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                          transformOrigin: 'center center'
                        }}
                      />

                      {/* SVG overlay Epoch B: Unchanged + Added */}
                      <svg 
                        className="absolute inset-0 w-full h-full pointer-events-none"
                        viewBox="0 0 1024 1024"
                        style={{
                          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                          transformOrigin: 'center center'
                        }}
                      >
                        {/* Unchanged Roads (Gray/Cyan) */}
                        {showUnchanged && unchangedLines.length > 0 && (
                          <g fill="none" stroke="#64748b" strokeWidth="2.5" strokeLinecap="round" opacity="0.8">
                            {unchangedLines.map((line: any, idx: number) => (
                              <path key={`u-b-${idx}`} d={getLinePath(line.geometry.coordinates)} />
                            ))}
                          </g>
                        )}
                        {/* Added Roads (Green) */}
                        {showAdded && addedLines.length > 0 && (
                          <g fill="none" stroke="#10b981" strokeWidth="3" strokeLinecap="round" opacity="0.95">
                            {addedLines.map((line: any, idx: number) => (
                              <path key={`a-b-${idx}`} d={getLinePath(line.geometry.coordinates)} />
                            ))}
                          </g>
                        )}
                        {/* Junction points */}
                        {showNodes && (
                          <g fill="#22d3ee" stroke="#0b0f14" strokeWidth="0.8" opacity="0.85">
                            {showUnchanged && unchangedNodes.map((node: any, idx: number) => (
                              <circle key={`un-b-${idx}`} cx={node.geometry.coordinates[0]} cy={node.geometry.coordinates[1]} r="4" />
                            ))}
                            {showAdded && addedNodes.map((node: any, idx: number) => (
                              <circle key={`an-b-${idx}`} cx={node.geometry.coordinates[0]} cy={node.geometry.coordinates[1]} r="4.5" fill="#10b981" />
                            ))}
                          </g>
                        )}
                      </svg>
                    </div>
                  </div>
                ) : (
                  <div className="w-full h-full relative overflow-hidden comparison-viewport">
                    <div className="absolute inset-0 gis-grid opacity-20 pointer-events-none" />

                    {/* Base Layer: Epoch B (Underneath) */}
                    <div className="absolute inset-0">
                      <img 
                        src={comparisonResult.imageBUrl}
                        alt={`Epoch B Base`}
                        className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                        style={{
                          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                          transformOrigin: 'center center'
                        }}
                      />
                      <svg 
                        className="absolute inset-0 w-full h-full pointer-events-none"
                        viewBox="0 0 1024 1024"
                        style={{
                          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                          transformOrigin: 'center center'
                        }}
                      >
                        {showUnchanged && unchangedLines.length > 0 && (
                          <g fill="none" stroke="#64748b" strokeWidth="2.5" strokeLinecap="round" opacity="0.8">
                            {unchangedLines.map((line: any, idx: number) => (
                              <path key={`su-b-${idx}`} d={getLinePath(line.geometry.coordinates)} />
                            ))}
                          </g>
                        )}
                        {showAdded && addedLines.length > 0 && (
                          <g fill="none" stroke="#10b981" strokeWidth="3" strokeLinecap="round" opacity="0.95">
                            {addedLines.map((line: any, idx: number) => (
                              <path key={`sa-b-${idx}`} d={getLinePath(line.geometry.coordinates)} />
                            ))}
                          </g>
                        )}
                        {showNodes && (
                          <g fill="#22d3ee" stroke="#0b0f14" strokeWidth="0.8" opacity="0.85">
                            {showUnchanged && unchangedNodes.map((node: any, idx: number) => (
                              <circle key={`sun-b-${idx}`} cx={node.geometry.coordinates[0]} cy={node.geometry.coordinates[1]} r="4" />
                            ))}
                            {showAdded && addedNodes.map((node: any, idx: number) => (
                              <circle key={`san-b-${idx}`} cx={node.geometry.coordinates[0]} cy={node.geometry.coordinates[1]} r="4" fill="#10b981" />
                            ))}
                          </g>
                        )}
                      </svg>
                    </div>

                    {/* Clip Layer: Epoch A (Clipped to show on left) */}
                    <div 
                      className="absolute inset-0 overflow-hidden pointer-events-none"
                      style={{
                        clipPath: `inset(0 ${100 - swipeOffset}% 0 0)`
                      }}
                    >
                      <img 
                        src={comparisonResult.imageAUrl}
                        alt={`Epoch A Overlay`}
                        className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                        style={{
                          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                          transformOrigin: 'center center'
                        }}
                      />
                      <svg 
                        className="absolute inset-0 w-full h-full pointer-events-none"
                        viewBox="0 0 1024 1024"
                        style={{
                          transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                          transformOrigin: 'center center'
                        }}
                      >
                        {showUnchanged && unchangedLines.length > 0 && (
                          <g fill="none" stroke="#64748b" strokeWidth="2.5" strokeLinecap="round" opacity="0.8">
                            {unchangedLines.map((line: any, idx: number) => (
                              <path key={`su-a-${idx}`} d={getLinePath(line.geometry.coordinates)} />
                            ))}
                          </g>
                        )}
                        {showRemoved && removedLines.length > 0 && (
                          <g fill="none" stroke="#ef4444" strokeWidth="3" strokeLinecap="round" opacity="0.9">
                            {removedLines.map((line: any, idx: number) => (
                              <path key={`sr-a-${idx}`} d={getLinePath(line.geometry.coordinates)} />
                            ))}
                          </g>
                        )}
                        {showNodes && (
                          <g fill="#22d3ee" stroke="#0b0f14" strokeWidth="0.8" opacity="0.85">
                            {showUnchanged && unchangedNodes.map((node: any, idx: number) => (
                              <circle key={`sun-a-${idx}`} cx={node.geometry.coordinates[0]} cy={node.geometry.coordinates[1]} r="4" />
                            ))}
                            {showRemoved && removedNodes.map((node: any, idx: number) => (
                              <circle key={`srn-a-${idx}`} cx={node.geometry.coordinates[0]} cy={node.geometry.coordinates[1]} r="4" fill="#ef4444" />
                            ))}
                          </g>
                        )}
                      </svg>
                    </div>

                    {/* Viewport Labels */}
                    <div className="absolute top-4 left-4 z-20 flex gap-2">
                      <div className="bg-[#0b0f14]/80 border border-[#1f242c] px-3 py-1.5 rounded text-xs font-mono font-semibold text-white">
                        Swipe Viewport: {yearA} (Left) vs {yearB} (Right)
                      </div>
                    </div>

                    {/* Draggable vertical divider */}
                    <div 
                      className="absolute top-0 bottom-0 w-0.5 bg-emerald-500 z-30 cursor-col-resize flex items-center justify-center pointer-events-auto"
                      style={{ left: `${swipeOffset}%` }}
                      onMouseDown={handleDividerMouseDown}
                    >
                      <div className="bg-emerald-500 text-[#070a0e] p-1 rounded-full shadow-lg -translate-x-1/2 select-none">
                        <Grid className="h-3 w-3" />
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Status Metadata Bar */}
              <div className="h-8 border-t border-[#1f242c] bg-[#0b0f14] flex justify-between items-center px-4 text-[10px] font-mono text-gray-500 shrink-0">
                <div>CURSOR: {cursorCoords.lat}, {cursorCoords.lng} (x: {cursorCoords.x}px, y: {cursorCoords.y}px)</div>
                <div className="flex gap-4">
                  <span className="uppercase text-emerald-400">MODE: {viewMode}</span>
                  <span>·</span>
                  <span>COMPARE ID: {comparisonResult.projectId}</span>
                  <span>·</span>
                  <span>ZOOM: {Math.round(zoom * 100)}%</span>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
