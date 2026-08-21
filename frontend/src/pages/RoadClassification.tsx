import React, { useState, useEffect, useRef } from 'react';
import { 
  Tag, 
  Layers, 
  ShieldCheck, 
  AlertTriangle, 
  Info, 
  CheckCircle2, 
  Activity,
  Sliders,
  Sparkles,
  HelpCircle,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Grid,
  RefreshCw
} from 'lucide-react';
import { apiService } from '../services/api';
import { AnalysisResult } from '../types';

export default function RoadClassification() {
  const [activeResult, setActiveResult] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Centralized Threshold Configurations
  const [arterialWidth, setArterialWidth] = useState(12.0);
  const [collectorWidth, setCollectorWidth] = useState(8.0);
  const [localWidth, setLocalWidth] = useState(5.0);
  const [arterialCurvature, setArterialCurvature] = useState(0.15);
  const [roughnessThreshold, setRoughnessThreshold] = useState(0.45);
  const [showThresholdConfigs, setShowThresholdConfigs] = useState(true);

  // Overlay geojsonData
  const [geojsonData, setGeojsonData] = useState<any>(null);
  const [geojsonLoading, setGeojsonLoading] = useState(false);

  // Class filters
  const [visibleClasses, setVisibleClasses] = useState({
    arterial: true,
    collector: true,
    local: true,
    minor: true
  });

  // Quality filters
  const [visibleQuality, setVisibleQuality] = useState({
    high: true,
    moderate: true,
    review: true
  });

  // Map viewport states
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [cursorCoords, setCursorCoords] = useState({ x: 0, y: 0, lat: '31.9700° N', lng: '97.2400° W' });

  const containerRef = useRef<HTMLDivElement>(null);

  // Scan sessionStorage for the latest run
  useEffect(() => {
    let latestData: any = null;
    for (let i = 0; i < sessionStorage.length; i++) {
      const key = sessionStorage.key(i);
      if (key && key.startsWith('analysis_result_')) {
        try {
          const raw = sessionStorage.getItem(key);
          if (raw) {
            const data = JSON.parse(raw);
            if (!latestData || new Date(data.analysisDate || 0) > new Date(latestData.analysisDate || 0)) {
              latestData = data;
            }
          }
        } catch {}
      }
    }

    if (latestData) {
      if (!latestData.classificationAvailable) {
        triggerClassification(latestData.projectId);
      } else {
        setActiveResult(latestData);
        loadGeojsonOverlays(latestData.projectId);
      }
    }
  }, []);

  const triggerClassification = async (projectId: string) => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const updated = await apiService.runClassification(
        projectId,
        arterialWidth,
        collectorWidth,
        localWidth,
        arterialCurvature,
        roughnessThreshold
      );
      setActiveResult(updated);
      sessionStorage.setItem(`analysis_result_${updated.projectId}`, JSON.stringify(updated));
      await loadGeojsonOverlays(updated.projectId);
    } catch (err: any) {
      console.error("Classification calculation failed:", err);
      setErrorMsg(err.message || "Failed to calculate road classifications.");
    } finally {
      setIsLoading(false);
    }
  };

  const loadGeojsonOverlays = async (projectId: string) => {
    setGeojsonLoading(true);
    try {
      const url = apiService.getLayerUrl(projectId, 'geojson');
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        setGeojsonData(data);
      }
    } catch (e) {
      console.error("Failed to load road network geojson:", e);
    } finally {
      setGeojsonLoading(false);
    }
  };

  // Map panning/zooming controls
  const handleMouseDown = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('button') || (e.target as HTMLElement).closest('select') || (e.target as HTMLElement).closest('input')) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    }

    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      const px = Math.min(1024, Math.max(0, Math.round(((e.clientX - rect.left) / rect.width) * 1024)));
      const py = Math.min(1024, Math.max(0, Math.round(((e.clientY - rect.top) / rect.height) * 1024)));

      const lat = (31.97 - (py / 1024) * 0.02).toFixed(4);
      const lng = (97.24 - (px / 1024) * 0.03).toFixed(4);

      setCursorCoords({ x: px, y: py, lat: `${lat}° N`, lng: `${lng}° W` });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
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

  const getLinePath = (coords: number[][]) => {
    return coords.map((c, i) => `${i === 0 ? 'M' : 'L'} ${c[0]} ${c[1]}`).join(' ');
  };

  // Filter SVG Features
  const features = geojsonData?.features || [];
  const lines = features.filter((f: any) => {
    if (f.geometry?.type !== 'LineString') return false;
    const rc = f.properties?.road_class || 'Local Road';
    const qt = f.properties?.quality_tier || 'Moderate';

    const classKeys: Record<string, keyof typeof visibleClasses> = {
      'Primary Arterial': 'arterial',
      'Secondary Collector': 'collector',
      'Local Road': 'local',
      'Minor / Unpaved': 'minor'
    };

    const qualityKeys: Record<string, keyof typeof visibleQuality> = {
      'High Quality': 'high',
      'Moderate': 'moderate',
      'Requires Review': 'review'
    };

    const isClassVisible = visibleClasses[classKeys[rc] || 'local'];
    const isQualityVisible = visibleQuality[qualityKeys[qt] || 'moderate'];

    return isClassVisible && isQualityVisible;
  });

  const getStrokeColor = (rc: string) => {
    switch (rc) {
      case 'Primary Arterial': return '#0ea5e9'; // sky-500
      case 'Secondary Collector': return '#f59e0b'; // amber-500
      case 'Local Road': return '#10b981'; // emerald-500
      case 'Minor / Unpaved': return '#ef4444'; // rose-500
      default: return '#a855f7'; // purple-500
    }
  };

  return (
    <div className="h-full flex overflow-hidden bg-[#070a0e]">
      {/* 1. Left Sidebar: Classification & Filter Status */}
      <div className="w-84 border-r border-[#1f242c] bg-[#0b0f14]/40 flex flex-col justify-between overflow-y-auto shrink-0 p-5 space-y-6">
        <div className="space-y-5">
          {/* Header */}
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Tag className="h-4 w-4 text-emerald-400" />
              <h2 className="text-xs font-semibold text-white tracking-wider font-mono uppercase">
                Road Classification
              </h2>
            </div>
            <p className="text-[11px] text-gray-400 leading-relaxed font-mono">
              Structural classification telemetry derived from width measurements.
            </p>
          </div>

          {/* Classification Status Notice */}
          {activeResult && activeResult.classificationAvailable && (
            <div className="p-3.5 border border-emerald-500/25 bg-emerald-950/10 rounded-lg space-y-2 text-xs font-mono">
              <div className="flex items-center gap-2 text-emerald-400 font-semibold">
                <CheckCircle2 className="h-4 w-4 shrink-0" />
                <span>Classification Active</span>
              </div>
              <p className="text-[11px] text-gray-300 leading-relaxed">
                Analysis complete. Classified segments based on mask-derived width profiles.
              </p>
            </div>
          )}

          {errorMsg && (
            <div className="p-3 border border-red-500/30 bg-red-950/20 rounded text-[11px] font-mono text-red-400">
              {errorMsg}
            </div>
          )}

          {isLoading && (
            <div className="p-4 border border-emerald-500/35 bg-emerald-500/5 rounded text-xs font-mono text-emerald-400 flex items-center gap-2">
              <div className="h-3.5 w-3.5 border border-emerald-500 border-t-transparent rounded-full animate-spin" />
              Calculating road classifications...
            </div>
          )}

          {/* Centralized Config Threshold Sliders */}
          {activeResult && (
            <div className="space-y-3 border-t border-gray-800/60 pt-3">
              <div className="flex justify-between items-center">
                <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider font-mono flex items-center gap-1.5">
                  <Sliders className="h-3.5 w-3.5 text-emerald-400" />
                  Heuristic Thresholds
                </label>
                <button 
                  onClick={() => setShowThresholdConfigs(!showThresholdConfigs)}
                  className="text-[10px] text-emerald-400 hover:text-emerald-300 underline font-mono cursor-pointer bg-transparent border-none outline-none"
                >
                  {showThresholdConfigs ? 'Hide' : 'Show'}
                </button>
              </div>

              {showThresholdConfigs && (
                <div className="space-y-3.5 text-[11px] font-mono text-gray-300">
                  <div className="space-y-1">
                    <div className="flex justify-between text-gray-400">
                      <span>Arterial Width (px):</span>
                      <span className="text-white">{arterialWidth} px</span>
                    </div>
                    <input 
                      type="range" min="8" max="20" step="0.5" value={arterialWidth}
                      onChange={(e) => setArterialWidth(parseFloat(e.target.value))}
                      className="w-full h-1 bg-gray-800 rounded appearance-none accent-emerald-500 cursor-pointer"
                    />
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between text-gray-400">
                      <span>Collector Width (px):</span>
                      <span className="text-white">{collectorWidth} px</span>
                    </div>
                    <input 
                      type="range" min="6" max="12" step="0.5" value={collectorWidth}
                      onChange={(e) => setCollectorWidth(parseFloat(e.target.value))}
                      className="w-full h-1 bg-gray-800 rounded appearance-none accent-emerald-500 cursor-pointer"
                    />
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between text-gray-400">
                      <span>Local Road Width (px):</span>
                      <span className="text-white">{localWidth} px</span>
                    </div>
                    <input 
                      type="range" min="3" max="8" step="0.5" value={localWidth}
                      onChange={(e) => setLocalWidth(parseFloat(e.target.value))}
                      className="w-full h-1 bg-gray-800 rounded appearance-none accent-emerald-500 cursor-pointer"
                    />
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between text-gray-400">
                      <span>Arterial Curvature:</span>
                      <span className="text-white">{arterialCurvature}</span>
                    </div>
                    <input 
                      type="range" min="0.05" max="0.30" step="0.01" value={arterialCurvature}
                      onChange={(e) => setArterialCurvature(parseFloat(e.target.value))}
                      className="w-full h-1 bg-gray-800 rounded appearance-none accent-emerald-500 cursor-pointer"
                    />
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between text-gray-400">
                      <span>Roughness Limit:</span>
                      <span className="text-white">{roughnessThreshold}</span>
                    </div>
                    <input 
                      type="range" min="0.20" max="0.80" step="0.05" value={roughnessThreshold}
                      onChange={(e) => setRoughnessThreshold(parseFloat(e.target.value))}
                      className="w-full h-1 bg-gray-800 rounded appearance-none accent-emerald-500 cursor-pointer"
                    />
                  </div>

                  <button
                    onClick={() => triggerClassification(activeResult.projectId)}
                    disabled={isLoading}
                    className="w-full py-2 bg-emerald-500 hover:bg-emerald-400 text-[#070a0e] rounded font-bold uppercase tracking-wider transition-colors flex items-center justify-center gap-1.5 cursor-pointer text-xs"
                  >
                    <RefreshCw className={`h-3.5 w-3.5 ${isLoading ? 'animate-spin' : ''}`} />
                    <span>Recalculate Classes</span>
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Filters Toggles */}
          {activeResult && (
            <div className="space-y-4 font-mono text-xs text-gray-300">
              {/* Functional Class Toggle */}
              <div className="space-y-2 border-t border-gray-800/60 pt-3">
                <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Functional Classes</label>
                <div className="space-y-1.5">
                  <label className="flex items-center justify-between p-2 rounded border border-[#1f242c] bg-gray-900/30 cursor-pointer hover:bg-gray-900/50">
                    <div className="flex items-center gap-2">
                      <input 
                        type="checkbox" 
                        checked={visibleClasses.arterial}
                        onChange={(e) => setVisibleClasses({ ...visibleClasses, arterial: e.target.checked })}
                        className="accent-[#0ea5e9]"
                      />
                      <span className="h-2 w-2 rounded-full bg-[#0ea5e9]" />
                      <span>Primary Arterial</span>
                    </div>
                  </label>

                  <label className="flex items-center justify-between p-2 rounded border border-[#1f242c] bg-gray-900/30 cursor-pointer hover:bg-gray-900/50">
                    <div className="flex items-center gap-2">
                      <input 
                        type="checkbox" 
                        checked={visibleClasses.collector}
                        onChange={(e) => setVisibleClasses({ ...visibleClasses, collector: e.target.checked })}
                        className="accent-[#f59e0b]"
                      />
                      <span className="h-2 w-2 rounded-full bg-[#f59e0b]" />
                      <span>Secondary Collector</span>
                    </div>
                  </label>

                  <label className="flex items-center justify-between p-2 rounded border border-[#1f242c] bg-gray-900/30 cursor-pointer hover:bg-gray-900/50">
                    <div className="flex items-center gap-2">
                      <input 
                        type="checkbox" 
                        checked={visibleClasses.local}
                        onChange={(e) => setVisibleClasses({ ...visibleClasses, local: e.target.checked })}
                        className="accent-[#10b981]"
                      />
                      <span className="h-2 w-2 rounded-full bg-[#10b981]" />
                      <span>Local Road</span>
                    </div>
                  </label>

                  <label className="flex items-center justify-between p-2 rounded border border-[#1f242c] bg-gray-900/30 cursor-pointer hover:bg-gray-900/50">
                    <div className="flex items-center gap-2">
                      <input 
                        type="checkbox" 
                        checked={visibleClasses.minor}
                        onChange={(e) => setVisibleClasses({ ...visibleClasses, minor: e.target.checked })}
                        className="accent-[#ef4444]"
                      />
                      <span className="h-2 w-2 rounded-full bg-[#ef4444]" />
                      <span>Minor / Unpaved</span>
                    </div>
                  </label>
                </div>
              </div>

              {/* Quality Tiers Toggle */}
              <div className="space-y-2 border-t border-gray-800/60 pt-3">
                <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Quality Tiers</label>
                <div className="space-y-1.5">
                  <label className="flex items-center justify-between p-2 rounded border border-[#1f242c] bg-gray-900/30 cursor-pointer hover:bg-gray-900/50">
                    <div className="flex items-center gap-2">
                      <input 
                        type="checkbox" 
                        checked={visibleQuality.high}
                        onChange={(e) => setVisibleQuality({ ...visibleQuality, high: e.target.checked })}
                        className="accent-emerald-500"
                      />
                      <span>High Quality (&ge;70%)</span>
                    </div>
                  </label>

                  <label className="flex items-center justify-between p-2 rounded border border-[#1f242c] bg-gray-900/30 cursor-pointer hover:bg-gray-900/50">
                    <div className="flex items-center gap-2">
                      <input 
                        type="checkbox" 
                        checked={visibleQuality.moderate}
                        onChange={(e) => setVisibleQuality({ ...visibleQuality, moderate: e.target.checked })}
                        className="accent-amber-500"
                      />
                      <span>Moderate (40-69%)</span>
                    </div>
                  </label>

                  <label className="flex items-center justify-between p-2 rounded border border-[#1f242c] bg-gray-900/30 cursor-pointer hover:bg-gray-900/50">
                    <div className="flex items-center gap-2">
                      <input 
                        type="checkbox" 
                        checked={visibleQuality.review}
                        onChange={(e) => setVisibleQuality({ ...visibleQuality, review: e.target.checked })}
                        className="accent-red-500"
                      />
                      <span>Requires Review (&lt;40%)</span>
                    </div>
                  </label>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer info */}
        <div className="pt-4 border-t border-[#1f242c] text-[10px] font-mono text-gray-500">
          <div>Model: U-Net Binary Road Extractor</div>
          <div>Classification: Width Telemetry Metrics</div>
        </div>
      </div>

      {/* 2. Main Map Viewport */}
      <div 
        className="flex-1 flex flex-col p-6 space-y-6 relative overflow-hidden min-w-0"
        ref={containerRef}
      >
        <div className="flex-1 flex flex-col bg-[#0d121a] border border-[#1f242c] rounded overflow-hidden relative select-none">
          {!activeResult ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center p-8 space-y-4 gis-grid">
              <div className="text-gray-600 bg-gray-900/60 p-4 rounded-full border border-[#1f242c] relative">
                <Tag className="h-10 w-10 text-gray-500" />
              </div>
              <div className="space-y-1">
                <h4 className="text-xs font-semibold text-white tracking-widest font-mono uppercase">NO ANALYSIS RUN LOADED</h4>
                <p className="text-[11px] text-gray-400 max-w-sm leading-relaxed">
                  Go to the workspace and run an extraction task first to build classification telemetry.
                </p>
              </div>
            </div>
          ) : (
            <>
              {/* Toolbar */}
              <div className="absolute top-4 left-4 right-4 flex justify-between items-center pointer-events-none z-10">
                <div className="bg-[#0b0f14]/85 border border-[#1f242c] px-3 py-1.5 rounded text-xs font-mono font-semibold text-white">
                  Road Classification Maps View
                </div>

                <div className="flex items-center gap-2 pointer-events-auto bg-[#0b0f14]/85 border border-[#1f242c] p-1 rounded-md backdrop-blur-sm text-xs">
                  <button onClick={() => handleZoom('in')} className="p-1 hover:text-white text-gray-400" title="Zoom In"><ZoomIn className="h-3.5 w-3.5" /></button>
                  <button onClick={() => handleZoom('out')} className="p-1 hover:text-white text-gray-400" title="Zoom Out"><ZoomOut className="h-3.5 w-3.5" /></button>
                  <button onClick={resetView} className="p-1 hover:text-white text-gray-400" title="Reset View"><RotateCcw className="h-3.5 w-3.5" /></button>
                </div>
              </div>

              {/* Viewport Canvas */}
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
                    Overlaying classification vectors...
                  </div>
                )}

                <img 
                  src={apiService.getLayerUrl(activeResult.projectId, 'original')}
                  alt={`Classification Imagery`}
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
                  <g fill="none" strokeWidth="3" strokeLinecap="round" opacity="0.85">
                    {lines.map((line: any, idx: number) => {
                      const rc = line.properties?.road_class || 'Local Road';
                      return (
                        <path 
                          key={`class-${idx}`} 
                          d={getLinePath(line.geometry.coordinates)} 
                          stroke={getStrokeColor(rc)}
                        />
                      );
                    })}
                  </g>
                </svg>
              </div>

              {/* Statusbar */}
              <div className="h-8 border-t border-[#1f242c] bg-[#0b0f14] flex justify-between items-center px-4 text-[10px] font-mono text-gray-500 shrink-0">
                <div>CURSOR: {cursorCoords.lat}, {cursorCoords.lng} (x: {cursorCoords.x}px, y: {cursorCoords.y}px)</div>
                <div className="flex gap-4">
                  <span>PROJECT ID: {activeResult.projectId}</span>
                  <span>·</span>
                  <span>ZOOM: {Math.round(zoom * 100)}%</span>
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* 3. Right Sidebar: Statistics Breakdown */}
      {activeResult && (
        <div className="w-80 border-l border-[#1f242c] bg-[#0b0f14]/40 flex flex-col justify-between overflow-y-auto shrink-0 p-5 space-y-6 text-xs font-mono">
          <div className="space-y-5">
            {/* Title */}
            <div className="space-y-1">
              <h3 className="text-xs font-semibold text-white tracking-wider uppercase">Classification Telemetry</h3>
              <p className="text-[11px] text-gray-400">Class and Quality distribution metrics.</p>
            </div>

            {/* Class distribution statistics table */}
            {activeResult.classificationStats?.distribution ? (
              <div className="space-y-3">
                <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Class Distribution</label>
                <div className="border border-[#1f242c] rounded overflow-hidden">
                  <table className="w-full text-left text-[11px] text-gray-400 bg-gray-900/20">
                    <thead className="bg-gray-900/50 text-[10px] text-gray-500 font-bold uppercase border-b border-[#1f242c]">
                      <tr>
                        <th className="p-2">Class</th>
                        <th className="p-2 text-right">Length</th>
                        <th className="p-2 text-right">%</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-[#1f242c]">
                      {Object.entries(activeResult.classificationStats.distribution).map(([rc, stat]: any) => (
                        <tr key={rc} className="hover:bg-gray-900/10">
                          <td className="p-2 flex items-center gap-1.5">
                            <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: getStrokeColor(rc) }} />
                            <span className="text-gray-300 truncate max-w-[80px]" title={rc}>{rc.replace('Road', '')}</span>
                          </td>
                          <td className="p-2 text-right text-white">{stat.lengthKm} km</td>
                          <td className="p-2 text-right text-gray-400">{stat.percentage}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : (
              <div className="p-4 rounded border border-[#1f242c] bg-gray-900/20 text-center text-gray-500">
                No class distribution data computed.
              </div>
            )}

            {/* Quality distribution breakdown */}
            {activeResult.classificationStats?.qualityBreakdown ? (
              <div className="space-y-2 pt-2 border-t border-gray-800/60">
                <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Quality Distribution</label>
                <div className="space-y-1.5">
                  {Object.entries(activeResult.classificationStats.qualityBreakdown).map(([tier, stat]: any) => {
                    let color = 'bg-emerald-400';
                    if (tier === 'Moderate') color = 'bg-amber-400';
                    if (tier === 'Requires Review') color = 'bg-red-400';

                    return (
                      <div key={tier} className="flex justify-between items-center p-2 rounded border border-[#1f242c] bg-gray-900/30">
                        <div className="flex items-center gap-2">
                          <span className={`h-2 w-2 rounded-full ${color}`} />
                          <span className="text-gray-300">{tier}</span>
                        </div>
                        <span className="text-white font-semibold">{stat.percentage}%</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className="p-4 rounded border border-[#1f242c] bg-gray-900/20 text-center text-gray-500">
                No quality breakdown computed.
              </div>
            )}
          </div>

          <div className="text-[10px] text-gray-500 border-t border-[#1f242c] pt-4">
            Road geometry properties (mean width and variation) are computed directly from skeleton intersections.
          </div>
        </div>
      )}
    </div>
  );
}
