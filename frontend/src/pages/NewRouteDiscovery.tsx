import React, { useState, useEffect, useRef } from 'react';
import { 
  Compass, 
  Upload, 
  Play, 
  Info, 
  Layers, 
  Route, 
  CheckCircle, 
  AlertTriangle,
  FileText,
  Activity,
  ArrowRight,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Download,
  Grid,
  MapPin,
  Eye,
  EyeOff
} from 'lucide-react';
import { apiService } from '../services/api';
import { ExtractionConfig, PotentialRoute } from '../types';

const DEFAULT_CONFIG: ExtractionConfig = {
  threshold: 0.25,
  closingRadius: 6,
  minObjectSize: 32,
  bridgeKernelSize: 21,
  imageSize: 512,
  useModel: true
};

export default function NewRouteDiscovery() {
  // Inputs & Metadata
  const [analysisName, setAnalysisName] = useState('Meridian Route Discovery');
  const [studyArea, setStudyArea] = useState('Meridian Region');
  const [imageYear, setImageYear] = useState('2026');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  // Reference Network Configuration
  const [refSourceType, setRefSourceType] = useState<'file' | 'run'>('file');
  const [refFile, setRefFile] = useState<File | null>(null);
  const [refJobId, setRefJobId] = useState('');
  const [availableRuns, setAvailableRuns] = useState<{ id: string; name: string }[]>([]);

  // Parameters
  const [tolerance, setTolerance] = useState(15.0);
  const [minLength, setMinLength] = useState(30.0);
  const [config, setConfig] = useState<ExtractionConfig>(DEFAULT_CONFIG);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // States
  const [isProcessing, setIsProcessing] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  // Overlay geojson layers
  const [refGeojson, setRefGeojson] = useState<any>(null);
  const [candGeojson, setCandGeojson] = useState<any>(null);
  const [unmappedGeojson, setUnmappedGeojson] = useState<any>(null);
  const [geojsonLoading, setGeojsonLoading] = useState(false);

  // Selected route and route details
  const [selectedRoute, setSelectedRoute] = useState<any>(null);
  const [potentialRoutes, setPotentialRoutes] = useState<any[]>([]);

  // Point-to-Point routing states
  const [startPoint, setStartPoint] = useState<{ x: number; y: number } | null>(null);
  const [endPoint, setEndPoint] = useState<{ x: number; y: number } | null>(null);
  const [avoidanceWeight, setAvoidanceWeight] = useState<number>(10.0);
  const [numRoutes, setNumRoutes] = useState<number>(4);
  const [allowExistingRoads, setAllowExistingRoads] = useState<boolean>(true);
  const [minSeparation, setMinSeparation] = useState<number>(10);
  const [minDiversity, setMinDiversity] = useState<number>(0.3);
  const [visibleRouteIds, setVisibleRouteIds] = useState<Set<number>>(new Set());
  const [placementMode, setPlacementMode] = useState<'start' | 'end' | null>(null);
  const [alternativeRoutes, setAlternativeRoutes] = useState<any[]>([]);
  const [selectedAlternativeRoute, setSelectedAlternativeRoute] = useState<any | null>(null);
  const [isRouting, setIsRouting] = useState<boolean>(false);
  const [infoMessage, setInfoMessage] = useState<string | null>(null);

  // Map viewport states
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [cursorCoords, setCursorCoords] = useState({ x: 0, y: 0, lat: '31.9700° N', lng: '97.2400° W' });

  // Layer visibility
  const [showReference, setShowReference] = useState(true);
  const [showExtracted, setShowExtracted] = useState(true);
  const [showNovel, setShowNovel] = useState(true);
  const [showNodes, setShowNodes] = useState(true);

  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  // Fetch runs list on load to populate recent runs selector
  useEffect(() => {
    apiService.getRuns()
      .then(data => {
        setAvailableRuns(data.map((r: any) => ({ id: r.id, name: r.name })));
      })
      .catch(err => {
        console.error("Failed to load runs history:", err);
      });
  }, []);

  // Synchronize visibleRouteIds with alternativeRoutes
  useEffect(() => {
    if (alternativeRoutes && alternativeRoutes.length > 0) {
      setVisibleRouteIds(new Set(alternativeRoutes.map(r => r.route_id)));
    } else {
      setVisibleRouteIds(new Set());
    }
  }, [alternativeRoutes]);

  const getRouteColor = (routeId: number) => {
    // Visually distinguishable route colors
    const colors = ['#6366f1', '#a855f7', '#ec4899', '#06b6d4', '#f43f5e'];
    return colors[(routeId - 1) % colors.length];
  };

  const getRouteLengthLabel = (lengthMeters: number, routes = alternativeRoutes) => {
    if (routes.length <= 1) return 'Medium';

    const lengths = routes.map(route => Number(route.length_meters ?? route.lengthMeters ?? 0)).sort((a, b) => a - b);
    const shortest = lengths[0];
    const longest = lengths[lengths.length - 1];
    const range = longest - shortest;

    if (range === 0) return 'Medium';
    if (lengthMeters <= shortest + range / 3) return 'Short';
    if (lengthMeters <= shortest + (range * 2) / 3) return 'Medium';
    return 'Long';
  };

  const toggleRouteVisibility = (routeId: number, e: React.MouseEvent) => {
    e.stopPropagation();
    const next = new Set(visibleRouteIds);
    if (next.has(routeId)) {
      next.delete(routeId);
    } else {
      next.add(routeId);
    }
    setVisibleRouteIds(next);
  };

  const loadDiscoveryGeojsons = async (res: any) => {
    setGeojsonLoading(true);
    try {
      const fetchJson = async (url: string) => {
        if (!url) return null;
        const r = await fetch(url);
        if (r.ok) return r.json();
        return null;
      };

      const [ref, cand, unmapped] = await Promise.all([
        fetchJson(res.referenceGeojsonUrl),
        fetchJson(res.candidateGeojsonUrl),
        fetchJson(res.unmappedGeojsonUrl)
      ]);

      setRefGeojson(ref);
      setCandGeojson(cand);
      setUnmappedGeojson(unmapped);

      // Extract LineString features as potential routes
      const novelFeatures = unmapped?.features?.filter((f: any) => f.geometry?.type === 'LineString') || [];
      const derived = novelFeatures.map((f: any) => ({
        id: f.properties?.id ?? f.id ?? 0,
        lengthPixels: f.properties?.length_pixels || 0,
        lengthMeters: f.properties?.length_meters || 0,
        status: f.properties?.status || 'Unmapped Route Segment',
        coordinates: f.geometry.coordinates
      }));

      setPotentialRoutes(derived);
      if (derived.length > 0) {
        setSelectedRoute(derived[0]);
      } else {
        setSelectedRoute(null);
      }
    } catch (e) {
      console.error('Failed to load discovery GeoJSON layers:', e);
    } finally {
      setGeojsonLoading(false);
    }
  };

  const handleStartAnalysis = async () => {
    if (!selectedFile) {
      setErrorMsg('Please select a satellite imagery file to run extraction.');
      return;
    }
    if (refSourceType === 'file' && !refFile) {
      setErrorMsg('Please upload a reference GeoJSON file.');
      return;
    }
    if (refSourceType === 'run' && !refJobId) {
      setErrorMsg('Please select a previous analysis run for reference.');
      return;
    }

    setIsProcessing(true);
    setErrorMsg(null);
    setResult(null);
    setRefGeojson(null);
    setCandGeojson(null);
    setUnmappedGeojson(null);
    setPotentialRoutes([]);
    setSelectedRoute(null);

    try {
      const data = await apiService.runDiscovery(
        selectedFile,
        refFile,
        refJobId,
        tolerance,
        minLength,
        analysisName,
        studyArea,
        imageYear,
        config
      );

      setResult(data);
      await loadDiscoveryGeojsons(data);
    } catch (err: any) {
      console.error('Discovery processing error:', err);
      setErrorMsg(err.message || 'Route discovery execution failed.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handlePointToPointRouting = async () => {
    if (!startPoint || !endPoint) {
      setErrorMsg('Please select both Start and End points on the imagery map.');
      return;
    }

    setIsRouting(true);
    setErrorMsg(null);
    setInfoMessage(null);
    setAlternativeRoutes([]);
    setSelectedAlternativeRoute(null);

    try {
      const activeJobId = result ? result.projectId : '';
      
      if (!activeJobId) {
        if (!selectedFile) {
          setErrorMsg('Please select a satellite imagery file to run extraction first.');
          setIsRouting(false);
          return;
        }
        if (refSourceType === 'file' && !refFile) {
          setErrorMsg('Please upload a reference GeoJSON file.');
          setIsRouting(false);
          return;
        }
        if (refSourceType === 'run' && !refJobId) {
          setErrorMsg('Please select a previous analysis run for reference.');
          setIsRouting(false);
          return;
        }
      }

      const data = await apiService.runPointToPoint(
        activeJobId ? null : selectedFile,
        activeJobId ? null : (refSourceType === 'file' ? refFile : null),
        activeJobId ? '' : refJobId,
        activeJobId,
        startPoint.x,
        startPoint.y,
        endPoint.x,
        endPoint.y,
        tolerance,
        avoidanceWeight,
        allowExistingRoads,
        numRoutes,
        analysisName,
        studyArea,
        imageYear,
        config,
        minSeparation,
        minDiversity
      );

      setAlternativeRoutes(data.routes);
      setInfoMessage(data.info_message || null);
      if (data.routes && data.routes.length > 0) {
        setSelectedAlternativeRoute(data.routes[0]);
      }
      if (data.start_point?.snapped) {
        setStartPoint(data.start_point.snapped);
      }
      if (data.end_point?.snapped) {
        setEndPoint(data.end_point.snapped);
      }
      
      if (!result && data.projectId) {
        setResult(data);
        await loadDiscoveryGeojsons(data);
      }
    } catch (err: any) {
      console.error('Point-to-point routing error:', err);
      setErrorMsg(err.message || 'Routing calculation failed.');
    } finally {
      setIsRouting(false);
    }
  };

  const downloadSelectedRoute = () => {
    if (!selectedAlternativeRoute) return;
    const geojson = {
      type: "FeatureCollection",
      features: [{
        type: "Feature",
        geometry: {
          type: "LineString",
          coordinates: selectedAlternativeRoute.coordinates
        },
        properties: {
          route_id: selectedAlternativeRoute.route_id,
          length_meters: selectedAlternativeRoute.length_meters,
          novel_length_meters: selectedAlternativeRoute.novel_length_meters,
          overlap_length_meters: selectedAlternativeRoute.overlap_length_meters,
          overlap_percentage: selectedAlternativeRoute.overlap_percentage
        }
      }]
    };
    const blob = new Blob([JSON.stringify(geojson, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `p2p_route_${selectedAlternativeRoute.route_id}.geojson`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Drag and drop handlers for imagery
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDropImage = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  };

  const handleDropRef = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setRefFile(e.dataTransfer.files[0]);
    }
  };

  // Map panning/zooming controls
  const handleMouseDown = (e: React.MouseEvent) => {
    if ((e.target as HTMLElement).closest('button') || (e.target as HTMLElement).closest('select') || (e.target as HTMLElement).closest('input')) return;
    
    if (placementMode && svgRef.current) {
      e.stopPropagation();
      const svg = svgRef.current;
      const point = svg.createSVGPoint();
      point.x = e.clientX;
      point.y = e.clientY;
      const svgPoint = point.matrixTransform(svg.getScreenCTM()?.inverse());
      if (svgPoint) {
        const x = Math.round(svgPoint.x);
        const y = Math.round(svgPoint.y);
        
        if (x < 0 || x > 1024 || y < 0 || y > 1024) {
          setErrorMsg('Invalid point selection. Click coordinate must be within the satellite imagery bounds.');
          setPlacementMode(null);
          return;
        }
        
        if (placementMode === 'start') {
          setStartPoint({ x, y });
        } else if (placementMode === 'end') {
          setEndPoint({ x, y });
        }
      }
      setPlacementMode(null);
      return;
    }
    
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

  // Extract geometries from layers
  const refLines = refGeojson?.features?.filter((f: any) => f.geometry?.type === 'LineString') || [];
  const refNodes = refGeojson?.features?.filter((f: any) => f.geometry?.type === 'Point') || [];

  const candLines = candGeojson?.features?.filter((f: any) => f.geometry?.type === 'LineString') || [];
  const candNodes = candGeojson?.features?.filter((f: any) => f.geometry?.type === 'Point') || [];

  const novelLines = unmappedGeojson?.features?.filter((f: any) => f.geometry?.type === 'LineString') || [];
  const novelNodes = unmappedGeojson?.features?.filter((f: any) => f.geometry?.type === 'Point') || [];

  return (
    <div className="h-full flex overflow-hidden bg-[#070a0e]">
      {/* Left Sidebar: Controls & List */}
      <div className="w-84 border-r border-[#1f242c] bg-[#0b0f14]/40 flex flex-col justify-between overflow-y-auto shrink-0 p-5 space-y-6">
        <div className="space-y-5">
          {/* Header */}
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <Compass className="h-4 w-4 text-emerald-400" />
              <h2 className="text-xs font-semibold text-white tracking-wider font-mono uppercase">
                New Route Discovery
              </h2>
            </div>
            <p className="text-[11px] text-gray-400 leading-relaxed">
              Find unmapped road routes missing from baseline reference records.
            </p>
          </div>



          {/* Form */}
          <div className="space-y-3.5 text-xs font-mono">
            <div className="space-y-1">
              <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Analysis Name</label>
              <input
                type="text"
                value={analysisName}
                onChange={(e) => setAnalysisName(e.target.value)}
                disabled={isProcessing}
                className="w-full bg-gray-900 border border-[#1f242c] rounded px-3 py-2 text-white outline-none focus:border-emerald-500/50"
              />
            </div>

            <div className="space-y-1">
              <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Study Area</label>
              <input
                type="text"
                value={studyArea}
                onChange={(e) => setStudyArea(e.target.value)}
                disabled={isProcessing}
                className="w-full bg-gray-900 border border-[#1f242c] rounded px-3 py-2 text-white outline-none focus:border-emerald-500/50"
              />
            </div>

            <div className="space-y-1">
              <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Imagery Year</label>
              <input
                type="text"
                value={imageYear}
                onChange={(e) => setImageYear(e.target.value)}
                disabled={isProcessing}
                className="w-full bg-gray-900 border border-[#1f242c] rounded px-3 py-2 text-white outline-none focus:border-emerald-500/50"
              />
            </div>

            {/* Reference Source Type Select */}
            <div className="space-y-2.5 border-t border-gray-800/60 pt-3">
              <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Reference Source Type</label>
              <div className="flex gap-4">
                <label className="flex items-center gap-1.5 cursor-pointer text-gray-300">
                  <input
                    type="radio"
                    name="refType"
                    checked={refSourceType === 'file'}
                    onChange={() => setRefSourceType('file')}
                    disabled={isProcessing}
                    className="accent-emerald-500"
                  />
                  <span>Upload File</span>
                </label>
                <label className="flex items-center gap-1.5 cursor-pointer text-gray-300">
                  <input
                    type="radio"
                    name="refType"
                    checked={refSourceType === 'run'}
                    onChange={() => setRefSourceType('run')}
                    disabled={isProcessing}
                    className="accent-emerald-500"
                  />
                  <span>Saved Run</span>
                </label>
              </div>
            </div>

            {refSourceType === 'file' ? (
              <div className="space-y-1">
                <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Reference GeoJSON</label>
                <label 
                  onDragOver={handleDragOver}
                  onDrop={handleDropRef}
                  className="border border-dashed border-[#1f242c] hover:border-emerald-500/40 rounded-lg p-3 flex flex-col items-center justify-center cursor-pointer bg-gray-900/30 text-center"
                >
                  <Upload className="h-4 w-4 text-emerald-400 mb-1" />
                  <span className="text-[10px] text-gray-300">
                    {refFile ? refFile.name : 'Upload GeoJSON reference'}
                  </span>
                  <input
                    type="file"
                    accept=".geojson,.json"
                    onChange={(e) => {
                      if (e.target.files && e.target.files[0]) setRefFile(e.target.files[0]);
                    }}
                    disabled={isProcessing}
                    className="hidden"
                  />
                </label>
              </div>
            ) : (
              <div className="space-y-1">
                <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Select Run Reference</label>
                <select
                  value={refJobId}
                  onChange={(e) => setRefJobId(e.target.value)}
                  disabled={isProcessing}
                  className="w-full bg-gray-900 border border-[#1f242c] rounded px-3 py-2 text-white outline-none focus:border-emerald-500/50 text-xs"
                >
                  <option value="">-- Choose Completed Run --</option>
                  {availableRuns.map(run => (
                    <option key={run.id} value={run.id}>{run.name}</option>
                  ))}
                </select>
              </div>
            )}

            {/* Satellite Imagery file upload */}
            <div className="space-y-1">
              <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Extraction Imagery</label>
              <label 
                onDragOver={handleDragOver}
                onDrop={handleDropImage}
                className="border border-dashed border-[#1f242c] hover:border-emerald-500/40 rounded-lg p-3 flex flex-col items-center justify-center cursor-pointer bg-gray-900/30 text-center"
              >
                <Upload className="h-4 w-4 text-emerald-400 mb-1" />
                <span className="text-[10px] text-gray-300">
                  {selectedFile ? selectedFile.name : 'Select or drop imagery'}
                </span>
                <input
                  type="file"
                  accept="image/png,image/jpeg,image/tiff"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) setSelectedFile(e.target.files[0]);
                  }}
                  disabled={isProcessing}
                  className="hidden"
                />
              </label>
            </div>

            {/* Config parameters */}
            <div className="space-y-3 border-t border-gray-800/60 pt-3">
              <div className="flex justify-between items-center">
                <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">5. Config Parameters</label>
                <button 
                  onClick={() => setShowAdvanced(!showAdvanced)}
                  className="text-[10px] text-emerald-400 hover:text-emerald-300 underline font-mono cursor-pointer bg-transparent border-none outline-none"
                >
                  {showAdvanced ? 'Collapse' : 'Advanced'}
                </button>
              </div>

              {showAdvanced && (
                <div className="space-y-3">
                  <div className="space-y-1">
                    <div className="flex justify-between text-gray-400">
                      <span>Spatial Tolerance (px):</span>
                      <span className="text-white">{tolerance}</span>
                    </div>
                    <input 
                      type="range" 
                      min="5" 
                      max="40" 
                      value={tolerance}
                      onChange={(e) => setTolerance(parseFloat(e.target.value))}
                      className="w-full h-1 bg-gray-800 rounded appearance-none cursor-pointer accent-emerald-500"
                    />
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between text-gray-400">
                      <span>Min Segment Len (px):</span>
                      <span className="text-white">{minLength}</span>
                    </div>
                    <input 
                      type="range" 
                      min="10" 
                      max="100" 
                      value={minLength}
                      onChange={(e) => setMinLength(parseFloat(e.target.value))}
                      className="w-full h-1 bg-gray-800 rounded appearance-none cursor-pointer accent-emerald-500"
                    />
                  </div>

                  <div className="space-y-1">
                    <div className="flex justify-between text-gray-400">
                      <span>Confidence Thresh:</span>
                      <span className="text-emerald-400">{config.threshold}</span>
                    </div>
                    <input 
                      type="range" 
                      min="0.10" 
                      max="0.80" 
                      step="0.05"
                      value={config.threshold}
                      onChange={(e) => setConfig({ ...config, threshold: parseFloat(e.target.value) })}
                      className="w-full h-1 bg-gray-800 rounded appearance-none cursor-pointer accent-emerald-500"
                    />
                  </div>

                  <div className="flex items-center justify-between border-t border-gray-800/40 pt-2 text-gray-400">
                    <span>Use U-Net Model:</span>
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
        </div>

        {/* Trigger for Section 1 (Unmapped Route Discovery) */}
            <button
              onClick={handleStartAnalysis}
              disabled={isProcessing}
              className={`w-full py-2.5 px-4 rounded text-xs font-mono font-bold uppercase tracking-wider transition-all flex items-center justify-center gap-2 cursor-pointer ${
                isProcessing
                  ? 'bg-gray-800 text-gray-500 cursor-not-allowed border border-transparent'
                  : 'bg-emerald-500 text-[#070a0e] hover:bg-emerald-400 shadow-md shadow-emerald-500/10'
              }`}
            >
              {isProcessing ? (
                <>
                  <div className="h-3 w-3 border-2 border-gray-500 border-t-emerald-400 rounded-full animate-spin" />
                  <span>Running Discovery...</span>
                </>
              ) : (
                <>
                  <Play className="h-3.5 w-3.5 fill-current" />
                  <span>Start Route Discovery</span>
                </>
              )}
            </button>

            {/* Discovered routes list for Section 1 (Unmapped Route Discovery) */}
            {result && potentialRoutes.length > 0 && (
              <div className="space-y-2 pt-2 border-t border-gray-800/60">
                <div className="flex justify-between items-center text-[10px] font-mono uppercase text-gray-500">
                  <span>Discovered Segments ({potentialRoutes.length})</span>
                  <span className="text-amber-400 font-semibold font-bold">Unmapped</span>
                </div>
                <div className="max-h-48 overflow-y-auto space-y-1.5 pr-1 text-xs font-mono">
                  {potentialRoutes.map((route) => {
                    const isSelected = selectedRoute?.id === route.id;
                    return (
                      <div
                        key={route.id}
                        onClick={() => setSelectedRoute(route)}
                        className={`p-2 rounded border cursor-pointer transition-colors flex justify-between items-center ${
                          isSelected 
                            ? 'border-amber-500/50 bg-amber-500/10 text-white' 
                            : 'border-[#1f242c] bg-gray-900/30 text-gray-400 hover:text-white hover:bg-gray-800/40'
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <Route className={`h-3.5 w-3.5 ${isSelected ? 'text-amber-400' : 'text-gray-500'}`} />
                          <span>Route Segment #{route.id}</span>
                        </div>
                        <span className="text-[10px] text-gray-500">{getRouteLengthLabel(route.lengthMeters, potentialRoutes)}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Error Message */}
            {errorMsg && (
              <div className="p-3 border border-red-500/30 bg-red-950/20 rounded text-[11px] font-mono text-red-400 flex items-start gap-2">
                <AlertTriangle className="h-4 w-4 shrink-0 text-red-400 mt-0.5" />
                <span>{errorMsg}</span>
              </div>
            )}

            {/* Visual Divider Separator */}
            <div className="border-t border-[#1f242c] my-4 pt-3" />

            {/* Section 2: POINT-TO-POINT ROUTE DISCOVERY */}
            <div className="space-y-4">
              <div className="space-y-1">
                <h3 className="text-xs font-semibold text-white tracking-wider font-mono uppercase">
                  POINT-TO-POINT ROUTE DISCOVERY
                </h3>
                <p className="text-[10px] text-gray-400 leading-normal font-mono">
                  Select two points on the imagery to discover multiple feasible new route alternatives between them.
                </p>
              </div>

              {/* Interactive Point Placement Pins */}
              <div className="grid grid-cols-2 gap-2 text-[9px] font-mono">
                <button
                  onClick={() => setPlacementMode(placementMode === 'start' ? null : 'start')}
                  className={`p-2 rounded border flex flex-col items-center justify-center gap-1.5 transition-all text-center cursor-pointer ${
                    placementMode === 'start'
                      ? 'border-emerald-500 bg-emerald-500/20 text-white animate-pulse font-bold'
                      : startPoint
                      ? 'border-emerald-500/30 bg-emerald-500/5 text-emerald-400 hover:bg-emerald-500/10 font-bold'
                      : 'border-[#1f242c] bg-gray-900/30 text-gray-400 hover:text-white hover:bg-gray-800/40'
                  }`}
                >
                  <MapPin className="h-4 w-4 text-emerald-400 animate-bounce" />
                  <span>START POINT</span>
                  <span className="text-[8px] text-gray-500">
                    {startPoint ? `x:${startPoint.x}, y:${startPoint.y}` : 'Click image to place'}
                  </span>
                </button>

                <button
                  onClick={() => setPlacementMode(placementMode === 'end' ? null : 'end')}
                  className={`p-2 rounded border flex flex-col items-center justify-center gap-1.5 transition-all text-center cursor-pointer ${
                    placementMode === 'end'
                      ? 'border-red-500 bg-red-500/20 text-white animate-pulse font-bold'
                      : endPoint
                      ? 'border-red-500/30 bg-red-500/5 text-red-400 hover:bg-red-500/10 font-bold'
                      : 'border-[#1f242c] bg-gray-900/30 text-gray-400 hover:text-white hover:bg-gray-800/40'
                  }`}
                >
                  <MapPin className="h-4 w-4 text-red-400 animate-bounce" />
                  <span>DESTINATION POINT</span>
                  <span className="text-[8px] text-gray-500">
                    {endPoint ? `x:${endPoint.x}, y:${endPoint.y}` : 'Click image to place'}
                  </span>
                </button>
              </div>

              {/* Advanced Options Accordion */}
              <div className="space-y-3 pt-1 text-[10px] font-mono text-gray-400">
                <div className="space-y-1">
                  <div className="flex justify-between">
                    <span>Number of routes:</span>
                    <span className="text-white font-bold">{numRoutes}</span>
                  </div>
                  <input
                    type="range"
                    min="2"
                    max="5"
                    step="1"
                    value={numRoutes}
                    onChange={(e) => setNumRoutes(parseInt(e.target.value))}
                    className="w-full h-1 bg-gray-800 rounded appearance-none cursor-pointer accent-emerald-500"
                  />
                </div>

                <div className="space-y-1">
                  <div className="flex justify-between">
                    <span>Existing-road penalty:</span>
                    <span className="text-white font-bold">{avoidanceWeight}x</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="50"
                    step="1"
                    value={avoidanceWeight}
                    onChange={(e) => setAvoidanceWeight(parseFloat(e.target.value))}
                    className="w-full h-1 bg-gray-800 rounded appearance-none cursor-pointer accent-emerald-500"
                  />
                </div>

                <div className="space-y-1">
                  <div className="flex justify-between">
                    <span>Min route separation:</span>
                    <span className="text-white font-bold">{minSeparation} px</span>
                  </div>
                  <input
                    type="range"
                    min="5"
                    max="50"
                    step="5"
                    value={minSeparation}
                    onChange={(e) => setMinSeparation(parseInt(e.target.value))}
                    className="w-full h-1 bg-gray-800 rounded appearance-none cursor-pointer accent-emerald-500"
                  />
                </div>

                <div className="space-y-1">
                  <div className="flex justify-between">
                    <span>Min route diversity:</span>
                    <span className="text-white font-bold">{(minDiversity * 100).toFixed(0)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={minDiversity}
                    onChange={(e) => setMinDiversity(parseFloat(e.target.value))}
                    className="w-full h-1 bg-gray-800 rounded appearance-none cursor-pointer accent-emerald-500"
                  />
                </div>

                <div className="flex items-center justify-between pt-1">
                  <span>Allow existing roads:</span>
                  <input
                    type="checkbox"
                    checked={allowExistingRoads}
                    onChange={(e) => setAllowExistingRoads(e.target.checked)}
                    className="h-3.5 w-3.5 cursor-pointer accent-emerald-500"
                  />
                </div>
              </div>

              {/* P2P Route Action Controls */}
              <div className="flex gap-2 font-mono">
                <button
                  onClick={() => {
                    setStartPoint(null);
                    setEndPoint(null);
                    setAlternativeRoutes([]);
                    setSelectedAlternativeRoute(null);
                  }}
                  disabled={!startPoint && !endPoint}
                  className={`flex-1 py-2 rounded text-[10px] font-bold uppercase tracking-wider transition-all cursor-pointer ${
                    !startPoint && !endPoint
                      ? 'bg-gray-800/40 text-gray-600 cursor-not-allowed border border-transparent'
                      : 'border border-gray-700 bg-transparent text-gray-300 hover:bg-gray-850 hover:text-white'
                  }`}
                >
                  Clear Points
                </button>
                
                <button
                  onClick={handlePointToPointRouting}
                  disabled={!result || !startPoint || !endPoint || isRouting}
                  className={`flex-[2] py-2 rounded text-[10px] font-bold uppercase tracking-wider transition-all flex items-center justify-center gap-1.5 cursor-pointer ${
                    !result || !startPoint || !endPoint || isRouting
                      ? 'bg-gray-800 text-gray-500 cursor-not-allowed border border-transparent'
                      : 'bg-emerald-500 text-[#070a0e] hover:bg-emerald-400 shadow-md shadow-emerald-500/10'
                  }`}
                >
                  {isRouting ? (
                    <>
                      <div className="h-3 w-3 border-2 border-gray-500 border-t-emerald-400 rounded-full animate-spin" />
                      <span>Finding...</span>
                    </>
                  ) : (
                    <span>Find Routes</span>
                  )}
                </button>
              </div>

              {/* Discovered Alternative Options List */}
              {alternativeRoutes.length > 0 && (
                <div className="space-y-2 pt-2 border-t border-gray-800/60 font-mono">
                  <div className="flex justify-between items-center text-[10px] uppercase text-gray-500">
                    <span>DISCOVERED ROUTES ({alternativeRoutes.length})</span>
                    <span className="text-emerald-400 font-semibold font-bold">Paths</span>
                  </div>
                  <div className="max-h-56 overflow-y-auto space-y-1.5 pr-1 text-xs">
                    {alternativeRoutes.map((route) => {
                      const isSelected = selectedAlternativeRoute?.route_id === route.route_id;
                      const isVisible = visibleRouteIds.has(route.route_id);
                      const routeColor = getRouteColor(route.route_id);
                      return (
                        <div
                          key={`side-route-${route.route_id}`}
                          onClick={() => setSelectedAlternativeRoute(route)}
                          className={`p-2 rounded border cursor-pointer transition-all flex flex-col gap-1 ${
                            isSelected 
                              ? 'border-emerald-500 bg-emerald-500/10 text-white font-bold' 
                              : 'border-[#1f242c] bg-gray-900/30 text-gray-400 hover:text-white hover:bg-gray-800/40'
                          }`}
                        >
                          <div className="flex justify-between items-center w-full">
                            <div className="flex items-center gap-1.5">
                              <div 
                                className="h-2 w-2 rounded-full shrink-0" 
                                style={{ backgroundColor: routeColor }} 
                              />
                              <span>Route #{route.route_id}</span>
                            </div>
                            <div className="flex items-center gap-2 pointer-events-auto">
                              <button
                                onClick={(e) => toggleRouteVisibility(route.route_id, e)}
                                className="p-0.5 hover:text-white text-gray-500 transition-colors bg-transparent border-none outline-none cursor-pointer"
                                title={isVisible ? "Hide route" : "Show route"}
                              >
                                {isVisible ? <Eye className="h-3.5 w-3.5" /> : <EyeOff className="h-3.5 w-3.5 text-gray-600" />}
                              </button>
                              <span className="text-[10px] text-gray-500">{getRouteLengthLabel(route.length_meters)}</span>
                            </div>
                          </div>
                          <div className="grid grid-cols-2 gap-x-2 text-[9px] text-gray-500 mt-1 border-t border-gray-800/40 pt-1 font-mono">
                            <div>Route scale: <span className="text-gray-300 font-bold">{getRouteLengthLabel(route.length_meters)}</span></div>
                            <div>Overlap: <span className="text-gray-300 font-bold">{route.overlap_percentage.toFixed(1)}%</span></div>
                            <div className="col-span-2 mt-0.5">
                              Score: <span className="text-amber-400 font-bold font-mono">{(route.score * 100).toFixed(0)}/100</span>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </div>

        <div className="p-4 border-t border-[#1f242c] bg-gray-900/10 text-[10px] text-gray-500 rounded flex gap-2">
          <Info className="h-4 w-4 shrink-0 text-emerald-400" />
          <span>Calculates candidate road lines excluding baseline buffered network.</span>
        </div>
      </div>

      {/* Center Viewport: Interactive Canvas Map */}
      <div 
        className="flex-1 flex flex-col p-6 space-y-6 relative overflow-hidden min-w-0"
        ref={containerRef}
      >
        <div className="flex-1 flex flex-col bg-[#0d121a] border border-[#1f242c] rounded overflow-hidden relative select-none">
          {isProcessing ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center p-8 space-y-4 gis-grid">
              <div className="h-10 w-10 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
              <div className="space-y-1">
                <h4 className="text-xs font-semibold text-white tracking-widest font-mono uppercase">DISCOVERY IN PROGRESS</h4>
                <p className="text-[11px] text-gray-400 max-w-sm leading-relaxed">
                  Running road extraction, parsing baseline vectors, and performing spatial difference operations.
                </p>
              </div>
            </div>
          ) : !result ? (
            <div className="flex-1 flex flex-col items-center justify-center text-center p-8 space-y-4 gis-grid">
              <div className="text-gray-600 bg-gray-900/60 p-4 rounded-full border border-[#1f242c] relative">
                <Compass className="h-10 w-10 text-gray-500" />
              </div>
              <div className="space-y-1">
                <h4 className="text-xs font-semibold text-white tracking-widest font-mono uppercase">NO DISCOVERY ANALYSIS RUN</h4>
                <p className="text-[11px] text-gray-400 max-w-sm leading-relaxed">
                  Upload extraction imagery and configure a reference vector source in the left panel to begin.
                </p>
              </div>
            </div>
          ) : (
            <>
              {/* Toolbar */}
              <div className="absolute top-4 left-4 right-4 flex justify-between items-center pointer-events-none z-10">
                {/* Toggles */}
                <div className="flex gap-1 bg-[#0b0f14]/85 border border-[#1f242c] p-1 rounded-md backdrop-blur-sm pointer-events-auto">
                  <button
                    onClick={() => setShowReference(!showReference)}
                    className={`px-2 py-1.5 text-[9px] font-mono font-semibold tracking-wider rounded uppercase transition-colors cursor-pointer flex items-center gap-1.5 ${
                      showReference 
                        ? 'bg-blue-500/10 text-blue-400 border border-blue-500/25' 
                        : 'text-gray-500 hover:text-white border border-transparent'
                    }`}
                  >
                    <div className="h-1.5 w-1.5 rounded-full bg-blue-400" />
                    Reference
                  </button>
                  <button
                    onClick={() => setShowExtracted(!showExtracted)}
                    className={`px-2 py-1.5 text-[9px] font-mono font-semibold tracking-wider rounded uppercase transition-colors cursor-pointer flex items-center gap-1.5 ${
                      showExtracted 
                        ? 'bg-orange-500/10 text-orange-400 border border-orange-500/25' 
                        : 'text-gray-500 hover:text-white border border-transparent'
                    }`}
                  >
                    <div className="h-1.5 w-1.5 rounded-full bg-orange-400" />
                    Extracted
                  </button>
                  <button
                    onClick={() => setShowNovel(!showNovel)}
                    className={`px-2 py-1.5 text-[9px] font-mono font-semibold tracking-wider rounded uppercase transition-colors cursor-pointer flex items-center gap-1.5 ${
                      showNovel 
                        ? 'bg-amber-500/10 text-amber-400 border border-amber-500/25' 
                        : 'text-gray-500 hover:text-white border border-transparent'
                    }`}
                  >
                    <div className="h-1.5 w-1.5 rounded-full bg-amber-400" />
                    Novel Routes
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

                {/* Zoom */}
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
                    Overlaying discovery geometries...
                  </div>
                )}

                <img 
                  src={result.imageBUrl}
                  alt={`Discovery Imagery`}
                  className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                  style={{
                    transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                    transformOrigin: 'center center'
                  }}
                />

                <svg 
                  ref={svgRef}
                  className="absolute inset-0 w-full h-full pointer-events-none"
                  viewBox="0 0 1024 1024"
                  style={{
                    transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                    transformOrigin: 'center center'
                  }}
                >
                  {/* 1. Reference road lines (Thin dashed gray/blue) */}
                  {showReference && refLines.length > 0 && (
                    <g fill="none" stroke="#3b82f6" strokeWidth="1.5" strokeDasharray="3,3" strokeLinecap="round" opacity="0.65">
                      {refLines.map((line: any, idx: number) => (
                        <path key={`ref-${idx}`} d={getLinePath(line.geometry.coordinates)} />
                      ))}
                    </g>
                  )}

                  {/* 2. Extracted network lines (Thin solid orange) */}
                  {showExtracted && candLines.length > 0 && (
                    <g fill="none" stroke="#f97316" strokeWidth="1.5" strokeLinecap="round" opacity="0.55">
                      {candLines.map((line: any, idx: number) => (
                        <path key={`cand-${idx}`} d={getLinePath(line.geometry.coordinates)} />
                      ))}
                    </g>
                  )}

                  {/* 3. Novel/unmapped routes lines (Thick solid amber) */}
                  {showNovel && novelLines.length > 0 && (
                    <g fill="none" stroke="#f59e0b" strokeWidth="4" strokeLinecap="round" opacity="0.9">
                      {novelLines.map((line: any, idx: number) => (
                        <path key={`nov-${idx}`} d={getLinePath(line.geometry.coordinates)} />
                      ))}
                    </g>
                  )}

                  {/* Selected highlighted segment path */}
                  {selectedRoute && (
                    <g fill="none" stroke="#10b981" strokeWidth="6" strokeLinecap="round" opacity="0.95">
                      <path d={getLinePath(selectedRoute.coordinates)} />
                    </g>
                  )}

                  {/* Point-to-Point Alternative Paths */}
                  {alternativeRoutes.map((route) => {
                    const isVisible = visibleRouteIds.has(route.route_id);
                    if (!isVisible) return null;
                    
                    const isSelected = selectedAlternativeRoute?.route_id === route.route_id;
                    const routeColor = getRouteColor(route.route_id);
                    return (
                      <g key={`route-path-${route.route_id}`}>
                        {/* Glow highlight under the selected path */}
                        {isSelected && (
                          <path
                            d={getLinePath(route.coordinates)}
                            fill="none"
                            stroke="#ffffff"
                            strokeWidth="8"
                            strokeLinecap="round"
                            opacity="0.45"
                          />
                        )}
                        <path
                          d={getLinePath(route.coordinates)}
                          fill="none"
                          stroke={routeColor}
                          strokeWidth={isSelected ? '4.5' : '3.0'}
                          strokeLinecap="round"
                          opacity={isSelected ? '1.0' : '0.6'}
                          pointerEvents="auto"
                          style={{ cursor: 'pointer' }}
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedAlternativeRoute(route);
                          }}
                        />
                        <path
                          d={getLinePath(route.coordinates)}
                          fill="none"
                          stroke="transparent"
                          strokeWidth="14"
                          pointerEvents="auto"
                          style={{ cursor: 'pointer' }}
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelectedAlternativeRoute(route);
                          }}
                        />
                      </g>
                    );
                  })}

                  {/* Nodes / Intersections */}
                  {showNodes && (
                    <g fill="#22d3ee" stroke="#0b0f14" strokeWidth="0.8" opacity="0.85">
                      {showReference && refNodes.map((node: any, idx: number) => (
                        <circle key={`rn-${idx}`} cx={node.geometry.coordinates[0]} cy={node.geometry.coordinates[1]} r="3" fill="#3b82f6" />
                      ))}
                      {showExtracted && candNodes.map((node: any, idx: number) => (
                        <circle key={`cn-${idx}`} cx={node.geometry.coordinates[0]} cy={node.geometry.coordinates[1]} r="3" fill="#f97316" />
                      ))}
                      {showNovel && novelNodes.map((node: any, idx: number) => (
                        <circle key={`nn-${idx}`} cx={node.geometry.coordinates[0]} cy={node.geometry.coordinates[1]} r="4" fill="#f59e0b" />
                      ))}
                    </g>
                  )}

                  {/* Start Point Marker */}
                  {startPoint && (
                    <g>
                      <circle 
                        cx={startPoint.x} 
                        cy={startPoint.y} 
                        r="8" 
                        fill="#10b981" 
                        stroke="#070a0e" 
                        strokeWidth="2" 
                        opacity="0.9"
                      />
                      <circle cx={startPoint.x} cy={startPoint.y} r="2.5" fill="#ffffff" />
                      <text 
                        x={startPoint.x} 
                        y={startPoint.y - 12} 
                        fill="#10b981" 
                        fontSize="9" 
                        fontWeight="bold" 
                        fontFamily="monospace" 
                        textAnchor="middle"
                        style={{ userSelect: 'none' }}
                      >
                        POINT A
                      </text>
                    </g>
                  )}

                  {/* End Point Marker */}
                  {endPoint && (
                    <g>
                      <circle 
                        cx={endPoint.x} 
                        cy={endPoint.y} 
                        r="8" 
                        fill="#ef4444" 
                        stroke="#070a0e" 
                        strokeWidth="2" 
                        opacity="0.9"
                      />
                      <circle cx={endPoint.x} cy={endPoint.y} r="2.5" fill="#ffffff" />
                      <text 
                        x={endPoint.x} 
                        y={endPoint.y - 12} 
                        fill="#ef4444" 
                        fontSize="9" 
                        fontWeight="bold" 
                        fontFamily="monospace" 
                        textAnchor="middle"
                        style={{ userSelect: 'none' }}
                      >
                        POINT B
                      </text>
                    </g>
                  )}
                </svg>
              </div>

              {/* Statusbar */}
              <div className="h-8 border-t border-[#1f242c] bg-[#0b0f14] flex justify-between items-center px-4 text-[10px] font-mono text-gray-500 shrink-0">
                <div>CURSOR: {cursorCoords.lat}, {cursorCoords.lng} (x: {cursorCoords.x}px, y: {cursorCoords.y}px)</div>
                <div className="flex gap-4">
                  <span>DISCOVERY ID: {result.projectId}</span>
                  <span>·</span>
                  <span>ZOOM: {Math.round(zoom * 100)}%</span>
                </div>
              </div>
            </>
          )}
        </div>
      </div>

      {/* Right Sidebar: Details Panel */}
      {result && (
        <div className="w-80 border-l border-[#1f242c] bg-[#0b0f14]/40 flex flex-col justify-between overflow-y-auto shrink-0 p-5 space-y-6 text-xs font-mono">
          <div className="space-y-6 flex-1">
            
            {/* Section 1: Unmapped Route Telemetry */}
            <div className="space-y-4">
              <div className="space-y-1">
                <h3 className="text-xs font-semibold text-white tracking-wider uppercase">Route Telemetry</h3>
                <p className="text-[10px] text-gray-400">Detailed stats for discovered unmapped roads.</p>
              </div>

              {/* Overall summary metrics card */}
              <div className="border border-[#1f242c] bg-gray-900/30 rounded p-4 space-y-3">
                <div className="flex justify-between items-center text-gray-400">
                  <span>Novel Segments:</span>
                  <span className="text-amber-400 font-bold">{result.candidateSegments} paths</span>
                </div>
              </div>

              {/* Selected Route segment details */}
              {selectedRoute ? (
                <div className="p-3 rounded border border-amber-500/30 bg-amber-500/10 space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-[10px] uppercase text-amber-400 font-bold">Candidate Segment</span>
                    <span className="text-[10px] text-gray-400 font-bold">#{selectedRoute.id}</span>
                  </div>

                  <div className="space-y-1.5 text-gray-300">
                    <div className="flex justify-between">
                      <span>Route scale:</span>
                      <span className="text-white font-bold">{getRouteLengthLabel(selectedRoute.lengthMeters, potentialRoutes)}</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="p-3 border border-[#1f242c] rounded bg-gray-900/10 text-center text-gray-500 text-[10px]">
                  No unmapped segment selected. Click a segment on the map or list.
                </div>
              )}

              {/* Download exported vector file */}
              <a
                href={result.unmappedGeojsonUrl}
                download="unmapped_routes.geojson"
                target="_blank"
                rel="noopener noreferrer"
                className="w-full py-2 px-4 rounded text-[10px] font-bold uppercase tracking-wider transition-all flex items-center justify-center gap-2 bg-[#1f242c] hover:bg-gray-800 text-emerald-400 hover:text-emerald-300 shadow cursor-pointer"
              >
                <Download className="h-3.5 w-3.5" />
                <span>Export Unmapped GeoJSON</span>
              </a>
            </div>

            {/* Divider */}
            <div className="border-t border-[#1f242c] my-4" />

            {/* Section 2: P2P Route Telemetry */}
            <div className="space-y-4">
              <div className="space-y-1">
                <h3 className="text-xs font-semibold text-white tracking-wider uppercase">P2P Route Telemetry</h3>
                <p className="text-[10px] text-gray-400">Detailed stats for alternative routing paths.</p>
              </div>

              {/* Overall summary metrics card */}
              <div className="border border-[#1f242c] bg-gray-900/30 rounded p-4 space-y-3">
                <div className="flex justify-between items-center text-gray-400">
                  <span>Alternatives Found:</span>
                  <span className="text-emerald-400 font-bold">{alternativeRoutes.length} options</span>
                </div>
                <div className="flex justify-between items-center text-gray-400">
                  <span>Start Location:</span>
                  <span className="text-white font-bold">{startPoint ? `[${startPoint.x}, ${startPoint.y}]` : 'Not set'}</span>
                </div>
                <div className="flex justify-between items-center text-gray-400">
                  <span>End Location:</span>
                  <span className="text-white font-bold">{endPoint ? `[${endPoint.x}, ${endPoint.y}]` : 'Not set'}</span>
                </div>
              </div>

              {infoMessage && (
                <div className="p-3 border border-blue-500/20 bg-blue-500/5 text-blue-400 rounded text-[10px] leading-relaxed flex gap-2">
                  <Info className="h-4 w-4 shrink-0 text-blue-400 mt-0.5" />
                  <span>{infoMessage}</span>
                </div>
              )}

              {/* Selected Alternative Route details */}
              {selectedAlternativeRoute ? (
                <>
                  <div className="p-3 rounded border border-emerald-500/30 bg-emerald-500/10 space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-[10px] uppercase text-emerald-400 font-bold">Selected Alternative</span>
                      <span className="text-[10px] text-gray-400 font-bold">#{selectedAlternativeRoute.route_id}</span>
                    </div>

                    <div className="space-y-1.5 text-gray-300">
                      <div className="flex justify-between">
                        <span>Route scale:</span>
                        <span className="text-white font-bold">{getRouteLengthLabel(selectedAlternativeRoute.length_meters)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Avoidance Efficiency:</span>
                        <span className="text-emerald-400 font-bold">
                          {(100 - selectedAlternativeRoute.overlap_percentage).toFixed(1)}%
                        </span>
                      </div>
                      <div className="flex justify-between border-t border-emerald-500/20 pt-1.5 mt-1.5">
                        <span>Suitability Score:</span>
                        <span className="text-amber-400 font-bold font-mono">
                          {selectedAlternativeRoute.score !== undefined ? (selectedAlternativeRoute.score * 100).toFixed(1) : 'N/A'}/100
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Phase 10 Disclaimer */}
                  <div className="p-3 border border-amber-500/20 bg-amber-500/5 text-amber-500/90 rounded text-[9px] leading-relaxed flex gap-2">
                    <AlertTriangle className="h-4 w-4 shrink-0 text-amber-400 mt-0.5" />
                    <span>
                      <strong>Disclaimer:</strong> Suitability scores represent corridor geometric evidence and connectivity from imagery. High scores do not prove physical road existence; they highlight discovery candidate corridors.
                    </span>
                  </div>
                </>
              ) : (
                <div className="p-3 border border-[#1f242c] rounded bg-gray-900/10 text-center text-gray-500 text-[10px]">
                  No alternative route selected. Choose a route from the list or click on the map.
                </div>
              )}

              {/* Export Route Option GeoJSON */}
              <button
                onClick={downloadSelectedRoute}
                disabled={!selectedAlternativeRoute}
                className={`w-full py-2 px-4 rounded text-[10px] font-bold uppercase tracking-wider transition-all flex items-center justify-center gap-2 shadow cursor-pointer ${
                  selectedAlternativeRoute
                    ? 'bg-[#1f242c] hover:bg-gray-800 text-emerald-400 hover:text-emerald-300'
                    : 'bg-gray-800 text-gray-500 cursor-not-allowed border border-transparent'
                }`}
              >
                <Download className="h-3.5 w-3.5" />
                <span>Export Route Option GeoJSON</span>
              </button>
            </div>

          </div>
        </div>
      )}
    </div>
  );
}
