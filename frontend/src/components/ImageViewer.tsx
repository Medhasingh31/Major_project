import React, { useState, useRef, useEffect } from 'react';
import { 
  ZoomIn, 
  ZoomOut, 
  RotateCcw,
  Sliders,
  Layers
} from 'lucide-react';
import { apiService } from '../services/api';
import { AnalysisResult, FlaggedIssue } from '../types';

interface ImageViewerProps {
  result: AnalysisResult | null;
  selectedIssue?: FlaggedIssue | null;
  onSelectIssue?: (issue: FlaggedIssue | null) => void;
  selectedRouteId?: number | string | null;
  onSelectRoute?: (routeId: number | string | null, routeData?: any) => void;
  highlightNewRoutes?: boolean;
  highlightQuality?: boolean;
}

export default function ImageViewer({ 
  result, 
  selectedIssue = null, 
  onSelectIssue,
  selectedRouteId = null,
  onSelectRoute,
  highlightNewRoutes = false,
  highlightQuality = false
}: ImageViewerProps) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [opacity, setOpacity] = useState(0.85);
  
  // Pipeline selector tabs
  // Options: 'input' | 'raw_mask' | 'repaired' | 'skeleton' | 'overlay' | 'network'
  const [activeBkg, setActiveBkg] = useState<'input' | 'raw_mask' | 'repaired' | 'skeleton' | 'overlay' | 'network'>('input');
  
  // Custom layer visibility toggles
  const [layers, setLayers] = useState({
    centerlines: true,
    nodes: true,
    issues: true,
  });

  const [cursorCoords, setCursorCoords] = useState({ x: 0, y: 0, lat: '31.9700° N', lng: '97.2400° W' });
  const [geojsonData, setGeojsonData] = useState<any>(null);
  const [isLoadingGeojson, setIsLoadingGeojson] = useState(false);

  const containerRef = useRef<HTMLDivElement>(null);

  // Fetch GeoJSON when a new result is loaded
  useEffect(() => {
    if (result?.projectId) {
      setIsLoadingGeojson(true);
      fetch(apiService.getLayerUrl(result.projectId, 'geojson'))
        .then((res) => {
          if (!res.ok) throw new Error('Failed to load');
          return res.json();
        })
        .then((data) => {
          setGeojsonData(data);
          setIsLoadingGeojson(false);
        })
        .catch((err) => {
          console.error('Error loading GeoJSON:', err);
          setIsLoadingGeojson(false);
        });
    } else {
      setGeojsonData(null);
    }
  }, [result]);

  // Handle default tab select when result arrives
  useEffect(() => {
    if (result) {
      setActiveBkg('input');
    }
  }, [result]);

  // Drag pan handlers
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
      
      // Geodetic offsets based on pipeline mapping formula
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

  // If no result is loaded, show the sophisticated empty state
  if (!result) {
    return (
      <div className="h-full flex flex-col items-center justify-center bg-[#0d121a] border border-[#1f242c] rounded text-center p-8 space-y-4 gis-grid">
        <div className="text-gray-600 bg-gray-900/60 p-4 rounded-full border border-[#1f242c] relative">
          <Layers className="h-10 w-10 text-gray-500" />
          <div className="absolute inset-0 border border-emerald-500/20 rounded-full animate-ping" />
        </div>
        <div className="space-y-1">
          <h4 className="text-xs font-semibold text-white tracking-widest font-mono uppercase">NO ANALYSIS LOADED</h4>
          <p className="text-[11px] text-gray-400 max-w-xs leading-relaxed">
            Upload an aerial image and run the extraction pipeline to begin.
          </p>
        </div>
      </div>
    );
  }

  // Determine back image url mapping
  const bkgType = activeBkg === 'network' || activeBkg === 'input' ? 'original' : activeBkg;
  const bkgUrl = apiService.getLayerUrl(result.projectId, bkgType);

  // Parse lines & points from loaded GeoJSON
  const features = geojsonData?.features || [];
  const lines = features.filter((f: any) => f.geometry?.type === 'LineString');
  const nodes = features.filter((f: any) => f.geometry?.type === 'Point');

  const getLinePath = (coords: number[][]) => {
    return coords.map((c, i) => `${i === 0 ? 'M' : 'L'} ${c[0]} ${c[1]}`).join(' ');
  };

  return (
    <div 
      className="flex flex-col h-full bg-[#0d121a] border border-[#1f242c] rounded overflow-hidden relative select-none"
      ref={containerRef}
    >
      {/* 1. Pipeline Selector Tabs (Top center overlay) */}
      <div className="absolute top-4 left-4 right-4 flex justify-between items-center pointer-events-none z-10">
        <div className="flex gap-1 pointer-events-auto bg-[#0b0f14]/85 border border-[#1f242c] p-1 rounded-md backdrop-blur-sm">
          {(['input', 'raw_mask', 'repaired', 'skeleton', 'overlay', 'network'] as const).map((stage) => (
            <button
              key={stage}
              onClick={() => setActiveBkg(stage)}
              className={`px-3 py-1.5 text-[9px] font-mono font-semibold tracking-wider rounded uppercase transition-colors cursor-pointer ${
                activeBkg === stage 
                  ? 'bg-emerald-500/25 text-emerald-400 border border-emerald-500/25' 
                  : 'text-gray-400 hover:text-white hover:bg-gray-800/35'
              }`}
            >
              {stage === 'raw_mask' ? 'RAW MASK' : stage === 'repaired' ? 'REPAIRED MASK' : stage}
            </button>
          ))}
        </div>

        {/* Zoom & Opacity controllers */}
        <div className="flex items-center gap-2 pointer-events-auto bg-[#0b0f14]/85 border border-[#1f242c] p-1 rounded-md backdrop-blur-sm text-xs">
          {activeBkg !== 'network' && activeBkg !== 'input' && (
            <div className="flex items-center gap-1.5 mr-2">
              <Sliders className="h-3.5 w-3.5 text-gray-500" />
              <input 
                type="range" 
                min="0.1" 
                max="1.0" 
                step="0.05"
                value={opacity}
                onChange={(e) => setOpacity(parseFloat(e.target.value))}
                className="w-16 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
              />
            </div>
          )}
          
          <button onClick={() => handleZoom('in')} className="p-1 hover:text-white text-gray-400" title="Zoom In"><ZoomIn className="h-3.5 w-3.5" /></button>
          <button onClick={() => handleZoom('out')} className="p-1 hover:text-white text-gray-400" title="Zoom Out"><ZoomOut className="h-3.5 w-3.5" /></button>
          <button onClick={resetView} className="p-1 hover:text-white text-gray-400" title="Reset View"><RotateCcw className="h-3.5 w-3.5" /></button>
        </div>
      </div>

      {/* 2. Interactive SVG Map Canvas Viewport */}
      <div 
        className="flex-1 relative cursor-crosshair overflow-hidden"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <div className="absolute inset-0 gis-grid opacity-25" />

        {/* Real Backend Raster Image */}
        <img 
          src={bkgUrl}
          alt={activeBkg}
          className="absolute inset-0 w-full h-full object-contain pointer-events-none transition-opacity duration-200"
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: 'center center',
            opacity: activeBkg === 'network' || activeBkg === 'input' ? 1.0 : opacity
          }}
        />

        {/* SVG Vector geometries layer */}
        <svg 
          className="absolute inset-0 w-full h-full pointer-events-none"
          viewBox="0 0 1024 1024"
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: 'center center',
            transition: isDragging ? 'none' : 'transform 0.12s ease-out'
          }}
        >
          {/* Real Centerlines paths parsed from GeoJSON */}
          {layers.centerlines && (activeBkg === 'network' || activeBkg === 'overlay') && lines.length > 0 && (
            <g fill="none" strokeLinecap="round" opacity="0.95">
              {lines.map((line: any, idx: number) => {
                const isSelected = selectedRouteId !== null && selectedRouteId !== undefined && (selectedRouteId === idx || selectedRouteId === line.properties?.source || selectedRouteId === `route-${idx}`);
                let strokeColor = '#10b981';
                let strokeWidth = isSelected ? '4.5' : '2.5';
                let strokeDash = 'none';

                if (highlightNewRoutes) {
                  // Potential New Routes styling
                  strokeColor = isSelected ? '#fbbf24' : '#f59e0b';
                } else if (highlightQuality) {
                  // Quality indicator coloring based on length / connectivity
                  const len = line.properties?.length_pixels || 0;
                  strokeColor = len > 40 ? '#10b981' : len > 15 ? '#f59e0b' : '#ef4444';
                }

                return (
                  <path 
                    key={idx} 
                    d={getLinePath(line.geometry.coordinates)} 
                    stroke={strokeColor}
                    strokeWidth={strokeWidth}
                    strokeDasharray={strokeDash}
                    className={(highlightNewRoutes || highlightQuality || onSelectRoute) ? "cursor-pointer pointer-events-auto hover:opacity-100 transition-all" : ""}
                    onClick={() => onSelectRoute?.(idx, line)}
                  />
                );
              })}
            </g>
          )}

          {/* Real Junction nodes from GeoJSON */}
          {layers.nodes && activeBkg === 'network' && nodes.length > 0 && (
            <g fill="#22d3ee" stroke="#0b0f14" strokeWidth="0.8" opacity="0.95">
              {nodes.map((node: any, idx: number) => {
                const [cx, cy] = node.geometry.coordinates;
                return <circle key={idx} cx={cx} cy={cy} r="4" />;
              })}
            </g>
          )}

          {/* Topology Issues Pin Markups */}
          {layers.issues && (activeBkg === 'network' || activeBkg === 'overlay') && result.flaggedIssues.length > 0 && (
            <g opacity="0.95">
              {result.flaggedIssues.map((issue) => {
                const isSelected = selectedIssue?.id === issue.id;
                // Translate percentage coordinates to 1024 viewBox pixels
                const cx = (issue.coords.x / 100) * 1024;
                const cy = (issue.coords.y / 100) * 1024;
                
                return (
                  <circle
                    key={issue.id}
                    cx={cx}
                    cy={cy}
                    r={isSelected ? 10 : 6}
                    fill={issue.confidence < 50 ? '#ef4444' : '#f59e0b'}
                    className="cursor-pointer pointer-events-auto"
                    onClick={() => onSelectIssue?.(issue)}
                  >
                    {isSelected && <animate attributeName="r" values="7;11;7" dur="1.2s" repeatCount="indefinite" />}
                  </circle>
                );
              })}
            </g>
          )}
        </svg>

        {/* Floating tooltip markup for active issue */}
        {selectedIssue && (activeBkg === 'network' || activeBkg === 'overlay') && (
          <div 
            className="absolute bg-[#0b0f14]/95 border border-red-500/40 px-3 py-1.5 rounded text-[10px] font-mono pointer-events-auto shadow-lg"
            style={{
              left: `calc(${selectedIssue.coords.x}% - 40px)`,
              top: `calc(${selectedIssue.coords.y}% - 48px)`,
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transformOrigin: 'center center'
            }}
          >
            <div className="text-red-400 font-semibold">{selectedIssue.reference}</div>
            <div className="text-gray-300">{selectedIssue.description}</div>
          </div>
        )}
      </div>

      {/* 3. Status Metadata Bar (Bottom) */}
      <div className="h-8 border-t border-[#1f242c] bg-[#0b0f14] flex justify-between items-center px-4 text-[10px] font-mono text-gray-500">
        <div>CURSOR: {cursorCoords.lat}, {cursorCoords.lng} (x: {cursorCoords.x}px, y: {cursorCoords.y}px)</div>
        <div className="flex gap-4">
          <span className="uppercase text-emerald-400">STAGE: {activeBkg.replace('_', ' ')}</span>
          <span>·</span>
          <span>IMAGE: 1024 × 1024</span>
          <span>·</span>
          <span>ZOOM: {Math.round(zoom * 100)}%</span>
        </div>
      </div>
    </div>
  );
}
