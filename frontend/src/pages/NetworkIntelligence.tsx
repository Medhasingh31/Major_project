import React, { useState, useEffect, useRef } from 'react';
import { 
  GitBranch, 
  GitCommit, 
  Hash, 
  Database,
  Info,
  TrendingUp,
  Activity,
  AlertTriangle,
  ZoomIn,
  ZoomOut,
  RotateCcw
} from 'lucide-react';
import { apiService } from '../services/api';

export default function NetworkIntelligence() {
  const [activeResult, setActiveResult] = useState<any>(null);
  const [intelData, setIntelData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // SVG Node-link graph states
  const [nodes, setNodes] = useState<any[]>([]);
  const [links, setLinks] = useState<any[]>([]);
  const [selectedNode, setSelectedNode] = useState<any>(null);
  const [hoveredNode, setHoveredNode] = useState<any>(null);
  const [draggedNodeId, setDraggedNodeId] = useState<number | null>(null);

  // Zoom/pan on topological SVG graph
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });

  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    // Scan sessionStorage for the latest run
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
      setActiveResult(latestData);
      fetchIntelligence(latestData.projectId);
    }
  }, []);

  const fetchIntelligence = async (projectId: string) => {
    setIsLoading(true);
    setErrorMsg(null);
    try {
      const data = await apiService.getIntelligence(projectId);
      setIntelData(data);
      initializeGraphLayout(data.d3Graph);
    } catch (err: any) {
      console.error("Failed to load intelligence stats:", err);
      setErrorMsg(err.message || "Failed to load network intelligence.");
    } finally {
      setIsLoading(false);
    }
  };

  const initializeGraphLayout = (d3Graph: any) => {
    if (!d3Graph || !d3Graph.nodes.length) {
      setNodes([]);
      setLinks([]);
      return;
    }

    const w = 600;
    const h = 400;
    const padding = 40;

    // Find bounds of node coordinates (x, y are spatial metrics)
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;

    d3Graph.nodes.forEach((n: any) => {
      if (n.x < minX) minX = n.x;
      if (n.x > maxX) maxX = n.x;
      if (n.y < minY) minY = n.y;
      if (n.y > maxY) maxY = n.y;
    });

    const rangeX = maxX - minX || 1.0;
    const rangeY = maxY - minY || 1.0;

    // Scale spatial nodes coordinates to SVG coordinates
    const scaledNodes = d3Graph.nodes.map((n: any) => {
      const sx = padding + ((n.x - minX) / rangeX) * (w - 2 * padding);
      // Flip Y axis for SVG standard viewport mapping
      const sy = h - (padding + ((n.y - minY) / rangeY) * (h - 2 * padding));
      return {
        ...n,
        x: sx,
        y: sy,
        origX: sx,
        origY: sy
      };
    });

    setNodes(scaledNodes);
    setLinks(d3Graph.links);
  };

  // Node Dragging Handlers
  const handleNodeMouseDown = (e: React.MouseEvent, nodeId: number) => {
    e.stopPropagation();
    setDraggedNodeId(nodeId);
  };

  const handleSvgMouseMove = (e: React.MouseEvent) => {
    if (draggedNodeId !== null && svgRef.current) {
      const rect = svgRef.current.getBoundingClientRect();
      // Translate client cursor to local SVG view bounds (600x400)
      const cursorX = ((e.clientX - rect.left) / rect.width) * 600;
      const cursorY = ((e.clientY - rect.top) / rect.height) * 400;

      // Adjust for current pan and zoom
      const adjX = (cursorX - pan.x) / zoom;
      const adjY = (cursorY - pan.y) / zoom;

      setNodes(prev => prev.map(n => n.id === draggedNodeId ? { ...n, x: adjX, y: adjY } : n));
      return;
    }

    if (isPanning) {
      setPan({
        x: e.clientX - panStart.x,
        y: e.clientY - panStart.y
      });
    }
  };

  const handleSvgMouseUp = () => {
    setDraggedNodeId(null);
    setIsPanning(false);
  };

  // Pan Graph handlers
  const handleSvgMouseDown = (e: React.MouseEvent) => {
    if (draggedNodeId !== null) return;
    setIsPanning(true);
    setPanStart({
      x: e.clientX - pan.x,
      y: e.clientY - pan.y
    });
  };

  const handleZoom = (direction: 'in' | 'out') => {
    setZoom(z => {
      const step = 0.2;
      return direction === 'in' ? Math.min(3, z + step) : Math.max(0.6, z - step);
    });
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    // Reset nodes back to original scaled layout positions
    setNodes(prev => prev.map(n => ({ ...n, x: n.origX, y: n.origY })));
  };

  // Hover node/links highlighting helpers
  const isAdjacent = (nodeIdA: number, nodeIdB: number) => {
    return links.some(l => 
      (l.source === nodeIdA && l.target === nodeIdB) || 
      (l.source === nodeIdB && l.target === nodeIdA)
    );
  };

  const getComponentColor = (groupId: number) => {
    const colors = [
      '#10b981', // emerald-500
      '#f59e0b', // amber-500
      '#3b82f6', // blue-500
      '#ec4899', // pink-500
      '#a855f7', // purple-500
      '#06b6d4', // cyan-500
      '#f43f5e'  // rose-500
    ];
    return colors[groupId % colors.length];
  };

  return (
    <div className="h-full overflow-y-auto p-8 space-y-8 bg-[#070a0e] gis-grid font-mono">
      {/* Header */}
      <div className="border border-[#1f242c] bg-[#0b0f14]/80 p-6 rounded-lg space-y-2">
        <div className="flex justify-between items-center">
          <h2 className="text-lg font-semibold text-white tracking-wider uppercase">Network Topology Intelligence</h2>
          {intelData && (
            <span className="text-xs text-emerald-400 bg-emerald-500/10 px-2 py-0.5 border border-emerald-500/20 rounded">
              Source: {intelData.projectName}
            </span>
          )}
        </div>
        <p className="text-xs text-gray-400 max-w-3xl leading-relaxed">
          Graph-theoretical analysis of the extracted segment model. Network structures are parsed into formal node-link descriptors to evaluate connectivity metrics, structural subgraphs, and junction dead-ends.
        </p>
      </div>

      {errorMsg && (
        <div className="p-4 border border-red-500/30 bg-red-950/20 rounded text-xs text-red-400 flex items-start gap-2">
          <AlertTriangle className="h-4.5 w-4.5 shrink-0 mt-0.5" />
          <span>{errorMsg}</span>
        </div>
      )}

      {isLoading && (
        <div className="p-5 border border-emerald-500/25 bg-emerald-500/5 rounded text-sm text-emerald-400 flex items-center gap-3">
          <div className="h-4 w-4 border border-emerald-500 border-t-transparent rounded-full animate-spin" />
          Analyzing topological graph network...
        </div>
      )}

      {intelData && (
        <>
          {/* Dynamic Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 text-xs">
            <StatsCard label="Nodes (V)" value={intelData.nodes} icon={GitCommit} desc="Junction coordinate intersections" />
            <StatsCard label="Edges (E)" value={intelData.edges} icon={GitBranch} desc="Connected centerline segments" />
            <StatsCard label="Average Degree (k)" value={intelData.avgDegree} icon={Activity} desc="Mean connections per node" />
            <StatsCard label="Subgraphs (C)" value={intelData.componentsCount} icon={Database} desc="Isolated network components" />
          </div>

          {/* Breakdown Distributions */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <div className="lg:col-span-2 border border-[#1f242c] bg-[#0b0f14]/80 rounded-lg p-6 space-y-6">
              <h3 className="text-sm font-semibold text-white tracking-wider uppercase">Structural Connectivity Analysis</h3>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Component distribution size */}
                <div className="border border-[#1f242c] p-4 rounded bg-[#0d121a] space-y-3">
                  <h4 className="text-[10px] uppercase tracking-widest text-gray-500">Connected Component Size Distribution</h4>
                  <div className="space-y-3 max-h-48 overflow-y-auto pr-1">
                    {intelData.components.map((comp: any) => (
                      <ComponentBar 
                        key={comp.id} 
                        label={comp.label} 
                        pct={comp.percentage} 
                        count={comp.count} 
                        color={getComponentColor(comp.id)}
                      />
                    ))}
                  </div>
                </div>

                {/* Junction Degree classification */}
                <div className="border border-[#1f242c] p-4 rounded bg-[#0d121a] space-y-3">
                  <h4 className="text-[10px] uppercase tracking-widest text-gray-500">Junction Degree Classes</h4>
                  <div className="space-y-3">
                    {intelData.degreeDistribution.map((deg: any, idx: number) => (
                      <DegreeBar 
                        key={idx} 
                        label={deg.label} 
                        count={deg.count} 
                        pct={deg.percentage} 
                      />
                    ))}
                  </div>
                </div>
              </div>

              {/* D3/SVG Interactive Node-link Graph */}
              <div className="border border-[#1f242c] p-4 rounded bg-[#0d121a] space-y-4">
                <div className="flex justify-between items-center">
                  <div>
                    <h4 className="text-[10px] uppercase tracking-widest text-gray-500">Topological Node-Link Graph View</h4>
                    <p className="text-[9px] text-gray-400 mt-0.5">Abstract mathematical node coordinates representation. Node dragging enabled.</p>
                  </div>

                  <div className="flex items-center gap-1.5 bg-[#0b0f14] border border-[#1f242c] p-1 rounded text-xs pointer-events-auto">
                    <button onClick={() => handleZoom('in')} className="p-1 hover:text-white text-gray-400 cursor-pointer" title="Zoom In"><ZoomIn className="h-3.5 w-3.5" /></button>
                    <button onClick={() => handleZoom('out')} className="p-1 hover:text-white text-gray-400 cursor-pointer" title="Zoom Out"><ZoomOut className="h-3.5 w-3.5" /></button>
                    <button onClick={resetView} className="p-1 hover:text-white text-gray-400 cursor-pointer" title="Reset Graph Layout"><RotateCcw className="h-3.5 w-3.5" /></button>
                  </div>
                </div>

                <div className="relative border border-[#1f242c] rounded overflow-hidden bg-[#070a0e]/95 h-96">
                  <svg
                    ref={svgRef}
                    className="w-full h-full cursor-grab active:cursor-grabbing"
                    viewBox="0 0 600 400"
                    onMouseMove={handleSvgMouseMove}
                    onMouseUp={handleSvgMouseUp}
                    onMouseDown={handleSvgMouseDown}
                  >
                    <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
                      {/* Edges */}
                      {links.map((link, idx) => {
                        const sourceNode = nodes.find(n => n.id === link.source);
                        const targetNode = nodes.find(n => n.id === link.target);
                        if (!sourceNode || !targetNode) return null;

                        const isHighlighted = hoveredNode && 
                          (hoveredNode.id === sourceNode.id || hoveredNode.id === targetNode.id);

                        return (
                          <line
                            key={`edge-${idx}`}
                            x1={sourceNode.x}
                            y1={sourceNode.y}
                            x2={targetNode.x}
                            y2={targetNode.y}
                            stroke={isHighlighted ? '#f59e0b' : '#334155'}
                            strokeWidth={isHighlighted ? 2.5 : 1.2}
                            opacity={hoveredNode ? (isHighlighted ? 1.0 : 0.25) : 0.6}
                          />
                        );
                      })}

                      {/* Nodes */}
                      {nodes.map(node => {
                        const isNodeHovered = hoveredNode && hoveredNode.id === node.id;
                        const isNodeAdjacent = hoveredNode && isAdjacent(node.id, hoveredNode.id);
                        
                        let opacity = 0.85;
                        if (hoveredNode) {
                          opacity = isNodeHovered || isNodeAdjacent ? 1.0 : 0.25;
                        }

                        return (
                          <circle
                            key={`node-${node.id}`}
                            cx={node.x}
                            cy={node.y}
                            r={isNodeHovered ? 7.5 : (node.degree >= 3 ? 5.5 : 4)}
                            fill={getComponentColor(node.group)}
                            stroke={isNodeHovered ? '#ffffff' : (selectedNode && selectedNode.id === node.id ? '#f59e0b' : '#0b0f14')}
                            strokeWidth={isNodeHovered || (selectedNode && selectedNode.id === node.id) ? 1.8 : 0.8}
                            opacity={opacity}
                            className="transition-all duration-75 cursor-pointer"
                            onMouseEnter={() => setHoveredNode(node)}
                            onMouseLeave={() => setHoveredNode(null)}
                            onMouseDown={(e) => handleNodeMouseDown(e, node.id)}
                            onClick={(e) => {
                              e.stopPropagation();
                              setSelectedNode(node);
                            }}
                          />
                        );
                      })}
                    </g>
                  </svg>

                  {/* Interactive Details Overlay inside Map */}
                  {(hoveredNode || selectedNode) && (
                    <div className="absolute bottom-3 left-3 bg-[#0b0f14]/90 border border-[#1f242c] p-3.5 rounded text-[10px] space-y-1 max-w-xs font-mono backdrop-blur-sm">
                      <div className="text-emerald-400 font-bold uppercase tracking-wider text-[9px] border-b border-gray-800 pb-1 mb-1 flex justify-between">
                        <span>Node Diagnostics</span>
                        <span className="text-gray-500 font-normal">{hoveredNode ? 'Hover' : 'Selected'}</span>
                      </div>
                      {(() => {
                        const n = hoveredNode || selectedNode;
                        return (
                          <>
                            <div className="text-white truncate">ID: {n.name}</div>
                            <div className="text-gray-300">Degree: {n.degree} ({n.degree === 1 ? 'Dead End' : (n.degree === 2 ? 'Continuation' : 'Junction')})</div>
                            <div className="text-gray-300">Connected Component: Subgraph C_{n.group}</div>
                          </>
                        );
                      })()}
                    </div>
                  )}
                </div>
              </div>

              {/* Geodetic note */}
              <div className="bg-emerald-500/5 border border-emerald-500/20 p-4 rounded flex gap-3 text-xs">
                <Info className="h-5 w-5 text-emerald-400 shrink-0 mt-0.5" />
                <div className="space-y-1">
                  <div className="font-semibold text-white">Geodetic Graph Verification</div>
                  <div className="text-gray-400 leading-normal">
                    Connectivity: {intelData.connectivityDesc}. Nodes and links are mapped onto geodetic parameters relative to the spatial input boundaries.
                  </div>
                </div>
              </div>
            </div>

            {/* Warnings sidebar card panel */}
            <div className="border border-[#1f242c] bg-[#0b0f14]/80 rounded-lg p-6 space-y-6 text-xs">
              <h3 className="text-sm font-semibold text-white tracking-wider uppercase">Network Warnings</h3>
              
              <div className="space-y-4">
                <WarningCard 
                  title="Disconnected Subgraphs" 
                  count={intelData.componentsCount} 
                  desc="Isolated network component subgraphs failing to link to the primary arterial network graph." 
                  severity={intelData.componentsCount > 1 ? "medium" : "low"}
                />
                <WarningCard 
                  title="Dead Ends" 
                  count={intelData.deadEndsCount} 
                  desc="Topological network continuation points terminating without loop closures or junction attachments." 
                  severity="low"
                />
                <WarningCard 
                  title="Total Junctions" 
                  count={intelData.junctionsCount} 
                  desc="Intersections with degree >= 3 representing nodes where roads join or cross." 
                  severity="low"
                />
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function StatsCard({ label, value, icon: Icon, desc }: { label: string; value: any; icon: any; desc: string }) {
  return (
    <div className="border border-[#1f242c] bg-[#0b0f14]/50 p-4 rounded-lg flex items-center justify-between">
      <div className="space-y-1.5">
        <span className="text-gray-500 text-[10px] tracking-widest uppercase">{label}</span>
        <div className="text-xl font-bold text-white">{value}</div>
        <div className="text-[9px] text-gray-400 leading-normal">{desc}</div>
      </div>
      <div className="text-gray-600 bg-gray-900/80 p-2 rounded border border-[#1f242c]">
        <Icon className="h-4.5 w-4.5" />
      </div>
    </div>
  );
}

function ComponentBar({ label, pct, count, color }: { label: string; pct: number; count: number; color: string }) {
  return (
    <div className="space-y-1 font-mono text-[10px]">
      <div className="flex justify-between text-gray-400">
        <span>{label}</span>
        <span className="text-white">{count} nodes ({pct}%)</span>
      </div>
      <div className="w-full bg-gray-800 h-1.5 rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function DegreeBar({ label, count, pct }: { label: string; count: number; pct: number }) {
  return (
    <div className="space-y-1 font-mono text-[10px]">
      <div className="flex justify-between text-gray-400">
        <span>{label}</span>
        <span className="text-white">{count} nodes ({pct}%)</span>
      </div>
      <div className="w-full bg-gray-800 h-1.5 rounded-full overflow-hidden">
        <div className="bg-cyan-500 h-full rounded-full" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function WarningCard({ title, count, desc, severity }: { title: string; count: number; desc: string; severity: 'low' | 'medium' | 'high' }) {
  return (
    <div className="p-4 rounded border border-[#1f242c] bg-gray-900/30 flex gap-3">
      <AlertTriangle className={`h-4.5 w-4.5 shrink-0 mt-0.5 ${
        severity === 'high' ? 'text-red-400' :
        severity === 'medium' ? 'text-amber-400' : 'text-blue-400'
      }`} />
      <div className="space-y-1.5 flex-1 min-w-0">
        <div className="flex justify-between items-center">
          <span className="font-semibold text-white font-mono">{title}</span>
          <span className="bg-gray-800 text-white font-mono px-1.5 py-0.5 rounded text-[10px]">{count}</span>
        </div>
        <p className="text-gray-400 text-[10px] leading-normal">{desc}</p>
      </div>
    </div>
  );
}
