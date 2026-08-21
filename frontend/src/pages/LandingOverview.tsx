import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  ArrowRight, 
  MapPin, 
  Clock, 
  Cpu, 
  GitBranch, 
  Hash, 
  Database,
  Calendar
} from 'lucide-react';
import { apiService } from '../services/api';

export default function LandingOverview() {
  const [recentRuns, setRecentRuns] = useState<{ id: string; name: string; date: string; segments: number; length: string }[]>([]);

  useEffect(() => {
    // Collect runs from session storage + default mock
    const runs = [
      { id: 'proj-001', name: 'Meridian County Corridor', date: '2026-08-21', segments: 342, length: '124.8 km' }
    ];

    // Scan sessionStorage for real user jobs
    for (let i = 0; i < sessionStorage.length; i++) {
      const key = sessionStorage.key(i);
      if (key && key.startsWith('analysis_result_')) {
        try {
          const raw = sessionStorage.getItem(key);
          if (raw) {
            const data = JSON.parse(raw);
            runs.unshift({
              id: data.projectId,
              name: data.projectName || 'Recent Run',
              date: data.created_at || new Date().toISOString().split('T')[0],
              segments: data.networkSummary?.totalSegments || 0,
              length: `${data.networkSummary?.totalRoadLength?.value || 0} ${data.networkSummary?.totalRoadLength?.unit || 'km'}`
            });
          }
        } catch {}
      }
    }
    setRecentRuns(runs);
  }, []);

  return (
    <div className="h-full overflow-y-auto p-8 space-y-8 bg-[#070a0e] gis-grid">
      {/* Hero Welcome Banner */}
      <div className="bg-gradient-to-r from-emerald-950/30 to-blue-950/20 border border-[#1f242c] p-8 rounded-lg relative overflow-hidden">
        <div className="absolute right-0 top-0 h-full w-1/3 bg-[radial-gradient(ellipse_at_top_right,rgba(16,185,129,0.08),transparent_50%)] pointer-events-none" />
        <div className="max-w-3xl space-y-4">
          <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-mono font-semibold tracking-wider text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 uppercase">
            Platform Ready
          </div>
          <h2 className="text-2xl font-bold text-white tracking-tight">ROADINTEL: AI-Powered Road Network Intelligence</h2>
          <p className="text-sm text-gray-400 leading-relaxed">
            Extract topological networks, intersection centerlines, and geodetic metrics directly from aerial and satellite imagery. RoadIntel combines custom U-Net computer vision segmentation with Classical morphological cleaning and structural graph-repair algorithms.
          </p>
          <div className="pt-2">
            <Link 
              to="/workspace"
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-emerald-500 hover:bg-emerald-600 text-[#070a0e] font-semibold rounded text-sm transition-all shadow-md shadow-emerald-500/10 cursor-pointer"
            >
              Start Analysis Workspace
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </div>

      {/* Quick Metrics (Primary Demo / Status statistics) */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <MetricCard title="Detected Roads" value="124.8 km" desc="Across study regions" icon={GitBranch} />
        <MetricCard title="Network Junctions" value="312" desc="Validated topology nodes" icon={Hash} />
        <MetricCard title="Connected Components" value="3" desc="Isolated subgraph clusters" icon={Database} />
        <MetricCard title="Extraction Confidence" value="89.6%" desc="Weighted metric reliability" icon={Cpu} />
      </div>

      {/* Dual Panel layouts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left 2 cols: Recent runs */}
        <div className="lg:col-span-2 border border-[#1f242c] bg-[#0b0f14]/80 rounded-lg p-6 space-y-4">
          <h3 className="text-sm font-semibold text-white tracking-wider font-mono uppercase">Recent Extraction Tasks</h3>
          <div className="overflow-hidden border border-[#1f242c] rounded">
            <table className="w-full text-left text-sm text-gray-400">
              <thead className="text-[10px] uppercase font-mono tracking-widest text-gray-500 border-b border-[#1f242c] bg-gray-900/50">
                <tr>
                  <th className="px-6 py-3 font-semibold">Run / Project Name</th>
                  <th className="px-6 py-3 font-semibold">Segments</th>
                  <th className="px-6 py-3 font-semibold">Length</th>
                  <th className="px-6 py-3 font-semibold">Execution Date</th>
                  <th className="px-6 py-3 font-semibold">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1f242c]">
                {recentRuns.map((run) => (
                  <tr key={run.id} className="hover:bg-gray-900/30 transition-colors">
                    <td className="px-6 py-4 font-medium text-white flex items-center gap-2">
                      <div className="h-2 w-2 rounded bg-emerald-400" />
                      {run.name}
                    </td>
                    <td className="px-6 py-4 font-mono">{run.segments}</td>
                    <td className="px-6 py-4 font-mono">{run.length}</td>
                    <td className="px-6 py-4 text-xs text-gray-500">{run.date}</td>
                    <td className="px-6 py-4">
                      <Link 
                        to={`/workspace?runId=${run.id}`}
                        className="text-xs text-emerald-400 hover:text-emerald-300 font-semibold"
                      >
                        Load Workspace →
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right 1 col: System health metrics */}
        <div className="border border-[#1f242c] bg-[#0b0f14]/80 rounded-lg p-6 space-y-6">
          <h3 className="text-sm font-semibold text-white tracking-wider font-mono uppercase font-medium">Pipeline Topology Prior</h3>
          <div className="space-y-4 text-xs">
            <div className="p-3 bg-gray-900/40 rounded border border-[#1f242c] space-y-2">
              <div className="flex justify-between items-center text-gray-400">
                <span>Morphological kernel:</span>
                <span className="font-mono text-white">Disk (Closing)</span>
              </div>
              <div className="flex justify-between items-center text-gray-400">
                <span>Default Threshold:</span>
                <span className="font-mono text-white">0.30</span>
              </div>
              <div className="flex justify-between items-center text-gray-400">
                <span>CRS mapping:</span>
                <span className="font-mono text-white">WGS 84 (EPSG:4326)</span>
              </div>
              <div className="flex justify-between items-center text-gray-400">
                <span>Grid Resolution:</span>
                <span className="font-mono text-white">0.15 m/pixel</span>
              </div>
            </div>

            <div className="space-y-3">
              <h4 className="font-semibold text-white tracking-wide text-[10px] uppercase font-mono text-gray-500">Pipeline Stages</h4>
              <div className="space-y-2 font-mono">
                <StageIndicator label="1. Satellite Raster Input" active />
                <StageIndicator label="2. Preprocessing & Augment" active />
                <StageIndicator label="3. Deep U-Net Inference" active />
                <StageIndicator label="4. Morphology & Spur Cleanup" active />
                <StageIndicator label="5. Node Graph Construction" active />
                <StageIndicator label="6. Topology Geodesic Validation" active />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ title, value, desc, icon: Icon }: { title: string; value: string; desc: string; icon: any }) {
  return (
    <div className="border border-[#1f242c] bg-[#0b0f14]/65 rounded-lg p-5 flex items-center justify-between">
      <div className="space-y-1">
        <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider font-mono">{title}</span>
        <div className="text-xl font-bold text-white font-mono">{value}</div>
        <div className="text-[11px] text-gray-400">{desc}</div>
      </div>
      <div className="text-gray-600 bg-gray-900 p-2.5 rounded border border-[#1f242c]">
        <Icon className="h-5 w-5" />
      </div>
    </div>
  );
}

function StageIndicator({ label, active }: { label: string; active?: boolean }) {
  return (
    <div className="flex items-center gap-2">
      <div className={`h-1.5 w-1.5 rounded-full ${active ? 'bg-emerald-400' : 'bg-gray-700'}`} />
      <span className={active ? 'text-gray-300' : 'text-gray-600'}>{label}</span>
    </div>
  );
}
