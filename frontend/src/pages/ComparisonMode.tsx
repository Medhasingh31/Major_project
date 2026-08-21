import React, { useState } from 'react';
import { 
  Layers, 
  ArrowRight, 
  HelpCircle,
  Eye,
  Sliders,
  Calendar,
  Grid
} from 'lucide-react';

export default function ComparisonMode() {
  const [selectedPreset, setSelectedPreset] = useState<'meridian'>('meridian');
  const [swipeOffset, setSwipeOffset] = useState(50);
  const [viewMode, setViewMode] = useState<'side-by-side' | 'swipe'>('side-by-side');

  return (
    <div className="h-full flex overflow-hidden bg-[#070a0e]">
      {/* Configuration left panel */}
      <div className="w-80 border-r border-[#1f242c] bg-[#0b0f14]/40 flex flex-col justify-between overflow-y-auto shrink-0 p-5 space-y-6">
        <div className="space-y-6">
          <div className="space-y-2">
            <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider font-mono">1. Select Baseline Preset</label>
            <select
              className="w-full bg-gray-900 border border-[#1f242c] rounded px-3 py-2 text-xs text-white outline-none focus:border-emerald-500/50"
              value={selectedPreset}
              onChange={() => setSelectedPreset('meridian')}
            >
              <option value="meridian">Meridian County (2016 vs 2026)</option>
            </select>
          </div>

          <div className="space-y-2">
            <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider font-mono">2. Visualization Mode</label>
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <button
                onClick={() => setViewMode('side-by-side')}
                className={`py-2 px-3 border rounded text-center transition-colors cursor-pointer ${
                  viewMode === 'side-by-side' 
                    ? 'bg-emerald-500/10 border-emerald-500 text-emerald-400 font-semibold' 
                    : 'border-[#1f242c] text-gray-500 hover:text-white'
                }`}
              >
                Side-by-Side
              </button>
              <button
                onClick={() => setViewMode('swipe')}
                className={`py-2 px-3 border rounded text-center transition-colors cursor-pointer ${
                  viewMode === 'swipe' 
                    ? 'bg-emerald-500/10 border-emerald-500 text-emerald-400 font-semibold' 
                    : 'border-[#1f242c] text-gray-500 hover:text-white'
                }`}
              >
                Swipe Slider
              </button>
            </div>
          </div>

          {viewMode === 'swipe' && (
            <div className="space-y-2 text-xs">
              <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider font-mono">Swipe Divider Position</label>
              <input 
                type="range" 
                min="0" 
                max="100" 
                value={swipeOffset} 
                onChange={(e) => setSwipeOffset(parseInt(e.target.value))}
                className="w-full h-1 bg-gray-800 rounded-lg appearance-none cursor-pointer accent-emerald-500"
              />
            </div>
          )}

          {/* Temporal metrics card */}
          <div className="space-y-3">
            <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider font-mono">Change Statistics</label>
            <div className="border border-[#1f242c] rounded bg-gray-900/40 p-4 space-y-3 text-xs font-mono">
              <div className="flex justify-between items-center text-gray-400">
                <span>Total Added Roads:</span>
                <span className="text-emerald-400 font-bold">+18.4 km</span>
              </div>
              <div className="flex justify-between items-center text-gray-400">
                <span>Removed/Abandoned:</span>
                <span className="text-red-400 font-bold">-2.1 km</span>
              </div>
              <div className="flex justify-between items-center text-gray-400">
                <span>Connectivity change:</span>
                <span className="text-cyan-400 font-bold">+14.2%</span>
              </div>
              <div className="flex justify-between items-center text-gray-400">
                <span>New Junction Nodes:</span>
                <span className="text-white font-bold">+31</span>
              </div>
            </div>
          </div>
        </div>

        <div className="p-4 border-t border-[#1f242c] bg-gray-900/10 text-[10px] text-gray-500 rounded flex gap-2">
          <HelpCircle className="h-4 w-4 shrink-0 text-emerald-400" />
          <span>Change comparison logic compares extracted vector subgraphs from epoch A and B.</span>
        </div>
      </div>

      {/* Center canvas content */}
      <div className="flex-1 flex flex-col p-6 space-y-6 relative overflow-hidden min-w-0">
        <div className="flex-1 min-h-0 border border-[#1f242c] rounded bg-[#0d121a] flex relative overflow-hidden">
          {viewMode === 'side-by-side' ? (
            <div className="flex w-full h-full divide-x divide-[#1f242c]">
              {/* Epoch A left */}
              <div className="flex-1 h-full relative select-none">
                <div className="absolute top-4 left-4 z-10 bg-[#0b0f14]/80 border border-[#1f242c] px-3 py-1.5 rounded text-xs font-mono font-semibold text-white">
                  EPOCH A: 2016 Baseline
                </div>
                <div className="absolute inset-0 gis-grid opacity-25" />
                <div className="absolute inset-0 flex items-center justify-center text-xs text-gray-600 font-mono">
                  [ 2016 Imagery Raster ]
                </div>
              </div>
              
              {/* Epoch B right */}
              <div className="flex-1 h-full relative select-none">
                <div className="absolute top-4 left-4 z-10 bg-[#0b0f14]/80 border border-[#1f242c] px-3 py-1.5 rounded text-xs font-mono font-semibold text-white">
                  EPOCH B: 2026 Extraction
                </div>
                <div className="absolute inset-0 gis-grid opacity-25" />
                <div className="absolute inset-0 flex items-center justify-center text-xs text-gray-600 font-mono">
                  [ 2026 Imagery Raster + Vectors ]
                </div>
              </div>
            </div>
          ) : (
            <div className="w-full h-full relative select-none overflow-hidden flex items-center justify-center">
              {/* Background layer representing Epoch A */}
              <div className="absolute inset-0 gis-grid opacity-10" />
              <div className="absolute top-4 left-4 z-10 bg-[#0b0f14]/80 border border-[#1f242c] px-3 py-1.5 rounded text-xs font-mono font-semibold text-white">
                Swipe Viewport
              </div>
              
              {/* Vertical swipe line divider */}
              <div 
                className="absolute top-0 bottom-0 w-0.5 bg-emerald-500 z-10 cursor-col-resize flex items-center justify-center"
                style={{ left: `${swipeOffset}%` }}
              >
                <div className="bg-emerald-500 text-[#070a0e] p-1 rounded-full shadow-lg -translate-x-1/2">
                  <Grid className="h-3 w-3" />
                </div>
              </div>

              {/* Mask split container */}
              <div className="absolute inset-y-0 left-0 right-0 pointer-events-none flex items-center justify-center text-gray-500 font-mono text-sm">
                <div className="w-1/2 text-center">[ 2016 Epoch ]</div>
                <div className="w-1/2 text-center">[ 2026 Epoch ]</div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
