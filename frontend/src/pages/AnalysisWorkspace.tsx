import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { 
  Play, 
  Upload, 
  Download, 
  Sliders, 
  Cpu, 
  Info,
  ChevronRight,
  CheckCircle,
  AlertTriangle,
  Layers
} from 'lucide-react';
import { apiService } from '../services/api';
import { ExtractionConfig, AnalysisResult, FlaggedIssue, PipelineStage } from '../types';
import ImageViewer from '../components/ImageViewer';

const DEFAULT_CONFIG: ExtractionConfig = {
  threshold: 0.30,
  closingRadius: 6,
  minObjectSize: 32,
  bridgeKernelSize: 5,
  imageSize: 512,
  useModel: true
};

const INITIAL_STAGES: PipelineStage[] = [
  { id: 'input', name: '01 — INPUT', status: 'idle', label: 'Image uploaded and processed' },
  { id: 'segmentation', name: '02 — AI SEGMENTATION', status: 'idle', label: 'U-Net inference model execution' },
  { id: 'raw_mask', name: '03 — RAW MASK', status: 'idle', label: 'Road mask generated' },
  { id: 'mask_repair', name: '04 — MASK REPAIR', status: 'idle', label: 'Morphological repair operations' },
  { id: 'skeletonization', name: '05 — SKELETONIZATION', status: 'idle', label: 'Centerline skeleton extracted' },
  { id: 'geometry', name: '06 — GEOMETRY', status: 'idle', label: 'Road geometry elements generated' },
  { id: 'topology', name: '07 — TOPOLOGY', status: 'idle', label: 'Connectivity structures analyzed' },
  { id: 'graph', name: '08 — GRAPH', status: 'idle', label: 'Formal road graph constructed' },
  { id: 'export', name: '09 — EXPORT', status: 'idle', label: 'GeoJSON, GraphML, and diagnostics written' },
];

export default function AnalysisWorkspace() {
  const [searchParams] = useSearchParams();
  const runIdParam = searchParams.get('runId');

  // Input states
  const [projectName, setProjectName] = useState('Meridian Corridors Analysis');
  const [studyArea, setStudyArea] = useState('Meridian County');
  const [imageYear, setImageYear] = useState('2026');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  
  // Pipeline configurations
  const [config, setConfig] = useState<ExtractionConfig>(DEFAULT_CONFIG);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Status & output states
  const [isProcessing, setIsProcessing] = useState(false);
  const [progressStages, setProgressStages] = useState<PipelineStage[]>(INITIAL_STAGES);
  const [activeStageIdx, setActiveStageIdx] = useState(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [selectedIssue, setSelectedIssue] = useState<FlaggedIssue | null>(null);

  // Load project if runId is in search query
  useEffect(() => {
    if (runIdParam) {
      loadSavedRun(runIdParam);
    }
  }, [runIdParam]);

  const loadSavedRun = async (runId: string) => {
    setIsProcessing(true);
    setErrorMsg(null);
    try {
      const data = await apiService.getAnalysis(runId);
      setResult(data);
      sessionStorage.setItem(`analysis_result_${runId}`, JSON.stringify(data));
    } catch (err: any) {
      console.warn("Could not load from server, checking sessionStorage:", err);
      const cached = sessionStorage.getItem(`analysis_result_${runId}`);
      if (cached) {
        setResult(JSON.parse(cached));
      } else {
        setErrorMsg(err.message || 'Failed to load analysis.');
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setErrorMsg(null);
    }
  };

  const generateDummyFile = (): File => {
    const canvas = document.createElement('canvas');
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.fillStyle = '#0a1a0f';
      ctx.fillRect(0, 0, 256, 256);
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 4;
      ctx.beginPath();
      ctx.moveTo(0, 128);
      ctx.lineTo(256, 128);
      ctx.stroke();
    }
    const dataUrl = canvas.toDataURL('image/png');
    const blobBin = atob(dataUrl.split(',')[1]);
    const array = [];
    for (let i = 0; i < blobBin.length; i++) {
      array.push(blobBin.charCodeAt(i));
    }
    const fileBlob = new Blob([new Uint8Array(array)], { type: 'image/png' });
    return new File([fileBlob], 'synthetic_aerial_raster.png', { type: 'image/png' });
  };

  const triggerAnalysis = async () => {
    const fileToUpload = selectedFile || generateDummyFile();
    setIsProcessing(true);
    setErrorMsg(null);
    setResult(null);
    setSelectedIssue(null);
    setProgressStages(INITIAL_STAGES.map(s => ({ ...s, status: 'idle' })));
    setActiveStageIdx(0);

    const jobId = `job-${Date.now()}`;

    // Pipeline stage animations simulation
    const updateStage = (idx: number, status: 'processing' | 'completed' | 'failed') => {
      setProgressStages(prev => prev.map((s, i) => {
        if (i === idx) return { ...s, status };
        if (i < idx) return { ...s, status: 'completed' };
        return s;
      }));
      setActiveStageIdx(idx);
    };

    try {
      updateStage(0, 'processing');
      await new Promise(r => setTimeout(r, 600));
      
      updateStage(1, 'processing');
      await new Promise(r => setTimeout(r, 800));
      
      updateStage(2, 'processing');
      await new Promise(r => setTimeout(r, 500));
      
      // Dispatch requests to actual backend pipeline
      updateStage(3, 'processing');
      
      const responseData = await apiService.submitAnalysis(
        fileToUpload,
        config,
        jobId,
        projectName,
        studyArea,
        imageYear
      );

      updateStage(4, 'processing');
      await new Promise(r => setTimeout(r, 400));
      updateStage(5, 'processing');
      await new Promise(r => setTimeout(r, 400));
      updateStage(6, 'processing');
      await new Promise(r => setTimeout(r, 400));
      updateStage(7, 'processing');
      await new Promise(r => setTimeout(r, 400));
      updateStage(8, 'processing');
      await new Promise(r => setTimeout(r, 400));

      setProgressStages(prev => prev.map(s => ({ ...s, status: 'completed' })));
      setResult(responseData);
      
      sessionStorage.setItem(`analysis_result_${jobId}`, JSON.stringify(responseData));

    } catch (err: any) {
      setProgressStages(prev => prev.map((s, i) => i === activeStageIdx ? { ...s, status: 'failed' } : s));
      setErrorMsg(err.message || 'Pipeline extraction failed.');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="h-full flex overflow-hidden bg-[#070a0e]">
      {/* LEFT PANEL: Parameters & upload config */}
      <div className="w-80 border-r border-[#1f242c] bg-[#0b0f14]/40 flex flex-col justify-between overflow-y-auto shrink-0 p-5 space-y-6">
        <div className="space-y-6">
          <div className="space-y-2">
            <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider font-mono">1. Metadata</label>
            <input 
              type="text" 
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              placeholder="Project Name"
              className="w-full text-xs bg-gray-900 border border-[#1f242c] rounded px-3 py-2 text-white outline-none focus:border-emerald-500/50"
            />
            <div className="grid grid-cols-2 gap-2">
              <input 
                type="text" 
                value={studyArea}
                onChange={(e) => setStudyArea(e.target.value)}
                placeholder="Region"
                className="text-xs bg-gray-900 border border-[#1f242c] rounded px-3 py-2 text-white outline-none focus:border-emerald-500/50"
              />
              <input 
                type="text" 
                value={imageYear}
                onChange={(e) => setImageYear(e.target.value)}
                placeholder="Year"
                className="text-xs bg-gray-900 border border-[#1f242c] rounded px-3 py-2 text-white outline-none focus:border-emerald-500/50"
              />
            </div>
          </div>

          <div className="space-y-2">
            <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider font-mono">2. Imagery Input</label>
            <label className="flex flex-col items-center justify-center border border-dashed border-[#1f242c] hover:border-emerald-500/30 rounded-lg p-5 bg-gray-900/35 cursor-pointer group transition-all text-center">
              <Upload className="h-6 w-6 text-gray-500 group-hover:text-emerald-400 mb-2 transition-colors" />
              <span className="text-xs text-gray-300 font-medium">
                {selectedFile ? selectedFile.name : 'Select satellite image'}
              </span>
              <span className="text-[9px] text-gray-500 mt-1">GeoTIFF, JPG or PNG (Max 16MB)</span>
              <input 
                type="file" 
                accept=".tif,.tiff,.jpg,.jpeg,.png"
                onChange={handleFileChange}
                className="hidden"
              />
            </label>
          </div>

          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider font-mono">3. Parameter Weights</label>
              <button 
                onClick={() => setShowAdvanced(!showAdvanced)} 
                className="text-[10px] text-emerald-400 hover:text-emerald-300 underline"
              >
                {showAdvanced ? 'Collapse' : 'Advanced'}
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div className="space-y-1">
                <div className="flex justify-between text-gray-400">
                  <span>Confidence Threshold:</span>
                  <span className="font-mono text-emerald-400 font-semibold">{config.threshold}</span>
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

              {showAdvanced && (
                <div className="space-y-3 border-t border-[#1f242c] pt-3 animate-fadeIn">
                  <div className="space-y-1">
                    <div className="flex justify-between text-gray-400">
                      <span>Closing Radius (px):</span>
                      <span className="font-mono text-white">{config.closingRadius}</span>
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
                      <span className="font-mono text-white">{config.minObjectSize}</span>
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

                  <div className="space-y-1">
                    <div className="flex justify-between text-gray-400">
                      <span>U-Net Input Size:</span>
                      <span className="font-mono text-white">{config.imageSize}×{config.imageSize}</span>
                    </div>
                    <select
                      value={config.imageSize}
                      onChange={(e) => setConfig({ ...config, imageSize: parseInt(e.target.value) })}
                      className="w-full bg-gray-900 border border-[#1f242c] rounded p-1.5 text-xs text-white outline-none"
                    >
                      <option value="256">256 px</option>
                      <option value="512">512 px</option>
                      <option value="1024">1024 px</option>
                    </select>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="pt-4 border-t border-[#1f242c]">
          <button 
            onClick={triggerAnalysis}
            disabled={isProcessing}
            className="w-full flex items-center justify-center gap-2 py-3 bg-emerald-500 hover:bg-emerald-600 disabled:bg-emerald-800/40 text-[#070a0e] font-bold rounded text-xs tracking-wider uppercase transition-all shadow-md shadow-emerald-500/5 cursor-pointer"
          >
            <Play className="h-4 w-4 fill-current" />
            {isProcessing ? 'Processing...' : 'Run Extraction Pipeline'}
          </button>
        </div>
      </div>

      {/* CENTER PANEL: Image visualization canvas */}
      <div className="flex-1 flex flex-col p-6 space-y-6">
        {errorMsg && (
          <div className="bg-red-500/10 border border-red-500/25 p-4 rounded text-xs text-red-400">
            <strong>Error:</strong> {errorMsg}
          </div>
        )}

        <div className="flex-1 min-h-0 relative">
          {isProcessing && !result ? (
            <div className="h-full flex flex-col items-center justify-center bg-[#0d121a] border border-[#1f242c] rounded p-8 space-y-8">
              <div className="w-64 space-y-2 text-center">
                <div className="flex justify-between items-center text-xs font-mono">
                  <span className="text-gray-500">STAGE PROGRESS:</span>
                  <span className="text-emerald-400 font-semibold">{progressStages[activeStageIdx]?.name}</span>
                </div>
                <div className="w-full bg-gray-800 h-1.5 rounded-full overflow-hidden">
                  <div className="bg-emerald-500 h-full animate-progress" style={{ width: `${((activeStageIdx + 1) / progressStages.length) * 100}%` }} />
                </div>
              </div>

              {/* Progress Steps list */}
              <div className="w-80 space-y-2.5 font-mono text-xs border border-[#1f242c] p-4 rounded bg-gray-950/45">
                {progressStages.map((stage, idx) => (
                  <div key={stage.id} className="flex items-start justify-between">
                    <span className={idx <= activeStageIdx ? 'text-white' : 'text-gray-600'}>
                      {stage.name}
                    </span>
                    <span className={`text-[10px] font-semibold ${
                      stage.status === 'completed' ? 'text-emerald-400' :
                      stage.status === 'processing' ? 'text-amber-400 animate-pulse' :
                      'text-gray-600'
                    }`}>
                      {stage.status === 'completed' ? 'DONE' :
                       stage.status === 'processing' ? 'ACTIVE' : 'QUEUED'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <ImageViewer 
              result={result} 
              selectedIssue={selectedIssue}
              onSelectIssue={setSelectedIssue}
            />
          )}
        </div>
      </div>

      {/* RIGHT PANEL: Extraction Summary & Exports */}
      <div className="w-80 border-l border-[#1f242c] bg-[#0b0f14]/40 flex flex-col justify-between overflow-y-auto shrink-0 p-5 space-y-6">
        {result ? (
          <div className="space-y-6">
            <div className="space-y-3">
              <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider font-mono">Real Analysis Results</label>
              
              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div className="bg-gray-900/60 p-3 border border-[#1f242c] rounded">
                  <div className="text-gray-500 text-[9px] uppercase tracking-wider">Road Length</div>
                  <div className="text-white font-bold text-sm mt-1">
                    {result.networkSummary.totalRoadLength?.value !== undefined ? `${result.networkSummary.totalRoadLength.value} km` : 'N/A'}
                  </div>
                </div>
                <div className="bg-gray-900/60 p-3 border border-[#1f242c] rounded">
                  <div className="text-gray-500 text-[9px] uppercase tracking-wider">Connectivity</div>
                  <div className="text-white font-bold text-sm mt-1">
                    {result.networkSummary.connectivity || 'N/A'}
                  </div>
                </div>
                <div className="bg-gray-900/60 p-3 border border-[#1f242c] rounded">
                  <div className="text-gray-500 text-[9px] uppercase tracking-wider">Segments</div>
                  <div className="text-white font-bold text-sm mt-1">
                    {result.networkSummary.roadSegments?.value || 'N/A'}
                  </div>
                </div>
                <div className="bg-gray-900/60 p-3 border border-[#1f242c] rounded">
                  <div className="text-gray-500 text-[9px] uppercase tracking-wider">Junctions</div>
                  <div className="text-white font-bold text-sm mt-1">
                    {result.topology.intersections !== undefined ? result.topology.intersections : 'N/A'}
                  </div>
                </div>
                <div className="bg-gray-900/60 p-3 border border-[#1f242c] rounded">
                  <div className="text-gray-500 text-[9px] uppercase tracking-wider">Dead Ends</div>
                  <div className="text-white font-bold text-sm mt-1">
                    {result.topology.deadEnds !== undefined ? result.topology.deadEnds : 'N/A'}
                  </div>
                </div>
                <div className="bg-gray-900/60 p-3 border border-[#1f242c] rounded">
                  <div className="text-gray-500 text-[9px] uppercase tracking-wider">Components</div>
                  <div className="text-white font-bold text-sm mt-1">
                    {result.topology.connectedComponents !== undefined ? result.topology.connectedComponents : 'N/A'}
                  </div>
                </div>
              </div>
            </div>

            {/* Topology Warnings layout */}
            <div className="space-y-2 flex-1 min-h-0 flex flex-col">
              <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider font-mono">Topology Warnings ({result.flaggedIssues.length})</label>
              <div className="border border-[#1f242c] rounded bg-gray-900/40 p-2 space-y-2 max-h-48 overflow-y-auto text-xs font-mono">
                {result.flaggedIssues.length > 0 ? (
                  result.flaggedIssues.map((issue) => (
                    <div 
                      key={issue.id}
                      onClick={() => setSelectedIssue(issue)}
                      className={`p-2 border rounded cursor-pointer transition-all ${
                        selectedIssue?.id === issue.id 
                          ? 'bg-amber-500/10 border-amber-500/50 text-white' 
                          : 'border-[#1f242c] text-gray-400 hover:text-white'
                      }`}
                    >
                      <div className="flex justify-between items-center font-semibold">
                        <span className="text-amber-400">{issue.reference}</span>
                        <span className="text-[9px] uppercase text-gray-500">CONF: {issue.confidence}%</span>
                      </div>
                      <div className="text-[10px] mt-1 text-gray-300 leading-normal">{issue.description}</div>
                    </div>
                  ))
                ) : (
                  <div className="p-4 text-center text-gray-500">No anomalies detected.</div>
                )}
              </div>
            </div>

            {/* Pipeline diagnostic downloads */}
            <div className="space-y-2">
              <label className="text-[10px] text-gray-500 font-bold uppercase tracking-wider font-mono">Exports & GIS Outputs</label>
              <div className="space-y-1.5 font-mono text-[10px]">
                <a 
                  href={apiService.getLayerUrl(result.projectId, 'geojson')} 
                  download 
                  className="flex justify-between items-center bg-gray-900/60 hover:bg-gray-800 border border-[#1f242c] rounded px-3 py-2 text-gray-300 hover:text-white transition-colors"
                >
                  <span>Vector road_network.geojson</span>
                  <Download className="h-3.5 w-3.5" />
                </a>
                <a 
                  href={apiService.getLayerUrl(result.projectId, 'graphml')} 
                  download 
                  className="flex justify-between items-center bg-gray-900/60 hover:bg-gray-800 border border-[#1f242c] rounded px-3 py-2 text-gray-300 hover:text-white transition-colors"
                >
                  <span>Network road_graph.graphml</span>
                  <Download className="h-3.5 w-3.5" />
                </a>
                <a 
                  href={apiService.getLayerUrl(result.projectId, 'raw_mask')} 
                  download 
                  className="flex justify-between items-center bg-gray-900/60 hover:bg-gray-800 border border-[#1f242c] rounded px-3 py-2 text-gray-300 hover:text-white transition-colors"
                >
                  <span>Raw Mask (raw_mask.png)</span>
                  <Download className="h-3.5 w-3.5" />
                </a>
                <a 
                  href={apiService.getLayerUrl(result.projectId, 'repaired')} 
                  download 
                  className="flex justify-between items-center bg-gray-900/60 hover:bg-gray-800 border border-[#1f242c] rounded px-3 py-2 text-gray-300 hover:text-white transition-colors"
                >
                  <span>Repaired (repaired_mask.png)</span>
                  <Download className="h-3.5 w-3.5" />
                </a>
                <a 
                  href={apiService.getLayerUrl(result.projectId, 'skeleton')} 
                  download 
                  className="flex justify-between items-center bg-gray-900/60 hover:bg-gray-800 border border-[#1f242c] rounded px-3 py-2 text-gray-300 hover:text-white transition-colors"
                >
                  <span>Skeleton (skeleton.png)</span>
                  <Download className="h-3.5 w-3.5" />
                </a>
                <a 
                  href={apiService.getLayerUrl(result.projectId, 'diagnostic')} 
                  download 
                  className="flex justify-between items-center bg-gray-900/60 hover:bg-gray-800 border border-[#1f242c] rounded px-3 py-2 text-gray-300 hover:text-white transition-colors"
                >
                  <span>Diagnostic Summary Image</span>
                  <Download className="h-3.5 w-3.5" />
                </a>
              </div>
            </div>
          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-center text-xs text-gray-500 p-4 border border-dashed border-[#1f242c] rounded-lg bg-gray-900/10">
            <span>Configure settings and run road extraction to visualize network metrics and export GIS files.</span>
          </div>
        )}
      </div>
    </div>
  );
}
