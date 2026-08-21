export interface ExtractionConfig {
  threshold: number;
  closingRadius: number;
  minObjectSize: number;
  bridgeKernelSize: number;
  imageSize: number;
  useModel: boolean;
}

export interface NetworkMetrics {
  totalRoadLength?: {
    value: number;
    unit: string;
  };
  roadSegments?: {
    value: number;
    unit: string;
  };
  intersections?: {
    value: number;
    unit: string;
  };
  connectedComponents?: {
    value: number;
    unit: string;
  };
  avgSegmentLength?: {
    value: number;
    unit: string;
  };
  connectivity?: string;
  overallConfidence?: string;
}

export interface PipelineStage {
  id: string;
  name: string;
  status: 'idle' | 'processing' | 'completed' | 'failed';
  label: string;
}

export interface FlaggedIssue {
  id: string;
  description: string;
  reference: string;
  category: string;
  coords: { x: number; y: number };
  latLng: string;
  confidence: number;
}

export interface ReviewItem {
  id: string;
  description: string;
  reference: string;
  category: string;
  coords: { x: number; y: number };
  latLng: string;
  confidence: number;
  reviewed?: boolean;
}

export interface AnalysisResult {
  projectId: string;
  projectName: string;
  location: string;
  analysisDate: string;
  status: string;
  imageYear: string;
  resolution: string;
  fileName: string;
  fileSize: string;
  networkSummary: NetworkMetrics;
  geometry: {
    totalRoadLength: string;
    avgSegmentLength: string;
    geometryIssues: number;
    roadContinuity: string;
  };
  topology: {
    intersections: number;
    deadEnds: number;
    connectedComponents: number;
    disconnectedSegments: number;
    topologyIssues: number;
  };
  healthMetrics: {
    [key: string]: {
      value: number;
      label: string;
      status: string;
    };
  };
  confidenceBreakdown: {
    high: number;
    mid: number;
    low: number;
  };
  flaggedIssues: FlaggedIssue[];
  reviewItems?: ReviewItem[];
}

export interface PotentialRoute {
  id: string | number;
  lengthPixels: number;
  lengthMeters: number;
  confidence: number;
  status: 'Potential New Route' | 'Potentially Unmapped Route';
  connectionInfo: string;
  sourceNode?: number;
  targetNode?: number;
  coordinates?: number[][];
}
