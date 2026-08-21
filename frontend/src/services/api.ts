import { ExtractionConfig, AnalysisResult } from '../types';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

export const apiService = {
  /**
   * Upload satellite imagery and trigger the backend road extraction pipeline.
   */
  async submitAnalysis(
    file: File,
    config: ExtractionConfig,
    jobId: string,
    projectName: string,
    studyArea: string,
    imageYear: string
  ): Promise<AnalysisResult> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('jobId', jobId);
    formData.append('name', projectName);
    formData.append('studyArea', studyArea);
    formData.append('imageYear', imageYear);
    formData.append('threshold', String(config.threshold));
    formData.append('closing_radius', String(config.closingRadius));
    formData.append('min_object_size', String(config.minObjectSize));
    formData.append('use_model', String(config.useModel));

    const response = await fetch(`${API_BASE}/api/process`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      let parsedError = 'Inference pipeline processing failed.';
      try {
        const errJson = JSON.parse(errorText);
        parsedError = errJson.error || parsedError;
      } catch {
        parsedError = errorText || parsedError;
      }
      throw new Error(parsedError);
    }

    return response.json();
  },

  /**
   * Fetch previously computed road network extraction results.
   */
  async getAnalysis(jobId: string): Promise<AnalysisResult> {
    const response = await fetch(`${API_BASE}/api/analysis/${jobId}`);
    if (!response.ok) {
      if (response.status === 404) {
        throw new Error(`Analysis results for run ID "${jobId}" could not be found.`);
      }
      const errText = await response.text();
      throw new Error(errText || 'Failed to retrieve analysis workspace results.');
    }
    return response.json();
  },

  /**
   * Submit two satellite images (or run IDs) to perform a spatial comparison pipeline.
   */
  async runComparison(
    fileA: File | null,
    fileB: File | null,
    jobIdA: string,
    jobIdB: string,
    yearA: string,
    yearB: string,
    name: string,
    studyArea: string,
    config: ExtractionConfig
  ): Promise<any> {
    const formData = new FormData();
    if (fileA) formData.append('file_a', fileA);
    if (fileB) formData.append('file_b', fileB);
    formData.append('job_id_a', jobIdA);
    formData.append('job_id_b', jobIdB);
    formData.append('year_a', yearA);
    formData.append('year_b', yearB);
    formData.append('name', name);
    formData.append('studyArea', studyArea);
    formData.append('threshold', String(config.threshold));
    formData.append('closing_radius', String(config.closingRadius));
    formData.append('min_object_size', String(config.minObjectSize));
    formData.append('use_model', String(config.useModel));

    const response = await fetch(`${API_BASE}/api/compare`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      let parsedError = 'Spatial comparison failed.';
      try {
        const errJson = JSON.parse(errorText);
        parsedError = errJson.error || parsedError;
      } catch {
        parsedError = errorText || parsedError;
      }
      throw new Error(parsedError);
    }

    return response.json();
  },

  /**
   * Fetch all persistent runs from the backend.
   */
  async getRuns(includeArchived: boolean = false): Promise<any[]> {
    const response = await fetch(`${API_BASE}/api/runs?include_archived=${includeArchived}`);
    if (!response.ok) {
      throw new Error('Failed to retrieve run history.');
    }
    return response.json();
  },

  /**
   * Delete an analysis run from the backend (permanently).
   */
  async deleteRun(jobId: string): Promise<any> {
    const response = await fetch(`${API_BASE}/api/analysis/${jobId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      const errText = await response.text();
      throw new Error(errText || 'Failed to delete the analysis run.');
    }
    return response.json();
  },

  /**
   * Archive an analysis run rather than permanently deleting it.
   */
  async archiveRun(jobId: string): Promise<any> {
    const response = await fetch(`${API_BASE}/api/analysis/${jobId}/archive`, {
      method: 'POST',
    });
    if (!response.ok) {
      const errText = await response.text();
      throw new Error(errText || 'Failed to archive the analysis run.');
    }
    return response.json();
  },

  /**
   * Run the unmapped route discovery pipeline.
   */
  async runDiscovery(
    file: File,
    refFile: File | null,
    refJobId: string,
    tolerance: number,
    minLength: number,
    name: string,
    studyArea: string,
    year: string,
    config: ExtractionConfig
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);
    if (refFile) formData.append('ref_file', refFile);
    formData.append('ref_job_id', refJobId);
    formData.append('tolerance', String(tolerance));
    formData.append('min_length', String(minLength));
    formData.append('name', name);
    formData.append('studyArea', studyArea);
    formData.append('year', year);
    formData.append('threshold', String(config.threshold));
    formData.append('closing_radius', String(config.closingRadius));
    formData.append('min_object_size', String(config.minObjectSize));
    formData.append('use_model', String(config.useModel));

    const response = await fetch(`${API_BASE}/api/discovery`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      let parsedError = 'Route discovery failed.';
      try {
        const errJson = JSON.parse(errorText);
        parsedError = errJson.error || parsedError;
      } catch {
        parsedError = errorText || parsedError;
      }
      throw new Error(parsedError);
    }

    return response.json();
  },

  /**
   * Run the Point-to-Point route discovery.
   */
  async runPointToPoint(
    file: File | null,
    refFile: File | null,
    refJobId: string,
    jobId: string,
    startX: number,
    startY: number,
    endX: number,
    endY: number,
    tolerance: number,
    avoidanceWeight: number,
    allowExistingRoads: boolean,
    maxRoutes: number,
    name: string,
    studyArea: string,
    year: string,
    config: ExtractionConfig,
    minSeparation?: number,
    minDiversity?: number
  ): Promise<any> {
    const formData = new FormData();
    if (file) formData.append('file', file);
    if (refFile) formData.append('ref_file', refFile);
    formData.append('ref_job_id', refJobId);
    formData.append('job_id', jobId);
    formData.append('start_x', String(startX));
    formData.append('start_y', String(startY));
    formData.append('end_x', String(endX));
    formData.append('end_y', String(endY));
    formData.append('tolerance', String(tolerance));
    formData.append('avoidance_weight', String(avoidanceWeight));
    formData.append('allow_existing_roads', String(allowExistingRoads));
    formData.append('max_routes', String(maxRoutes));
    formData.append('name', name);
    formData.append('studyArea', studyArea);
    formData.append('year', year);
    formData.append('threshold', String(config.threshold));
    formData.append('closing_radius', String(config.closingRadius));
    formData.append('min_object_size', String(config.minObjectSize));
    formData.append('use_model', String(config.useModel));
    if (minSeparation !== undefined) formData.append('min_separation', String(minSeparation));
    if (minDiversity !== undefined) formData.append('min_diversity', String(minDiversity));

    const response = await fetch(`${API_BASE}/api/route-discovery/point-to-point`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      let parsedError = 'Point-to-point route discovery failed.';
      try {
        const errJson = JSON.parse(errorText);
        parsedError = errJson.error || parsedError;
      } catch {
        parsedError = errorText || parsedError;
      }
      throw new Error(parsedError);
    }

    return response.json();
  },

  /**
   * Run the classification telemetry analysis on a completed job's vectors.
   */
  async runClassification(
    jobId: string,
    arterialWidth?: number,
    collectorWidth?: number,
    localWidth?: number,
    arterialCurvature?: number,
    roughnessThreshold?: number
  ): Promise<any> {
    const formData = new FormData();
    formData.append('job_id', jobId);
    if (arterialWidth !== undefined) formData.append('arterial_width', String(arterialWidth));
    if (collectorWidth !== undefined) formData.append('collector_width', String(collectorWidth));
    if (localWidth !== undefined) formData.append('local_width', String(localWidth));
    if (arterialCurvature !== undefined) formData.append('arterial_curvature', String(arterialCurvature));
    if (roughnessThreshold !== undefined) formData.append('roughness_threshold', String(roughnessThreshold));

    const response = await fetch(`${API_BASE}/api/classification`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      let parsedError = 'Classification analysis failed.';
      try {
        const errJson = JSON.parse(errorText);
        parsedError = errJson.error || parsedError;
      } catch {
        parsedError = errorText || parsedError;
      }
      throw new Error(parsedError);
    }

    return response.json();
  },

  /**
   * Fetch graph network intelligence diagnostics from the backend.
   */
  async getIntelligence(jobId: string): Promise<any> {
    const response = await fetch(`${API_BASE}/api/intelligence/${jobId}`);
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || 'Failed to retrieve network intelligence.');
    }
    return response.json();
  },

  /**
   * Request server-side generation of the executive summary PDF report.
   */
  async generateReport(jobId: string): Promise<{ success: boolean; reportUrl: string }> {
    const response = await fetch(`${API_BASE}/api/analysis/${jobId}/report`, {
      method: 'POST',
    });
    if (!response.ok) {
      const errText = await response.text();
      throw new Error(errText || 'Failed to generate PDF report.');
    }
    return response.json();
  },

  /**
   * Poll active tiled prediction progress from backend.
   */
  async getProgress(jobId: string): Promise<{ success: boolean; progress: string }> {
    const response = await fetch(`${API_BASE}/api/analysis/${jobId}/progress`);
    if (!response.ok) {
      throw new Error('Failed to retrieve progress.');
    }
    return response.json();
  },

  /**
   * Helper to construct static URLs to generated layers
   */
  getLayerUrl(jobId: string, layerType: 'original' | 'raw_mask' | 'repaired' | 'skeleton' | 'overlay' | 'diagnostic' | 'graph' | 'geojson' | 'graphml'): string {
    const filenames = {
      original: 'original_rgb.png',
      raw_mask: 'raw_mask.png',
      repaired: 'geometry_clean_mask.png',
      skeleton: 'geometry_skeleton.png',
      overlay: 'overlay.png',
      diagnostic: 'geometry_diagnostic.png',
      graph: 'road_graph.png',
      geojson: 'road_network.geojson',
      graphml: 'road_network.graphml',
    };
    return `${API_BASE}/static/outputs/${jobId}/${filenames[layerType]}`;
  }
};
