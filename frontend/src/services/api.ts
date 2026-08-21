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
