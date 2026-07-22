// Mirrors of the backend response schemas (imem/api/schemas.py).

export interface AppConfig {
  default_top_k: number;
  default_collection: string;
  extensions: string[];
  model_id: string;
  distance: string;
}

export interface CollectionInfo {
  name: string;
  count: number;
}

export interface CollectionsResponse {
  collections: CollectionInfo[];
}

export interface Health {
  status: string;
  model_loaded: boolean;
  qdrant_reachable: boolean;
}

export type QueryMode = 'auto' | 'text' | 'image';

export interface ResultItem {
  path: string;
  score: number;
}

export interface QueryResponse {
  results: ResultItem[];
  mode: 'text' | 'image';
}
