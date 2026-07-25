import type {
  AppConfig,
  CollectionsResponse,
  Health,
  QueryMode,
  QueryResponse,
} from './types';

const BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

/** Extract a human-readable message from a failed response (FastAPI puts it in `detail`). */
async function errorMessage(res: Response): Promise<string> {
  try {
    const data = await res.json();
    if (data && typeof data.detail === 'string') return data.detail;
  } catch {
    /* fall through */
  }
  return `${res.status} ${res.statusText}`;
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(await errorMessage(res));
  return (await res.json()) as T;
}

export const api = {
  base: BASE,

  getConfig: () => getJSON<AppConfig>('/config'),
  getCollections: () => getJSON<CollectionsResponse>('/collections'),
  getHealth: () => getJSON<Health>('/health'),

  async queryText(
    collection: string,
    query: string,
    mode: QueryMode,
    topK: number,
  ): Promise<QueryResponse> {
    const res = await fetch(`${BASE}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ collection, query, mode, top_k: topK }),
    });
    if (!res.ok) throw new Error(await errorMessage(res));
    return (await res.json()) as QueryResponse;
  },

  async queryImage(collection: string, file: File, topK: number): Promise<QueryResponse> {
    const form = new FormData();
    form.append('file', file);
    form.append('collection', collection);
    form.append('top_k', String(topK));
    const res = await fetch(`${BASE}/query/upload`, { method: 'POST', body: form });
    if (!res.ok) throw new Error(await errorMessage(res));
    return (await res.json()) as QueryResponse;
  },

  /** URL for the /image endpoint — used for thumbnails, open-in-tab, and download. */
  imageUrl(path: string, opts?: { thumb?: number; download?: boolean }): string {
    const p = new URLSearchParams({ path });
    if (opts?.thumb) p.set('thumb', String(opts.thumb));
    if (opts?.download) p.set('download', 'true');
    return `${BASE}/image?${p.toString()}`;
  },
};
