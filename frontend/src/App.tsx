import { useEffect, useState } from 'react';
import { Alert, Container, Divider, Group, Stack, Title } from '@mantine/core';
import { api } from './api';
import type { CollectionInfo, Health, ResultItem } from './types';
import { StatusBanner } from './components/StatusBanner';
import { CollectionSelect } from './components/CollectionSelect';
import { TopKControl } from './components/TopKControl';
import { TextQuery } from './components/TextQuery';
import { ImageQuery } from './components/ImageQuery';
import { ResultsGrid } from './components/ResultsGrid';

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [connError, setConnError] = useState<string | null>(null);

  const [collections, setCollections] = useState<CollectionInfo[]>([]);
  const [collection, setCollection] = useState<string | null>(null);
  const [topK, setTopK] = useState<number>(5);

  const [results, setResults] = useState<ResultItem[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [queryError, setQueryError] = useState<string | null>(null);

  // Initial load: config (defaults) + collections + health.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [cfg, cols, hp] = await Promise.all([
          api.getConfig(),
          api.getCollections(),
          api.getHealth(),
        ]);
        if (cancelled) return;
        setHealth(hp);
        setTopK(cfg.default_top_k);
        setCollections(cols.collections);
        // Prefer the configured default collection if present and non-empty,
        // else the first non-empty one, else whatever exists.
        const preferred =
          cols.collections.find((c) => c.name === cfg.default_collection && c.count > 0) ??
          cols.collections.find((c) => c.count > 0) ??
          cols.collections[0];
        setCollection(preferred ? preferred.name : null);
      } catch (e) {
        if (!cancelled) setConnError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  async function run(work: () => Promise<{ results: ResultItem[] }>) {
    setLoading(true);
    setQueryError(null);
    try {
      const res = await work();
      setResults(res.results);
    } catch (e) {
      setQueryError(e instanceof Error ? e.message : String(e));
      setResults(null);
    } finally {
      setLoading(false);
    }
  }

  const onText = (query: string) => {
    if (collection) run(() => api.queryText(collection, query, 'text', topK));
  };
  const onImage = (file: File) => {
    if (collection) run(() => api.queryImage(collection, file, topK));
  };

  const ready = !!collection;

  return (
    <Container size="lg" py="lg">
      <Stack gap="lg">
        <Group justify="space-between" align="center">
          <Title order={1}>iMem</Title>
          <StatusBanner health={health} error={connError} />
        </Group>

        <Group align="flex-end" gap="md">
          <CollectionSelect
            collections={collections}
            value={collection}
            onChange={setCollection}
          />
          <TopKControl value={topK} onChange={setTopK} />
        </Group>

        <Group align="stretch" gap="md" grow wrap="wrap">
          <TextQuery onSubmit={onText} disabled={!ready} loading={loading} />
          <ImageQuery onSubmit={onImage} disabled={!ready} loading={loading} />
        </Group>

        {queryError && (
          <Alert
            color="red"
            title="Query failed"
            withCloseButton
            onClose={() => setQueryError(null)}
          >
            {queryError}
          </Alert>
        )}

        <Divider label="Results" labelPosition="left" />
        <ResultsGrid results={results} loading={loading} />
      </Stack>
    </Container>
  );
}
