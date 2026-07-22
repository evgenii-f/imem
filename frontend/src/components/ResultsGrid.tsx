import { Center, Loader, SimpleGrid, Skeleton, Text } from '@mantine/core';
import type { ResultItem } from '../types';
import { ResultCard } from './ResultCard';

interface Props {
  results: ResultItem[] | null;
  loading: boolean;
}

/** Grid of result cards, with loading skeletons and an empty state. */
export function ResultsGrid({ results, loading }: Props) {
  if (loading) {
    return (
      <SimpleGrid cols={{ base: 2, sm: 3, md: 4, lg: 5 }} spacing="md">
        {Array.from({ length: 10 }).map((_, i) => (
          <Skeleton key={i} height={230} radius="md" />
        ))}
      </SimpleGrid>
    );
  }

  if (results === null) {
    return (
      <Center mih={120}>
        <Text c="dimmed">Run a text or image query to see matches.</Text>
      </Center>
    );
  }

  if (results.length === 0) {
    return (
      <Center mih={120}>
        <Text c="dimmed">No matches found.</Text>
      </Center>
    );
  }

  return (
    <SimpleGrid cols={{ base: 2, sm: 3, md: 4, lg: 5 }} spacing="md">
      {results.map((item) => (
        <ResultCard key={item.path} item={item} />
      ))}
    </SimpleGrid>
  );
}

/** Small inline spinner used while the model is still loading, etc. */
export function InlineLoader() {
  return (
    <Center mih={120}>
      <Loader />
    </Center>
  );
}
