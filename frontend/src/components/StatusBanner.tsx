import { Alert, Badge, Group, Loader, Text } from '@mantine/core';
import type { Health } from '../types';

interface Props {
  health: Health | null;
  error: string | null;
}

/** Compact readiness indicator: shows model/Qdrant status from /health. */
export function StatusBanner({ health, error }: Props) {
  if (error) {
    return (
      <Alert color="red" title="Backend unreachable">
        {error} — is <Text span ff="monospace">imem serve</Text> running?
      </Alert>
    );
  }

  if (!health) {
    return (
      <Group gap="xs">
        <Loader size="xs" />
        <Text size="sm" c="dimmed">
          Connecting…
        </Text>
      </Group>
    );
  }

  const ok = health.model_loaded && health.qdrant_reachable;
  return (
    <Group gap="xs">
      <Badge color={ok ? 'teal' : 'yellow'} variant="light">
        {ok ? 'Ready' : 'Degraded'}
      </Badge>
      {!health.model_loaded && (
        <Text size="sm" c="dimmed">
          model loading…
        </Text>
      )}
      {!health.qdrant_reachable && (
        <Text size="sm" c="dimmed">
          Qdrant unreachable
        </Text>
      )}
    </Group>
  );
}
