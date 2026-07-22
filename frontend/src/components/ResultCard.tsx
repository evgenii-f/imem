import { ActionIcon, Badge, Card, Group, Image, Text, Tooltip } from '@mantine/core';
import { IconCopy, IconDownload, IconExternalLink } from '@tabler/icons-react';
import { api } from '../api';
import type { ResultItem } from '../types';

interface Props {
  item: ResultItem;
}

/** One search hit: thumbnail, score, filename, and open/download/copy actions. */
export function ResultCard({ item }: Props) {
  const filename = item.path.split('/').pop() || item.path;

  return (
    <Card withBorder padding="xs" radius="md">
      <Card.Section>
        <Image
          src={api.imageUrl(item.path, { thumb: 320 })}
          h={160}
          fit="contain"
          bg="var(--mantine-color-default-hover)"
          alt={filename}
        />
      </Card.Section>

      <Group justify="space-between" mt="xs" gap="xs" wrap="nowrap">
        <Tooltip label={item.path} multiline maw={420} openDelay={400}>
          <Text size="sm" truncate="end" flex={1}>
            {filename}
          </Text>
        </Tooltip>
        <Badge variant="light" color="grape">
          {item.score.toFixed(3)}
        </Badge>
      </Group>

      <Group gap={4} mt="xs">
        <Tooltip label="Open in new tab">
          <ActionIcon
            variant="subtle"
            component="a"
            href={api.imageUrl(item.path)}
            target="_blank"
            rel="noreferrer"
          >
            <IconExternalLink size={18} />
          </ActionIcon>
        </Tooltip>
        <Tooltip label="Download">
          <ActionIcon
            variant="subtle"
            component="a"
            href={api.imageUrl(item.path, { download: true })}
          >
            <IconDownload size={18} />
          </ActionIcon>
        </Tooltip>
        <Tooltip label="Copy path">
          <ActionIcon
            variant="subtle"
            onClick={() => navigator.clipboard.writeText(item.path)}
          >
            <IconCopy size={18} />
          </ActionIcon>
        </Tooltip>
      </Group>
    </Card>
  );
}
