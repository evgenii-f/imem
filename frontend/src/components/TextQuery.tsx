import { useState } from 'react';
import { Button, Group, Paper, Stack, Text, TextInput } from '@mantine/core';
import { IconSearch } from '@tabler/icons-react';

interface Props {
  onSubmit: (query: string) => void;
  disabled: boolean;
  loading: boolean;
}

/** Free-text query box. Submits on Enter or the button. */
export function TextQuery({ onSubmit, disabled, loading }: Props) {
  const [text, setText] = useState('');
  const canSubmit = !disabled && text.trim().length > 0;

  const submit = () => {
    if (canSubmit) onSubmit(text.trim());
  };

  return (
    <Paper withBorder p="md" radius="md" flex={1}>
      <Stack gap="xs">
        <Text fw={500}>Search by text</Text>
        <Group gap="xs" wrap="nowrap">
          <TextInput
            flex={1}
            placeholder="e.g. red cat on a sofa"
            value={text}
            onChange={(e) => setText(e.currentTarget.value)}
            onKeyDown={(e) => e.key === 'Enter' && submit()}
            disabled={disabled}
          />
          <Button
            onClick={submit}
            disabled={!canSubmit}
            loading={loading}
            leftSection={<IconSearch size={16} />}
          >
            Search
          </Button>
        </Group>
      </Stack>
    </Paper>
  );
}
