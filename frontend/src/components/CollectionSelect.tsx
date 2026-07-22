import { Select } from '@mantine/core';
import type { CollectionInfo } from '../types';

interface Props {
  collections: CollectionInfo[];
  value: string | null;
  onChange: (value: string) => void;
}

/** Dropdown of collections, labelled with image counts. */
export function CollectionSelect({ collections, value, onChange }: Props) {
  const data = collections.map((c) => ({
    value: c.name,
    label: `${c.name} (${c.count})`,
  }));

  return (
    <Select
      label="Collection"
      placeholder={collections.length ? 'Pick a collection' : 'No collections'}
      data={data}
      value={value}
      onChange={(v) => v && onChange(v)}
      disabled={collections.length === 0}
      searchable
      allowDeselect={false}
      w={280}
    />
  );
}
