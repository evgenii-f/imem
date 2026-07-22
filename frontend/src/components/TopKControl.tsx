import { NumberInput } from '@mantine/core';

interface Props {
  value: number;
  onChange: (value: number) => void;
}

/** How many results to request (top-k). */
export function TopKControl({ value, onChange }: Props) {
  return (
    <NumberInput
      label="Results (top-k)"
      value={value}
      onChange={(v) => onChange(typeof v === 'number' ? v : 1)}
      min={1}
      max={100}
      clampBehavior="strict"
      w={140}
    />
  );
}
