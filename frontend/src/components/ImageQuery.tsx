import { useEffect, useRef, useState } from 'react';
import { Button, Group, Image, Paper, Stack, Text } from '@mantine/core';
import { Dropzone, IMAGE_MIME_TYPE } from '@mantine/dropzone';
import type { DropzoneProps } from '@mantine/dropzone';
import { IconPhoto, IconUpload, IconX } from '@tabler/icons-react';

interface Props {
  onSubmit: (file: File) => void;
  disabled: boolean;
  loading: boolean;
}

/**
 * Query-by-image input: a dropzone that also accepts a clipboard paste
 * (Ctrl/Cmd+V) and a button to pick a file from the host. Any of the three
 * yields a File, which is previewed and sent to /query/upload.
 */
export function ImageQuery({ onSubmit, disabled, loading }: Props) {
  const openRef = useRef<() => void>(null);
  const [preview, setPreview] = useState<string | null>(null);

  const accept = (file: File) => {
    setPreview((prev) => {
      if (prev) URL.revokeObjectURL(prev);
      return URL.createObjectURL(file);
    });
    onSubmit(file);
  };

  const onDrop: DropzoneProps['onDrop'] = (files) => {
    if (files.length > 0) accept(files[0]);
  };

  // Paste an image from the clipboard anywhere on the page.
  useEffect(() => {
    if (disabled) return;
    const onPaste = (e: ClipboardEvent) => {
      const file = Array.from(e.clipboardData?.files ?? []).find((f) =>
        f.type.startsWith('image/'),
      );
      if (file) accept(file);
    };
    window.addEventListener('paste', onPaste);
    return () => window.removeEventListener('paste', onPaste);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [disabled]);

  // Revoke the preview URL on unmount.
  useEffect(() => () => {
    if (preview) URL.revokeObjectURL(preview);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <Paper withBorder p="md" radius="md" flex={1}>
      <Stack gap="xs">
        <Group justify="space-between">
          <Text fw={500}>Search by image</Text>
          <Button
            variant="light"
            size="xs"
            onClick={() => openRef.current?.()}
            disabled={disabled}
          >
            Choose file…
          </Button>
        </Group>

        <Dropzone
          openRef={openRef}
          onDrop={onDrop}
          accept={IMAGE_MIME_TYPE}
          disabled={disabled}
          loading={loading}
          multiple={false}
        >
          <Group justify="center" gap="md" mih={100} style={{ pointerEvents: 'none' }}>
            <Dropzone.Accept>
              <IconUpload size={36} />
            </Dropzone.Accept>
            <Dropzone.Reject>
              <IconX size={36} />
            </Dropzone.Reject>
            <Dropzone.Idle>
              {preview ? (
                <Image src={preview} h={90} w="auto" radius="sm" fit="contain" />
              ) : (
                <IconPhoto size={36} />
              )}
            </Dropzone.Idle>
            {!preview && (
              <Text size="sm" c="dimmed">
                Drop, paste, or choose an image
              </Text>
            )}
          </Group>
        </Dropzone>
      </Stack>
    </Paper>
  );
}
