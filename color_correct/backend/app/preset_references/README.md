# Preset Reference Images

This directory contains preset reference image collections. Each subdirectory represents a preset that can be loaded in the color correction tool.

## Structure

```
preset_references/
├── preset_name_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── preset_name_2/
│   ├── image1.jpg
│   └── ...
└── ...
```

## Usage

1. Create a new subdirectory for your preset (e.g., `warm_tones`, `cool_tones`, `neutral`)
2. Add reference images to that directory
3. The preset will automatically appear in the "Load Preset" dropdown in the Reference Panel
4. Clicking on a preset will load all images from that directory into the current session's reference images

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- BMP (.bmp)
- WebP (.webp)

## Example

To create a preset called "warm_tones":
1. Create directory: `preset_references/warm_tones/`
2. Add your reference images to that directory
3. The preset will be available in the UI as "warm_tones"

