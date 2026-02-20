# koten-layout-detector

Japanese classical document layout analysis library using ONNX Runtime for detecting text regions, illustrations, and stamps in historical Japanese documents.

## Features

- 🏯 Specialized for Japanese classical documents (古典籍)
- 🚀 Browser-based inference using ONNX Runtime Web
- 📦 Lightweight and easy to integrate
- 🎯 Detects 5 types of regions:
  - Overall layout (全体)
  - Handwritten text (手書き)
  - Typographic text (活字)
  - Illustrations (図版)
  - Stamps/Seals (印判)

## See It In Action

Check out the live demo at [https://koten-layout.netlify.app/](https://koten-layout.netlify.app/)

## Installation

```bash
npm install koten-layout-detector onnxruntime-web
```

## Usage

```javascript
import {
  loadModel,
  preprocess,
  runInference,
  postprocess,
  drawDetections,
  CLASSES,
  COLORS
} from 'koten-layout-detector'

// Load the ONNX model
const session = await loadModel('/path/to/your/model.onnx')

// Load an image
const img = new Image()
img.src = '/path/to/classical-document.jpg'
await img.decode()

// Preprocess the image
const { tensor, meta } = preprocess(img)

// Run inference
const outputTensor = await runInference(session, tensor)

// Postprocess results
const detections = postprocess(outputTensor, meta, 0.5, 0.45)

// Draw detections on canvas
const canvas = document.getElementById('output-canvas')
drawDetections(canvas, img, detections)

console.log('Detected regions:', detections)
```

## API Reference

### `loadModel(modelUrl: string): Promise<InferenceSession>`

Loads an ONNX model from the specified URL.

### `preprocess(img: HTMLImageElement): { tensor: Tensor, meta: Object }`

Preprocesses an image for inference with letterbox resizing.

Returns:
- `tensor`: ONNX tensor ready for inference
- `meta`: Metadata for postprocessing (scale, padding, original dimensions)

### `runInference(session: InferenceSession, tensor: Tensor): Promise<Tensor>`

Runs inference on the preprocessed tensor.

### `postprocess(outputTensor: Tensor, meta: Object, confThreshold?: number, iouThreshold?: number): Array<Detection>`

Postprocesses the model output into detection results.

Parameters:
- `confThreshold`: Confidence threshold (default: 0.5)
- `iouThreshold`: IoU threshold for NMS (default: 0.45)

Returns an array of detections with:
- `x1, y1, x2, y2`: Bounding box coordinates
- `conf`: Confidence score
- `classId`: Class ID
- `label`: Japanese label
- `color`: Color for visualization

### `drawDetections(canvas: HTMLCanvasElement, img: HTMLImageElement, detections: Array<Detection>): void`

Draws the original image and detection boxes on a canvas.

### `CLASSES`

Array of class definitions with ID, key, and Japanese labels.

### `COLORS`

Array of colors for each class for visualization.

## Dataset

This model is trained on the [NDL-DocL Layout Dataset](https://github.com/ndl-lab/layout-dataset) provided by the National Diet Library of Japan. The dataset contains annotated layout information for Japanese classical documents.

## Model

The detection model is based on YOLOv12, optimized for classical Japanese document analysis.

**Note:** You need to provide your own trained ONNX model file. This library provides the inference pipeline but does not include the model weights.

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [NDL-DocL Layout Dataset](https://github.com/ndl-lab/layout-dataset) - National Diet Library of Japan
- ONNX Runtime Web team for the excellent inference engine
