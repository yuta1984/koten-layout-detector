# koten-layout-detector

[English](#english) | [日本語](#日本語)

---

## 日本語

日本語古典籍資料のレイアウト解析ライブラリです。ONNX Runtimeを使用し、ブラウザ上で文書画像から本文領域、図版、印判などを検出します。

### 特徴

- 🏯 日本語古典籍に特化したレイアウト解析
- 🚀 ONNX Runtime Webによるブラウザ上での推論
- 📦 軽量で統合が容易（約6KB）
- 📘 TypeScript完全対応
- 🎯 5種類の領域を検出：
  - 全体（1_overall）
  - 手書き（2_handwritten）
  - 活字（3_typography）
  - 図版（4_illustration）
  - 印判（5_stamp）

### デモ

実際の動作は[https://koten-layout.netlify.app/](https://koten-layout.netlify.app/)でご確認いただけます。

### インストール

```bash
npm install koten-layout-detector onnxruntime-web
```

### モデルのダウンロード

事前学習済みONNXモデルはGitHub Releases経由で利用可能です：

```
https://github.com/yuta1984/koten-layout-detector/releases/download/v1.1.0/best.onnx
```

最新バージョンを使用する場合：

```
https://github.com/yuta1984/koten-layout-detector/releases/latest/download/best.onnx
```

モデルサイズ：約36MB

### 使い方

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

// GitHub ReleasesからONNXモデルをロード
const MODEL_URL = 'https://github.com/yuta1984/koten-layout-detector/releases/download/v1.1.0/best.onnx'
const session = await loadModel(MODEL_URL)

// 画像を読み込む
const img = new Image()
img.src = '/path/to/classical-document.jpg'
await img.decode()

// 前処理
const { tensor, meta } = preprocess(img)

// 推論実行
const outputTensor = await runInference(session, tensor)

// 後処理
const detections = postprocess(outputTensor, meta, 0.5, 0.45)

// Canvasに検出結果を描画
const canvas = document.getElementById('output-canvas')
drawDetections(canvas, img, detections)

console.log('検出された領域:', detections)
```

#### TypeScript

TypeScriptで使用する場合、完全な型定義が利用できます：

```typescript
import {
  loadModel,
  preprocess,
  runInference,
  postprocess,
  drawDetections,
  type Detection,
  type PreprocessResult,
  type ClassDefinition
} from 'koten-layout-detector'
import type { InferenceSession } from 'onnxruntime-web'

const MODEL_URL = 'https://github.com/yuta1984/koten-layout-detector/releases/download/v1.1.0/best.onnx'

// 型安全な推論
const session: InferenceSession = await loadModel(MODEL_URL)

const img = new Image()
img.src = '/path/to/classical-document.jpg'
await img.decode()

const { tensor, meta }: PreprocessResult = preprocess(img)
const outputTensor = await runInference(session, tensor)
const detections: Detection[] = postprocess(outputTensor, meta, 0.5, 0.45)

// 型チェックされた検出結果
detections.forEach((det: Detection) => {
  console.log(`検出: ${det.label} (信頼度: ${(det.conf * 100).toFixed(1)}%)`)
  console.log(`位置: (${det.x1}, ${det.y1}) - (${det.x2}, ${det.y2})`)
})
```

### API リファレンス

##### `loadModel(modelUrl: string): Promise<InferenceSession>`

指定されたURLからONNXモデルをロードします。

##### `preprocess(img: HTMLImageElement): { tensor: Tensor, meta: Object }`

画像を推論用に前処理します（レターボックスリサイズ）。

戻り値：
- `tensor`: 推論用のONNXテンソル
- `meta`: 後処理用のメタデータ（スケール、パディング、元画像の寸法）

##### `runInference(session: InferenceSession, tensor: Tensor): Promise<Tensor>`

前処理済みのテンソルで推論を実行します。

##### `postprocess(outputTensor: Tensor, meta: Object, confThreshold?: number, iouThreshold?: number): Array<Detection>`

モデルの出力を検出結果に変換します。

パラメータ：
- `confThreshold`: 信頼度閾値（デフォルト：0.5）
- `iouThreshold`: NMSのIoU閾値（デフォルト：0.45）

戻り値は以下を含む検出結果の配列：
- `x1, y1, x2, y2`: バウンディングボックスの座標
- `conf`: 信頼度スコア
- `classId`: クラスID
- `label`: 日本語ラベル
- `color`: 可視化用の色

##### `drawDetections(canvas: HTMLCanvasElement, img: HTMLImageElement, detections: Array<Detection>): void`

元画像と検出ボックスをCanvasに描画します。

##### `CLASSES`

ID、キー、日本語ラベルを含むクラス定義の配列。

##### `COLORS`

各クラスの可視化用の色の配列。

### データセット

このモデルは、国立国会図書館が提供する[NDL-DocL レイアウトデータセット](https://github.com/ndl-lab/layout-dataset)を使用して学習されています。このデータセットには日本語古典籍のレイアウト情報が含まれています。

### モデル

検出モデルはYOLOv12ベースで、日本語古典籍の解析に最適化されています。

事前学習済みモデルはGitHub Releases経由で利用可能です（上記の[モデルのダウンロード](#モデルのダウンロード)セクションを参照）。モデルはnpmパッケージとは別に配布され、パッケージサイズを軽量に保っています。

### ライセンス

MIT

### コントリビューション

プルリクエストを歓迎します！

### 謝辞

- [NDL-DocL レイアウトデータセット](https://github.com/ndl-lab/layout-dataset) - 国立国会図書館
- ONNX Runtime Webチーム

---

## English

Japanese classical document layout analysis library using ONNX Runtime for detecting text regions, illustrations, and stamps in historical Japanese documents.

### Features

- 🏯 Specialized for Japanese classical documents (古典籍)
- 🚀 Browser-based inference using ONNX Runtime Web
- 📦 Lightweight and easy to integrate
- 📘 Full TypeScript support
- 🎯 Detects 5 types of regions:
  - Overall layout (全体)
  - Handwritten text (手書き)
  - Typographic text (活字)
  - Illustrations (図版)
  - Stamps/Seals (印判)

### See It In Action

Check out the live demo at [https://koten-layout.netlify.app/](https://koten-layout.netlify.app/)

### Installation

```bash
npm install koten-layout-detector onnxruntime-web
```

### Model Download

The pre-trained ONNX model is available via GitHub Releases:

```
https://github.com/yuta1984/koten-layout-detector/releases/download/v1.1.0/best.onnx
```

Or use the latest version:

```
https://github.com/yuta1984/koten-layout-detector/releases/latest/download/best.onnx
```

Model size: ~36MB

### Usage

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

// Load the ONNX model from GitHub Releases
const MODEL_URL = 'https://github.com/yuta1984/koten-layout-detector/releases/download/v1.1.0/best.onnx'
const session = await loadModel(MODEL_URL)

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

#### TypeScript

Full TypeScript type definitions are available:

```typescript
import {
  loadModel,
  preprocess,
  runInference,
  postprocess,
  drawDetections,
  type Detection,
  type PreprocessResult,
  type ClassDefinition
} from 'koten-layout-detector'
import type { InferenceSession } from 'onnxruntime-web'

const MODEL_URL = 'https://github.com/yuta1984/koten-layout-detector/releases/download/v1.1.0/best.onnx'

// Type-safe inference
const session: InferenceSession = await loadModel(MODEL_URL)

const img = new Image()
img.src = '/path/to/classical-document.jpg'
await img.decode()

const { tensor, meta }: PreprocessResult = preprocess(img)
const outputTensor = await runInference(session, tensor)
const detections: Detection[] = postprocess(outputTensor, meta, 0.5, 0.45)

// Type-checked detection results
detections.forEach((det: Detection) => {
  console.log(`Detected: ${det.label} (confidence: ${(det.conf * 100).toFixed(1)}%)`)
  console.log(`Position: (${det.x1}, ${det.y1}) - (${det.x2}, ${det.y2})`)
})
```

### API Reference

##### `loadModel(modelUrl: string): Promise<InferenceSession>`

Loads an ONNX model from the specified URL.

#### `preprocess(img: HTMLImageElement): { tensor: Tensor, meta: Object }`

Preprocesses an image for inference with letterbox resizing.

Returns:
- `tensor`: ONNX tensor ready for inference
- `meta`: Metadata for postprocessing (scale, padding, original dimensions)

#### `runInference(session: InferenceSession, tensor: Tensor): Promise<Tensor>`

Runs inference on the preprocessed tensor.

#### `postprocess(outputTensor: Tensor, meta: Object, confThreshold?: number, iouThreshold?: number): Array<Detection>`

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

#### `drawDetections(canvas: HTMLCanvasElement, img: HTMLImageElement, detections: Array<Detection>): void`

Draws the original image and detection boxes on a canvas.

#### `CLASSES`

Array of class definitions with ID, key, and Japanese labels.

#### `COLORS`

Array of colors for each class for visualization.

### Dataset

This model is trained on the [NDL-DocL Layout Dataset](https://github.com/ndl-lab/layout-dataset) provided by the National Diet Library of Japan. The dataset contains annotated layout information for Japanese classical documents.

### Model

The detection model is based on YOLOv12, optimized for classical Japanese document analysis.

The pre-trained model is available via GitHub Releases (see [Model Download](#model-download) section above). The model is distributed separately from the npm package to keep the package lightweight.

### License

MIT

### Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Acknowledgments

- [NDL-DocL Layout Dataset](https://github.com/ndl-lab/layout-dataset) - National Diet Library of Japan
- ONNX Runtime Web team for the excellent inference engine
