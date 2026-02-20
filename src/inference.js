import * as ort from 'onnxruntime-web'
import { nms } from './nms.js'

/** モデル入力サイズ */
const MODEL_SIZE = 640

/** レターボックスのパディング色（YOLO デフォルト: グレー 114） */
const PAD_COLOR = 114

/**
 * クラス定義
 * NDL-DocL 古典籍データセットの 5 クラス
 */
export const CLASSES = [
  { id: 0, key: '1_overall',      ja: '全体'   },
  { id: 1, key: '2_handwritten',  ja: '手書き' },
  { id: 2, key: '3_typography',   ja: '活字'   },
  { id: 3, key: '4_illustration', ja: '図版'   },
  { id: 4, key: '5_stamp',        ja: '印判'   },
]

export const COLORS = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

// ---------------------------------------------------------------------------
// モデルのロード
// ---------------------------------------------------------------------------

/**
 * ONNX セッションを作成して返す
 * @param {string} modelUrl - モデルファイルの URL
 * @returns {Promise<ort.InferenceSession>}
 */
export async function loadModel(modelUrl) {
  // WASM ファイルのパスを明示（Vite の static copy 先に合わせる）
  ort.env.wasm.wasmPaths = '/'

  const session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  })
  return session
}

// ---------------------------------------------------------------------------
// 前処理
// ---------------------------------------------------------------------------

/**
 * 画像をレターボックスリサイズして Float32Array テンソルに変換する
 * @param {HTMLImageElement} img
 * @returns {{ tensor: ort.Tensor, meta: Object }}
 *   meta: { scale, padX, padY, origW, origH }
 */
export function preprocess(img) {
  const canvas = document.createElement('canvas')
  canvas.width = MODEL_SIZE
  canvas.height = MODEL_SIZE
  const ctx = canvas.getContext('2d')

  // パディング色で塗りつぶし
  ctx.fillStyle = `rgb(${PAD_COLOR},${PAD_COLOR},${PAD_COLOR})`
  ctx.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE)

  // アスペクト比を保ったまま縮小
  const scale = Math.min(MODEL_SIZE / img.width, MODEL_SIZE / img.height)
  const newW = Math.round(img.width * scale)
  const newH = Math.round(img.height * scale)
  const padX = Math.floor((MODEL_SIZE - newW) / 2)
  const padY = Math.floor((MODEL_SIZE - newH) / 2)

  ctx.drawImage(img, padX, padY, newW, newH)

  const { data } = ctx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE)

  // HWC (RGBA) → CHW (RGB) Float32 & 正規化 ÷255
  const float32 = new Float32Array(3 * MODEL_SIZE * MODEL_SIZE)
  const pixelCount = MODEL_SIZE * MODEL_SIZE
  for (let i = 0; i < pixelCount; i++) {
    float32[i]                  = data[i * 4]     / 255.0 // R
    float32[i + pixelCount]     = data[i * 4 + 1] / 255.0 // G
    float32[i + pixelCount * 2] = data[i * 4 + 2] / 255.0 // B
  }

  return {
    tensor: new ort.Tensor('float32', float32, [1, 3, MODEL_SIZE, MODEL_SIZE]),
    meta: { scale, padX, padY, origW: img.width, origH: img.height },
  }
}

// ---------------------------------------------------------------------------
// 推論
// ---------------------------------------------------------------------------

/**
 * ONNX セッションで推論を実行する
 * @param {ort.InferenceSession} session
 * @param {ort.Tensor} tensor
 * @returns {Promise<ort.Tensor>} 出力テンソル [1, 9, 8400]
 */
export async function runInference(session, tensor) {
  const inputName = session.inputNames[0]
  const feeds = { [inputName]: tensor }
  const results = await session.run(feeds)
  return results[session.outputNames[0]]
}

// ---------------------------------------------------------------------------
// 後処理
// ---------------------------------------------------------------------------

/**
 * ONNX 出力テンソルを検出結果に変換する
 * @param {ort.Tensor} outputTensor - shape [1, 4+nc, 8400]
 * @param {Object} meta - preprocess() が返す meta
 * @param {number} confThreshold - 信頼度閾値（デフォルト 0.5）
 * @param {number} iouThreshold  - NMS IoU 閾値（デフォルト 0.45）
 * @returns {Array} 検出結果リスト { x1, y1, x2, y2, conf, classId, label, color }
 */
export function postprocess(outputTensor, meta, confThreshold = 0.5, iouThreshold = 0.45) {
  const [, numChannels, numPreds] = outputTensor.dims
  const data = outputTensor.data
  const nc = numChannels - 4 // クラス数

  const raw = []

  for (let i = 0; i < numPreds; i++) {
    // クラス別スコアの最大値と ID を取得
    let maxScore = -Infinity
    let classId = 0
    for (let c = 0; c < nc; c++) {
      const score = data[(4 + c) * numPreds + i]
      if (score > maxScore) {
        maxScore = score
        classId = c
      }
    }

    if (maxScore < confThreshold) continue

    // cx, cy, w, h（640px スケール）→ x1, y1, x2, y2（元画像スケール）
    const cx = data[0 * numPreds + i]
    const cy = data[1 * numPreds + i]
    const w  = data[2 * numPreds + i]
    const h  = data[3 * numPreds + i]

    // レターボックスのパディングと縮小率を元に戻す
    const x1 = ((cx - w / 2) - meta.padX) / meta.scale
    const y1 = ((cy - h / 2) - meta.padY) / meta.scale
    const x2 = ((cx + w / 2) - meta.padX) / meta.scale
    const y2 = ((cy + h / 2) - meta.padY) / meta.scale

    raw.push({ x1, y1, x2, y2, conf: maxScore, classId })
  }

  const kept = nms(raw, iouThreshold)

  return kept.map((d) => ({
    ...d,
    label: CLASSES[d.classId]?.ja ?? String(d.classId),
    color: COLORS[d.classId] ?? '#ffffff',
  }))
}

// ---------------------------------------------------------------------------
// Canvas 描画
// ---------------------------------------------------------------------------

/**
 * 元画像と検出結果を Canvas に描画する
 * @param {HTMLCanvasElement} canvas
 * @param {HTMLImageElement} img
 * @param {Array} detections - postprocess() の戻り値
 */
export function drawDetections(canvas, img, detections) {
  canvas.width  = img.width
  canvas.height = img.height
  const ctx = canvas.getContext('2d')
  ctx.drawImage(img, 0, 0)

  for (const d of detections) {
    const x1 = Math.max(0, d.x1)
    const y1 = Math.max(0, d.y1)
    const bw = d.x2 - x1
    const bh = d.y2 - y1

    // ボックス
    ctx.strokeStyle = d.color
    ctx.lineWidth = Math.max(2, img.width / 300)
    ctx.strokeRect(x1, y1, bw, bh)

    // ラベル背景
    const fontSize = Math.max(14, img.width / 50)
    ctx.font = `bold ${fontSize}px sans-serif`
    const text = `${d.label} ${(d.conf * 100).toFixed(0)}%`
    const textW = ctx.measureText(text).width
    const textH = fontSize * 1.4
    ctx.fillStyle = d.color
    ctx.fillRect(x1, y1 - textH, textW + 8, textH)

    // ラベルテキスト
    ctx.fillStyle = '#ffffff'
    ctx.fillText(text, x1 + 4, y1 - fontSize * 0.2)
  }
}
