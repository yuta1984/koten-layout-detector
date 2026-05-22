/**
 * deskew.js — 古典籍(縦書き)ページの傾き補正
 *
 * 縦書きの本文は「列」が縦に走る。ページが傾くと、墨を x 軸へ射影した
 * プロファイル(列ごとのピーク + 列間の谷)がなまる。逆に正しい角度へ
 * 戻すとプロファイルのエッジが最も鋭くなる。これを利用して
 *   score(a) = Σ (P[b+1]-P[b])^2   (P = x' 射影ヒストグラム)
 * を最大化する回転角 a を粗→細で探索する(投影プロファイル法)。
 *
 * 角度推定(estimateSkewFromInk)と画像回転(deskewImage)は **同一の回転行列**
 *   x' = (x-cx)cos a - (y-cy)sin a,  y' = (x-cx)sin a + (y-cy)cos a
 * を使う。これは canvas の `ctx.rotate(a)`(y 下向き=時計回り)と一致するので、
 * 推定した a をそのまま `ctx.rotate(a)` に渡せば真っ直ぐになる(符号ずれ無し)。
 *
 * 計算コア(estimateSkewFromInk / scoreAngle)は DOM 非依存 → Node でテスト可能。
 * canvas 依存部(rasterize / deskewImage)はブラウザ用の薄いラッパ。
 */

const DEG = Math.PI / 180

// ---------------------------------------------------------------------------
// 計算コア(DOM 非依存)
// ---------------------------------------------------------------------------

/**
 * ある回転角 a(rad)での x' 射影プロファイルのエッジエネルギーを返す。
 * span/nbins を全角度で固定するので、スコアは角度間で直接比較できる。
 * @param {Float64Array|number[]} xs - 墨ピクセル x
 * @param {Float64Array|number[]} ys - 墨ピクセル y
 * @param {Float64Array|number[]} ws - 墨ピクセル重み(darkness)
 * @param {number} cx @param {number} cy - 回転中心
 * @param {number} aRad - 回転角(rad)
 * @param {number} nbins - ヒストグラムビン数
 * @param {number} span - x' の想定全幅(対角長)
 * @returns {number}
 */
export function scoreAngle(xs, ys, ws, cx, cy, aRad, nbins, span) {
  const c = Math.cos(aRad), s = Math.sin(aRad)
  const hist = new Float64Array(nbins)
  const half = span / 2
  const scale = nbins / span
  for (let i = 0; i < xs.length; i++) {
    const xp = (xs[i] - cx) * c - (ys[i] - cy) * s
    let b = ((xp + half) * scale) | 0
    if (b < 0) b = 0; else if (b >= nbins) b = nbins - 1
    hist[b] += ws[i]
  }
  let energy = 0
  for (let b = 1; b < nbins; b++) {
    const d = hist[b] - hist[b - 1]
    energy += d * d
  }
  return energy
}

/**
 * 墨ピクセル群から傾き角[deg]を推定(粗→細探索)。
 * 返り値 a は `ctx.rotate(a*DEG)` でページが真っ直ぐになる角度。
 * @param {Float64Array|number[]} xs
 * @param {Float64Array|number[]} ys
 * @param {Float64Array|number[]} ws
 * @param {number} w - ラスタ幅 @param {number} h - ラスタ高
 * @param {Object} [opts]
 * @param {number} [opts.maxAngle=10] - 探索範囲 ±deg
 * @param {number} [opts.coarseStep=1] - 粗探索の刻み deg
 * @param {number} [opts.fineStep=0.2] - 細探索の刻み deg
 * @returns {{ angle:number, score:number, baseScore:number }}
 */
export function estimateSkewFromInk(xs, ys, ws, w, h, opts = {}) {
  const maxAngle = opts.maxAngle ?? 10
  const coarseStep = opts.coarseStep ?? 1
  const fineStep = opts.fineStep ?? 0.2
  const cx = w / 2, cy = h / 2
  const span = Math.sqrt(w * w + h * h)
  const nbins = Math.max(64, Math.round(w))

  const search = (lo, hi, step) => {
    let best = -Infinity, bestA = 0
    for (let a = lo; a <= hi + 1e-9; a += step) {
      const sc = scoreAngle(xs, ys, ws, cx, cy, a * DEG, nbins, span)
      if (sc > best) { best = sc; bestA = a }
    }
    return { bestA, best }
  }

  const coarse = search(-maxAngle, maxAngle, coarseStep)
  const fine = search(coarse.bestA - coarseStep, coarse.bestA + coarseStep, fineStep)
  const baseScore = scoreAngle(xs, ys, ws, cx, cy, 0, nbins, span)
  return { angle: fine.bestA, score: fine.best, baseScore }
}

// ---------------------------------------------------------------------------
// canvas 依存部(ブラウザ用)
// ---------------------------------------------------------------------------

/**
 * 画像(またはcanvas/ImageBitmap)を縮小グレースケール化し、墨ピクセルの
 * 座標と重みを抽出する。重み = max(0, darkness - 平均darkness)(閾値フリー)。
 * @param {CanvasImageSource & {width:number,height:number}} source
 * @param {number} maxDim - 縮小後の最大辺
 * @param {{x1:number,y1:number,x2:number,y2:number}|null} region - 元画像座標での領域(任意)
 * @returns {{xs:Float64Array, ys:Float64Array, ws:Float64Array, w:number, h:number}}
 */
function rasterizeInk(source, maxDim, region) {
  const sx = region ? region.x1 : 0
  const sy = region ? region.y1 : 0
  const sw = region ? (region.x2 - region.x1) : source.width
  const sh = region ? (region.y2 - region.y1) : source.height
  const scale = Math.min(1, maxDim / Math.max(sw, sh))
  const w = Math.max(1, Math.round(sw * scale))
  const h = Math.max(1, Math.round(sh * scale))

  const canvas = document.createElement('canvas')
  canvas.width = w; canvas.height = h
  const ctx = canvas.getContext('2d', { willReadFrequently: true })
  ctx.drawImage(source, sx, sy, sw, sh, 0, 0, w, h)
  const { data } = ctx.getImageData(0, 0, w, h)

  // darkness = 1 - luma/255
  const dark = new Float64Array(w * h)
  let sum = 0
  for (let i = 0, p = 0; i < dark.length; i++, p += 4) {
    const luma = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2]
    const d = 1 - luma / 255
    dark[i] = d; sum += d
  }
  const mean = sum / dark.length

  const xs = [], ys = [], ws = []
  for (let y = 0, i = 0; y < h; y++) {
    for (let x = 0; x < w; x++, i++) {
      const wgt = dark[i] - mean
      if (wgt > 0) { xs.push(x); ys.push(y); ws.push(wgt) }
    }
  }
  return { xs: Float64Array.from(xs), ys: Float64Array.from(ys), ws: Float64Array.from(ws), w, h }
}

/**
 * 画像(縦書きページ)の傾き角[deg]を推定する。
 * @param {CanvasImageSource & {width:number,height:number}} source
 * @param {Object} [opts]
 * @param {number} [opts.maxDim=600] - 推定用に縮小する最大辺(速度/精度のトレードオフ)
 * @param {{x1,y1,x2,y2}} [opts.region] - 推定に使う領域(検出した本文領域など)
 * @param {number} [opts.maxAngle=10] @param {number} [opts.coarseStep=1] @param {number} [opts.fineStep=0.2]
 * @returns {number} ctx.rotate(angle*π/180) で真っ直ぐになる角度[deg]
 */
export function estimateSkewAngle(source, opts = {}) {
  const { xs, ys, ws, w, h } = rasterizeInk(source, opts.maxDim ?? 600, opts.region ?? null)
  if (xs.length < 100) return 0
  return estimateSkewFromInk(xs, ys, ws, w, h, opts).angle
}

/**
 * 画像を angle[deg] 回転して傾きを補正した canvas を返す。
 * 角は切れないようにキャンバスを拡張し、余白は背景色で塗る。
 * @param {CanvasImageSource & {width:number,height:number}} source
 * @param {number} angleDeg - estimateSkewAngle が返す角度
 * @param {Object} [opts]
 * @param {string} [opts.background='#ffffff'] - 余白色(OCR向けは白、検出器再投入なら 'rgb(114,114,114)')
 * @returns {HTMLCanvasElement}
 */
export function deskewImage(source, angleDeg, opts = {}) {
  const a = angleDeg * DEG
  const W = source.width, H = source.height
  const cos = Math.abs(Math.cos(a)), sin = Math.abs(Math.sin(a))
  const outW = Math.ceil(W * cos + H * sin)
  const outH = Math.ceil(W * sin + H * cos)

  const canvas = document.createElement('canvas')
  canvas.width = outW; canvas.height = outH
  const ctx = canvas.getContext('2d')
  ctx.fillStyle = opts.background ?? '#ffffff'
  ctx.fillRect(0, 0, outW, outH)
  ctx.translate(outW / 2, outH / 2)
  ctx.rotate(a)
  ctx.drawImage(source, -W / 2, -H / 2)
  return canvas
}

/**
 * 推定 + 補正をまとめて行う。
 * @param {CanvasImageSource & {width:number,height:number}} source
 * @param {Object} [opts] - estimateSkewAngle / deskewImage のオプション + minAngle
 * @param {number} [opts.minAngle=0.3] - これ未満は補正せずそのまま返す(過補正・無駄な回転を防ぐ)
 * @returns {{ canvas: (HTMLCanvasElement|CanvasImageSource), angle: number }}
 */
export function deskew(source, opts = {}) {
  const angle = estimateSkewAngle(source, opts)
  if (Math.abs(angle) < (opts.minAngle ?? 0.3)) return { canvas: source, angle: 0 }
  return { canvas: deskewImage(source, angle, opts), angle }
}
