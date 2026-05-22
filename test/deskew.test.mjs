import { estimateSkewFromInk, scoreAngle } from '../src/deskew.js'

const DEG = Math.PI / 180
// 縦書き「列」を α 度傾けた合成パターンの墨点を生成
function makeTiltedStripes(alphaDeg, W = 300, H = 420, P = 18, lw = 5) {
  const cx = W / 2, cy = H / 2, slope = Math.tan(alphaDeg * DEG)
  const xs = [], ys = [], ws = []
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let phase = ((x - cx) - slope * (y - cy)) % P
      phase = ((phase % P) + P) % P
      if (phase < lw) { xs.push(x); ys.push(y); ws.push(1) }
    }
  }
  return { xs, ys, ws, W, H }
}

let pass = 0, fail = 0
for (const alpha of [-7, -3, 0, 2.4, 5, 8]) {
  const { xs, ys, ws, W, H } = makeTiltedStripes(alpha)
  const { angle, score, baseScore } = estimateSkewFromInk(xs, ys, ws, W, H, { maxAngle: 12 })
  const err = Math.abs(angle - alpha)
  const ok = err < 0.4 && (alpha === 0 || score >= baseScore)
  console.log(`inject ${String(alpha).padStart(5)}°  recovered ${angle.toFixed(2).padStart(6)}°  err ${err.toFixed(2)}  score/base ${(score/baseScore).toFixed(2)}  ${ok ? 'OK' : 'FAIL'}`)
  ok ? pass++ : fail++
}
console.log(`\n${pass} passed / ${fail} failed`)
process.exit(fail ? 1 : 0)
