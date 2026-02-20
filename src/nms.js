/**
 * IoU（Intersection over Union）を計算する
 * @param {Object} a - { x1, y1, x2, y2 }
 * @param {Object} b - { x1, y1, x2, y2 }
 * @returns {number} IoU 値（0〜1）
 */
export function iou(a, b) {
  const ix1 = Math.max(a.x1, b.x1)
  const iy1 = Math.max(a.y1, b.y1)
  const ix2 = Math.min(a.x2, b.x2)
  const iy2 = Math.min(a.y2, b.y2)

  const interW = Math.max(0, ix2 - ix1)
  const interH = Math.max(0, iy2 - iy1)
  const interArea = interW * interH

  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1)
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1)
  const unionArea = areaA + areaB - interArea

  return unionArea <= 0 ? 0 : interArea / unionArea
}

/**
 * Non-Maximum Suppression を適用する
 * @param {Array} detections - 検出結果の配列 ({ x1, y1, x2, y2, conf, classId })
 * @param {number} iouThreshold - IoU 閾値（デフォルト 0.45）
 * @returns {Array} NMS 後の検出結果
 */
export function nms(detections, iouThreshold = 0.45) {
  // クラスごとに NMS を適用
  const classIds = [...new Set(detections.map((d) => d.classId))]
  const result = []

  for (const cid of classIds) {
    let boxes = detections
      .filter((d) => d.classId === cid)
      .sort((a, b) => b.conf - a.conf) // 信頼度の高い順にソート

    while (boxes.length > 0) {
      const best = boxes.shift()
      result.push(best)
      boxes = boxes.filter((b) => iou(best, b) < iouThreshold)
    }
  }

  return result
}
