/**
 * Type definitions for koten-layout-detector
 */

import type { InferenceSession, Tensor } from 'onnxruntime-web'

/**
 * Class definition for detected regions
 */
export interface ClassDefinition {
  id: number
  key: string
  ja: string
}

/**
 * Detection result with bounding box and classification
 */
export interface Detection {
  x1: number
  y1: number
  x2: number
  y2: number
  conf: number
  classId: number
  label: string
  color: string
}

/**
 * Metadata from preprocessing
 */
export interface PreprocessMeta {
  scale: number
  padX: number
  padY: number
  origW: number
  origH: number
}

/**
 * Result of preprocessing
 */
export interface PreprocessResult {
  tensor: Tensor
  meta: PreprocessMeta
}

/**
 * Class definitions for NDL-DocL classical document dataset
 */
export const CLASSES: ClassDefinition[]

/**
 * Colors for visualization of each class
 */
export const COLORS: string[]

/**
 * Loads an ONNX model from the specified URL
 * @param modelUrl - URL to the ONNX model file
 * @returns Promise resolving to an ONNX Runtime inference session
 */
export function loadModel(modelUrl: string): Promise<InferenceSession>

/** Anything drawable to a canvas with width/height (image / canvas / bitmap) */
export type ImageSource = HTMLImageElement | HTMLCanvasElement | ImageBitmap

/**
 * Preprocesses an image for inference with letterbox resizing.
 * Also accepts a deskewed canvas (see `deskew`).
 * @param img - image / canvas / bitmap to preprocess
 * @returns Preprocessed tensor and metadata
 */
export function preprocess(img: ImageSource): PreprocessResult

/** Region in original-image coordinates */
export interface Region { x1: number; y1: number; x2: number; y2: number }

/** Options for skew estimation / deskew */
export interface DeskewOptions {
  /** downscale longest side to this many px for estimation (default 600) */
  maxDim?: number
  /** restrict estimation to this region (e.g. a detected text region) */
  region?: Region
  /** search range ±deg (default 10) */
  maxAngle?: number
  /** coarse search step in deg (default 1) */
  coarseStep?: number
  /** fine search step in deg (default 0.2) */
  fineStep?: number
  /** fill color for the rotated canvas margins (default '#ffffff') */
  background?: string
  /** skip correction when |angle| below this (default 0.3) */
  minAngle?: number
}

/**
 * Estimate the skew angle [deg] of a vertical-text page.
 * The returned angle straightens the page when passed to `ctx.rotate(angle*PI/180)`.
 */
export function estimateSkewAngle(source: ImageSource, opts?: DeskewOptions): number

/** Rotate the source by angleDeg and return a deskewed canvas (margins padded). */
export function deskewImage(source: ImageSource, angleDeg: number, opts?: { background?: string }): HTMLCanvasElement

/**
 * Estimate + correct in one call. Returns the (possibly unchanged) source and the
 * applied angle. Detections from `preprocess(result.canvas)` are in deskewed coords.
 */
export function deskew(source: ImageSource, opts?: DeskewOptions): { canvas: ImageSource; angle: number }

/** Pure core: estimate skew [deg] from ink pixel coordinates/weights (DOM-free, testable). */
export function estimateSkewFromInk(
  xs: ArrayLike<number>, ys: ArrayLike<number>, ws: ArrayLike<number>,
  w: number, h: number, opts?: DeskewOptions
): { angle: number; score: number; baseScore: number }

/** Pure core: edge-energy of the x' projection histogram at rotation a (rad). */
export function scoreAngle(
  xs: ArrayLike<number>, ys: ArrayLike<number>, ws: ArrayLike<number>,
  cx: number, cy: number, aRad: number, nbins: number, span: number
): number

/**
 * Runs inference on the preprocessed tensor
 * @param session - ONNX Runtime inference session
 * @param tensor - Preprocessed input tensor
 * @returns Promise resolving to output tensor
 */
export function runInference(session: InferenceSession, tensor: Tensor): Promise<Tensor>

/**
 * Postprocesses the model output into detection results
 * @param outputTensor - Output tensor from inference
 * @param meta - Metadata from preprocessing
 * @param confThreshold - Confidence threshold (default: 0.5)
 * @param iouThreshold - IoU threshold for NMS (default: 0.45)
 * @returns Array of detection results
 */
export function postprocess(
  outputTensor: Tensor,
  meta: PreprocessMeta,
  confThreshold?: number,
  iouThreshold?: number
): Detection[]

/**
 * Draws detection results on a canvas
 * @param canvas - HTML canvas element to draw on
 * @param img - Original image
 * @param detections - Array of detection results
 */
export function drawDetections(
  canvas: HTMLCanvasElement,
  img: HTMLImageElement,
  detections: Detection[]
): void

/**
 * Calculates Intersection over Union (IoU) between two bounding boxes
 * @param a - First bounding box
 * @param b - Second bounding box
 * @returns IoU value (0-1)
 */
export function iou(
  a: { x1: number; y1: number; x2: number; y2: number },
  b: { x1: number; y1: number; x2: number; y2: number }
): number

/**
 * Applies Non-Maximum Suppression to detection results
 * @param detections - Array of detection results
 * @param iouThreshold - IoU threshold (default: 0.45)
 * @returns Filtered array of detection results after NMS
 */
export function nms(
  detections: Array<{ x1: number; y1: number; x2: number; y2: number; conf: number; classId: number }>,
  iouThreshold?: number
): Array<{ x1: number; y1: number; x2: number; y2: number; conf: number; classId: number }>
