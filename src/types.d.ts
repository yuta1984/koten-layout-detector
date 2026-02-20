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

/**
 * Preprocesses an image for inference with letterbox resizing
 * @param img - HTML image element to preprocess
 * @returns Preprocessed tensor and metadata
 */
export function preprocess(img: HTMLImageElement): PreprocessResult

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
