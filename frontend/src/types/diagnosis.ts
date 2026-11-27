export type DiagnosisStatus =
  | 'ok'
  | 'needs_clarification'
  | 'unclear_image'
  | 'irrelevant_image'
  | 'no_indicator_visible'
  | 'unknown'

export type ImageStatus =
  | 'suitable'
  | 'unclear_image'
  | 'irrelevant_image'
  | 'no_indicator_visible'

export type DiagnosisStrategy = 'ordinary' | 'contrastive'

/** Fields accepted by POST /api/v1/diagnose as multipart form data. */
export interface DiagnosisRequest {
  image: File
  manufacturer?: string | null
  model?: string | null
  year?: number | null
  question?: string | null
  strategy?: DiagnosisStrategy | null
  answers?: Record<string, string> | null
}

export interface ClarificationQuestion {
  id: string
  question: string
  optional: boolean
}

export interface IndicatorResult {
  label_id: string
  display_name: string
  confidence: 'low' | 'medium' | 'high'
  visible_colour: string | null
  state: 'steady' | 'flashing' | 'unknown'
  visual_evidence: string
  meaning: string
  severity: string
  recommended_action: string
}

export interface DiagnosisMetadata {
  strategy: DiagnosisStrategy
  model: string
  prompt_version: string
  temperature: number
  latency_ms: number
  few_shot_examples_used: boolean
  few_shot_example_count: number
  demonstration_ids: string[]
  prompt_mode: 'suitability_only' | 'ordinary_few_shot' | 'contrastive_few_shot'
}

export interface DiagnosisResponse {
  status: DiagnosisStatus
  image_status: ImageStatus
  indicator: IndicatorResult | null
  clarification_questions: ClarificationQuestion[]
  message: string
  safety_disclaimer: string
  metadata: DiagnosisMetadata
}
