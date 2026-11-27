import type { DiagnosisRequest, DiagnosisResponse } from '../types/diagnosis'

const DIAGNOSE_ENDPOINT = '/api/v1/diagnose'

interface ApiErrorBody {
  detail?: string | Array<{ msg?: string }>
}

export class DiagnosisApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.name = 'DiagnosisApiError'
    this.status = status
  }
}

function errorMessage(body: ApiErrorBody, status: number): string {
  if (typeof body.detail === 'string') return body.detail
  if (Array.isArray(body.detail)) {
    const messages = body.detail.map((item) => item.msg).filter(Boolean)
    if (messages.length) return messages.join(' ')
  }
  return `Diagnosis request failed (HTTP ${status}). Please try again.`
}

export async function diagnose(request: DiagnosisRequest): Promise<DiagnosisResponse> {
  const form = new FormData()
  form.append('image', request.image)
  if (request.manufacturer?.trim()) form.append('manufacturer', request.manufacturer.trim())
  if (request.model?.trim()) form.append('model', request.model.trim())
  if (request.year !== undefined && request.year !== null) {
    form.append('year', String(request.year))
  }
  if (request.question?.trim()) form.append('question', request.question.trim())
  if (request.strategy) form.append('strategy', request.strategy)
  if (request.answers) form.append('answers', JSON.stringify(request.answers))

  let response: Response
  try {
    response = await fetch(DIAGNOSE_ENDPOINT, { method: 'POST', body: form })
  } catch {
    throw new DiagnosisApiError(
      'Could not reach VehicleIQ. Check that the backend is running on port 8000.',
      0,
    )
  }

  if (!response.ok) {
    let body: ApiErrorBody = {}
    try {
      body = (await response.json()) as ApiErrorBody
    } catch {
      // Keep a useful HTTP error when the server did not return JSON.
    }
    throw new DiagnosisApiError(errorMessage(body, response.status), response.status)
  }

  return (await response.json()) as DiagnosisResponse
}
