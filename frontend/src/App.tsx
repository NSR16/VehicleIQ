import { useEffect, useRef, useState, type ChangeEvent, type FormEvent } from 'react'
import { diagnose, DiagnosisApiError } from './api/diagnose'
import type { ClarificationQuestion, DiagnosisRequest, DiagnosisResponse, DiagnosisStrategy } from './types/diagnosis'
import './App.css'

const MAX_IMAGE_BYTES = 10 * 1024 * 1024
const ACCEPTED_TYPES = ['image/png', 'image/jpeg', 'image/webp']

interface ClarificationRound {
  questions: ClarificationQuestion[]
  answers: Record<string, string>
}

function App() {
  const [image, setImage] = useState<File | null>(null)
  const [manufacturer, setManufacturer] = useState('')
  const [model, setModel] = useState('')
  const [year, setYear] = useState('')
  const [question, setQuestion] = useState('')
  const [strategy, setStrategy] = useState<DiagnosisStrategy>('ordinary')
  const [preview, setPreview] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [result, setResult] = useState<DiagnosisResponse | null>(null)
  const [originalRequest, setOriginalRequest] = useState<DiagnosisRequest | null>(null)
  const [submittedAnswers, setSubmittedAnswers] = useState<Record<string, string>>({})
  const [answerDrafts, setAnswerDrafts] = useState<Record<string, string>>({})
  const previewRef = useRef('')
  const resultRegionRef = useRef<HTMLDivElement>(null)
  const requestInFlightRef = useRef(false)
  const [clarificationRounds, setClarificationRounds] = useState<ClarificationRound[]>([])

  useEffect(() => {
    return () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current)
    }
  }, [])

  useEffect(() => {
    if (!result) return
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    resultRegionRef.current?.scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth', block: 'start' })
  }, [result])

  function setSelectedImage(file: File | null) {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current)
    previewRef.current = file ? URL.createObjectURL(file) : ''
    setPreview(previewRef.current)
    setImage(file)
    setResult(null)
    setOriginalRequest(null)
    setSubmittedAnswers({})
    setAnswerDrafts({})
    setClarificationRounds([])
    setError('')
  }

  function selectImage(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0]
    event.target.value = ''
    if (!selected) return
    if (!ACCEPTED_TYPES.includes(selected.type)) {
      setError('Choose a PNG, JPEG, or WebP image.')
      return
    }
    if (selected.size > MAX_IMAGE_BYTES) {
      setError('The image must be 10 MB or smaller.')
      return
    }
    setSelectedImage(selected)
  }

  async function submitDiagnosis(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (requestInFlightRef.current) return
    if (!image) {
      setError('Add a dashboard image to continue.')
      return
    }
    requestInFlightRef.current = true
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const request: DiagnosisRequest = {
        image,
        manufacturer: manufacturer || null,
        model: model || null,
        year: year ? Number(year) : null,
        question: question || null,
        strategy,
      }
      setOriginalRequest(request)
      setSubmittedAnswers({})
      setAnswerDrafts({})
      setClarificationRounds([])
      const diagnosis = await diagnose(request)
      if (diagnosis.status === 'needs_clarification') {
        setClarificationRounds([{ questions: diagnosis.clarification_questions, answers: {} }])
      }
      setResult(diagnosis)
    } catch (caught) {
      setError(
        caught instanceof DiagnosisApiError
          ? caught.message
          : 'Something went wrong while preparing the diagnosis. Please try again.',
      )
    } finally {
      requestInFlightRef.current = false
      setLoading(false)
    }
  }

  async function submitClarification(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (requestInFlightRef.current || !originalRequest || result?.status !== 'needs_clarification') return

    const newAnswers = Object.fromEntries(
      result.clarification_questions
        .map(({ id }) => [id, answerDrafts[id]?.trim() ?? ''] as const)
        .filter(([, answer]) => answer.length > 0),
    )
    const combinedAnswers = { ...submittedAnswers, ...newAnswers }

    requestInFlightRef.current = true
    setLoading(true)
    setError('')
    try {
      const diagnosis = await diagnose({ ...originalRequest, answers: combinedAnswers })
      setSubmittedAnswers(combinedAnswers)
      setAnswerDrafts({})
      setClarificationRounds((rounds) => {
        const updatedRounds = rounds.map((round, index) =>
          index === rounds.length - 1
            ? { ...round, answers: { ...round.answers, ...newAnswers } }
            : round,
        )
        return diagnosis.status === 'needs_clarification'
          ? [...updatedRounds, { questions: diagnosis.clarification_questions, answers: {} }]
          : updatedRounds
      })
      setResult(diagnosis)
    } catch (caught) {
      setError(
        caught instanceof DiagnosisApiError
          ? caught.message
          : 'Something went wrong while preparing the diagnosis. Please try again.',
      )
    } finally {
      requestInFlightRef.current = false
      setLoading(false)
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="VehicleIQ home">
          <span className="brand-mark" aria-hidden="true">V</span>
          <span>Vehicle<span className="brand-accent">IQ</span></span>
        </a>
        <div className="service-status"><span className="status-dot" /> Diagnostic workspace</div>
      </header>

      <main id="top" className="main-content">
        <section className="intro">
          <div className="eyebrow"><span className="eyebrow-line" /> DASHBOARD INDICATOR ANALYSIS</div>
          <h1>Understand what your<br className="desktop-break" /> dashboard is telling you.</h1>
          <p className="intro-copy">Upload a clear photo of your dashboard to identify a visible warning or status indicator.</p>
        </section>

        <form className="workspace" onSubmit={submitDiagnosis}>
          <section className="panel upload-panel" aria-labelledby="upload-title">
            <div className="section-heading">
              <span className="step-number">01</span>
              <div><h2 id="upload-title">Dashboard image</h2><p>Use a clear, well-lit photo with the indicator in view.</p></div>
            </div>
            <div className={`upload-area ${image ? 'has-image' : ''}`}>
              {image && preview ? (
                <div className="preview-wrap">
                  <img src={preview} alt="Selected dashboard preview" className="image-preview" />
                  <div className="preview-caption"><span className="file-symbol">▧</span><span className="file-name">{image.name}</span><span className="file-size">{(image.size / (1024 * 1024)).toFixed(2)} MB</span></div>
                  <button className="remove-image" type="button" disabled={loading} onClick={() => setSelectedImage(null)}>Remove image</button>
                </div>
              ) : (
                <label className="upload-prompt" htmlFor="dashboard-image">
                  <span className="upload-icon" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M5 15v4h14v-4" /></svg></span>
                  <span className="upload-title">Choose an image or <span>browse files</span></span>
                  <span className="upload-hint">PNG, JPG or WebP · up to 10 MB</span>
                  <input id="dashboard-image" type="file" accept="image/png,image/jpeg,image/webp" onChange={selectImage} />
                </label>
              )}
            </div>
            <div className="privacy-note"><span aria-hidden="true">◇</span> Your image is processed for this diagnosis and is not saved by the app.</div>
          </section>

          <section className="panel details-panel" aria-labelledby="details-title">
            <div className="section-heading">
              <span className="step-number">02</span>
              <div><h2 id="details-title">Vehicle details <span className="optional-tag">OPTIONAL</span></h2><p>Context can help interpret the dashboard.</p></div>
            </div>
            <div className="field-grid">
              <label className="field"><span>Manufacturer</span><input value={manufacturer} onChange={(event) => setManufacturer(event.target.value)} maxLength={100} placeholder="e.g. Toyota" /></label>
              <label className="field"><span>Model</span><input value={model} onChange={(event) => setModel(event.target.value)} maxLength={100} placeholder="e.g. Corolla" /></label>
              <label className="field year-field"><span>Year</span><input type="number" min={1886} max={2100} value={year} onChange={(event) => setYear(event.target.value)} placeholder="e.g. 2022" /></label>
            </div>
            <label className="field question-field"><span>What would you like to know?</span><textarea value={question} onChange={(event) => setQuestion(event.target.value)} maxLength={1000} rows={3} placeholder="Ask a question about the indicator…" /></label>
          </section>

          <details className="panel research-settings">
            <summary className="research-summary">
              <span className="step-number">03</span>
              <span className="research-summary-copy">
                <strong>Research / Experimental Settings</strong>
                <small>Few-shot prompt strategy · experimental project setting</small>
              </span>
              <span className="research-chevron" aria-hidden="true" />
            </summary>
            <div className="research-content">
              <p className="research-note">These options are for comparing prompting strategies as part of the research project.</p>
              <div className="strategy-options">
                <label className={`strategy-option ${strategy === 'ordinary' ? 'selected' : ''}`}>
                  <input type="radio" name="strategy" value="ordinary" checked={strategy === 'ordinary'} onChange={() => setStrategy('ordinary')} />
                  <span className="radio-mark" /><span className="strategy-copy"><strong>Ordinary few-shot</strong><small>Learn from labeled examples</small></span><span className="recommended">DEFAULT</span>
                </label>
                <label className={`strategy-option ${strategy === 'contrastive' ? 'selected' : ''}`}>
                  <input type="radio" name="strategy" value="contrastive" checked={strategy === 'contrastive'} onChange={() => setStrategy('contrastive')} />
                  <span className="radio-mark" /><span className="strategy-copy"><strong>Contrastive few-shot</strong><small>Compare visual differences</small></span>
                </label>
              </div>
            </div>
          </details>

          {error && result?.status !== 'needs_clarification' && <div className="error-banner" role="alert"><span className="error-icon">!</span><span>{error}</span></div>}

          <div className="submit-row">
            <p className="submit-note">For informational assistance. Always refer to your owner’s manual.</p>
            <button className="submit-button" type="submit" disabled={loading || !image}>
              {loading ? <><span className="spinner" /> Analyzing image…</> : <>Analyze dashboard <span aria-hidden="true">→</span></>}
            </button>
          </div>
        </form>

        {result && <div className="result-region" ref={resultRegionRef}>
          {originalRequest && <ClarificationHistory request={originalRequest} rounds={clarificationRounds} />}
          <DiagnosisResult response={result} />
          {result.status === 'needs_clarification' && originalRequest && (
            <ClarificationForm
              response={result}
              drafts={answerDrafts}
              loading={loading}
              error={error}
              onDraftChange={(id, answer) => setAnswerDrafts((current) => ({ ...current, [id]: answer }))}
              onSubmit={submitClarification}
            />
          )}
        </div>}
      </main>

      <footer className="footer"><span>VehicleIQ</span><span>Preliminary guidance · Always verify with your owner’s manual</span></footer>
    </div>
  )
}

function DiagnosisResult({ response }: { response: DiagnosisResponse }) {
  const indicator = response.indicator
  const severityClass = indicator ? `result-severity-${indicator.severity}` : 'result-no-indicator'
  const headings = {
    ok: indicator?.display_name ?? 'Diagnosis complete',
    needs_clarification: 'More detail may help',
    unknown: 'Indicator not identified',
    unclear_image: 'Image is unclear',
    irrelevant_image: 'Image is unsuitable',
    no_indicator_visible: 'No indicator detected',
  } satisfies Record<DiagnosisResponse['status'], string>
  return (
    <section className={`result-card ${response.status === 'ok' ? 'result-success' : 'result-notice'} ${severityClass}`} aria-live="polite" aria-labelledby="result-title">
      <div className="result-heading"><span className="result-icon">{response.status === 'ok' ? '✓' : 'i'}</span><div><span className="result-kicker">ANALYSIS RESULT</span><h2 id="result-title">{headings[response.status]}</h2></div></div>
      <p className="result-message">{response.message}</p>
      {indicator && <div className="result-details">
        <div className="result-detail"><span>Severity</span><strong className={`severity-badge severity-${indicator.severity}`}><span aria-hidden="true">{severityIcon(indicator.severity)}</span>{indicator.severity}</strong></div>
        <div className="result-detail"><span>Indicator state</span><strong>{indicator.state}{indicator.visible_colour ? ` · ${indicator.visible_colour}` : ''}</strong></div>
        <div className="result-detail full-detail"><span>Visual evidence</span><strong>{indicator.visual_evidence}</strong></div>
        <div className="result-detail full-detail"><span>Meaning</span><strong>{indicator.meaning}</strong></div>
        <div className="result-detail full-detail"><span>Recommended action</span><strong>{indicator.recommended_action}</strong></div>
      </div>}
      <p className="safety-disclaimer">{response.safety_disclaimer}</p>
    </section>
  )
}

function severityIcon(severity: string): string {
  if (severity === 'high' || severity === 'medium') return '!'
  if (severity === 'low') return '✓'
  return 'i'
}

function ClarificationHistory({ request, rounds }: { request: DiagnosisRequest; rounds: ClarificationRound[] }) {
  const vehicleDetails = [request.manufacturer, request.model, request.year]
    .filter((value) => value !== null && value !== undefined && value !== '')
    .join(' · ')

  return (
    <section className="history-card" aria-labelledby="history-title">
      <h2 id="history-title">Diagnosis history</h2>
      <div className="history-entry">
        <div className="history-entry-heading"><span className="history-marker">01</span><strong>Initial request</strong></div>
        <dl className="history-facts">
          <div><dt>Image</dt><dd>{request.image.name}</dd></div>
          {vehicleDetails && <div><dt>Vehicle</dt><dd>{vehicleDetails}</dd></div>}
          {request.question && <div><dt>Question</dt><dd>{request.question}</dd></div>}
          <div><dt>Strategy</dt><dd>{request.strategy}</dd></div>
        </dl>
      </div>
      {rounds.map((round, roundIndex) => (
        <div className="history-entry" key={`round-${roundIndex}`}>
          <div className="history-entry-heading"><span className="history-marker">{String(roundIndex + 2).padStart(2, '0')}</span><strong>Clarification {roundIndex + 1}</strong></div>
          <ul className="history-answers">
            {round.questions.map((question) => (
              <li key={`${roundIndex}-${question.id}`}>
                <span className="history-question">{question.question}</span>
                <span className="history-answer">{round.answers[question.id] || 'No answer submitted'}</span>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </section>
  )
}

interface ClarificationFormProps {
  response: DiagnosisResponse
  drafts: Record<string, string>
  loading: boolean
  error: string
  onDraftChange: (id: string, answer: string) => void
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
}

function ClarificationForm({ response, drafts, loading, error, onDraftChange, onSubmit }: ClarificationFormProps) {
  return (
    <section className="clarification-card" aria-labelledby="clarification-title">
      <div className="clarification-heading">
        <span className="clarification-step">FOLLOW-UP</span>
        <div>
          <h2 id="clarification-title">A few questions about the indicator</h2>
          <p>Your original image and vehicle details will be included with your answers.</p>
        </div>
      </div>
      <form className="clarification-form" onSubmit={onSubmit}>
        {response.clarification_questions.map((question, index) => (
          <label className="clarification-question" key={question.id}>
            <span className="clarification-question-label">
              <span className="clarification-index">{String(index + 1).padStart(2, '0')}</span>
              <span>{question.question}</span>
              <span className="optional-label">{question.optional ? 'Optional' : 'Required'}</span>
            </span>
            <textarea
              id={`clarification-${question.id}`}
              value={drafts[question.id] ?? ''}
              onChange={(event) => onDraftChange(question.id, event.target.value)}
              maxLength={300}
              rows={2}
              required={!question.optional}
              placeholder="Type your answer…"
              aria-label={question.question}
            />
          </label>
        ))}
        {error && <div className="error-banner" role="alert"><span className="error-icon">!</span><span>{error}</span></div>}
        <div className="clarification-submit-row">
          <p>Optional questions can be left blank.</p>
          <button className="submit-button" type="submit" disabled={loading}>
            {loading ? <><span className="spinner" /> Sending answers…</> : <>Submit answers <span aria-hidden="true">→</span></>}
          </button>
        </div>
      </form>
    </section>
  )
}

export default App
