import { useState } from 'react'

/**
 * AnswerViewer
 * ─────────────────────────────────────────────────────────
 * Renders the most recent RAG response:
 *   - The LLM-generated answer (prominently)
 *   - Processing metadata (time, chunk count)
 *   - An expandable "Retrieved Context" section listing
 *     each source chunk with its similarity score
 *
 * Props:
 *   response — the raw API response from POST /api/v1/query:
 *   {
 *     question, answer, retrieved_chunks, processing_time_seconds
 *   }
 */
export default function AnswerViewer({ response }) {
  const [chunksOpen, setChunksOpen] = useState(false)

  // Nothing to show yet
  if (!response) {
    return (
      <section className="card flex flex-col items-center justify-center gap-3 py-10 text-center">
        <AnswerPlaceholderIcon className="h-12 w-12 text-slate-200" />
        <p className="text-sm font-medium text-slate-400">
          Generated answers will appear here
        </p>
        <p className="max-w-xs text-xs text-slate-400">
          Upload documents and ask a question to see a RAG-grounded response from Mistral-7B-Instruct.
        </p>
      </section>
    )
  }

  const { question, answer, retrieved_chunks = [], processing_time_seconds } = response
  const chunkCount = retrieved_chunks.length

  return (
    <section className="card space-y-5">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-100">
          <SparkIcon className="h-5 w-5 text-emerald-600" />
        </div>
        <div>
          <h2 className="text-base font-semibold text-slate-800">Generated Answer</h2>
          <p className="text-xs text-slate-500">Grounded in your document corpus</p>
        </div>
      </div>

      {/* Question recap */}
      <div className="rounded-xl bg-slate-50 px-4 py-3 text-sm">
        <span className="mr-1.5 text-xs font-semibold uppercase tracking-wide text-slate-400">Q</span>
        <span className="text-slate-700">{question}</span>
      </div>

      {/* Answer */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">Answer</span>
          <div className="h-px flex-1 bg-slate-100" />
        </div>
        <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-800">
          {answer}
        </p>
      </div>

      {/* Metadata row */}
      <div className="flex flex-wrap items-center gap-2">
        {processing_time_seconds != null && (
          <span className="badge badge-info">
            <ClockIcon className="h-3 w-3" />
            {processing_time_seconds.toFixed(2)}s
          </span>
        )}
        {chunkCount > 0 && (
          <span className="badge badge-success">
            <LayersIcon className="h-3 w-3" />
            {chunkCount} source chunk{chunkCount !== 1 ? 's' : ''}
          </span>
        )}
      </div>

      {/* Expandable source chunks */}
      {chunkCount > 0 && (
        <div className="space-y-2">
          <button
            type="button"
            onClick={() => setChunksOpen((o) => !o)}
            className="flex w-full items-center justify-between rounded-xl border border-slate-200 bg-slate-50 px-4 py-2.5 text-sm font-medium text-slate-700 transition-colors hover:bg-slate-100"
          >
            <span className="flex items-center gap-2">
              <DatabaseIcon className="h-4 w-4 text-slate-400" />
              Retrieved Context Chunks
            </span>
            <ChevronIcon
              className={`h-4 w-4 text-slate-400 transition-transform duration-200 ${
                chunksOpen ? 'rotate-180' : ''
              }`}
            />
          </button>

          {chunksOpen && (
            <ul className="space-y-3 pt-1">
              {retrieved_chunks.map((chunk, idx) => (
                <ChunkCard key={idx} chunk={chunk} index={idx + 1} />
              ))}
            </ul>
          )}
        </div>
      )}
    </section>
  )
}

// ── sub-components ───────────────────────────────────────────────────────────

function ChunkCard({ chunk, index }) {
  const [expanded, setExpanded] = useState(false)
  const preview = chunk.text.slice(0, 220)
  const isLong  = chunk.text.length > 220

  // Map cosine score (0–1) to a colour class
  const scoreColour =
    chunk.score >= 0.8 ? 'text-emerald-600 bg-emerald-50 border-emerald-100' :
    chunk.score >= 0.6 ? 'text-amber-600  bg-amber-50  border-amber-100'   :
                         'text-slate-500  bg-slate-50  border-slate-100'

  return (
    <li className="overflow-hidden rounded-xl border border-slate-200 bg-white">
      {/* Chunk header */}
      <div className="flex items-center justify-between gap-3 border-b border-slate-100 bg-slate-50 px-4 py-2.5">
        <div className="flex items-center gap-2 min-w-0">
          <span className="shrink-0 text-xs font-bold text-slate-500">#{index}</span>
          <span className="truncate text-xs font-medium text-slate-600">
            {chunk.filename || 'Unknown source'}
          </span>
          {chunk.chunk_id != null && (
            <span className="shrink-0 text-xs text-slate-400">chunk {chunk.chunk_id}</span>
          )}
        </div>
        <span
          className={`shrink-0 rounded-lg border px-2 py-0.5 text-xs font-semibold ${scoreColour}`}
          title="Cosine similarity score"
        >
          {(chunk.score * 100).toFixed(1)}%
        </span>
      </div>

      {/* Chunk text */}
      <div className="px-4 py-3">
        <p className="text-xs leading-relaxed text-slate-700">
          {expanded || !isLong ? chunk.text : `${preview}…`}
        </p>
        {isLong && (
          <button
            type="button"
            onClick={() => setExpanded((e) => !e)}
            className="mt-1.5 text-xs font-medium text-brand-600 hover:underline"
          >
            {expanded ? 'Show less' : 'Show more'}
          </button>
        )}
      </div>
    </li>
  )
}

// ── icons ────────────────────────────────────────────────────────────────────

function SparkIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
    </svg>
  )
}

function AnswerPlaceholderIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M7.5 8.25h9m-9 3H12m-9.75 1.51c0 1.6 1.123 2.994 2.707 3.227 1.129.166 2.27.293 3.423.379.35.026.67.21.865.501L12 21l2.755-4.133a1.14 1.14 0 01.865-.501 48.172 48.172 0 003.423-.379c1.584-.233 2.707-1.626 2.707-3.228V6.741c0-1.602-1.123-2.995-2.707-3.228A48.394 48.394 0 0012 3c-2.392 0-4.744.175-7.043.513C3.373 3.746 2.25 5.14 2.25 6.741v6.018z" />
    </svg>
  )
}

function ClockIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  )
}

function LayersIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M6.429 9.75L2.25 12l4.179 2.25m0-4.5l5.571 3 5.571-3m-11.142 0L2.25 7.5 12 2.25l9.75 5.25-4.179 2.25m0 0L21.75 12l-4.179 2.25m0 0l4.179 2.25L12 21.75 2.25 16.5l4.179-2.25m11.142 0l-5.571 3-5.571-3" />
    </svg>
  )
}

function DatabaseIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375m16.5 5.625c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125" />
    </svg>
  )
}

function ChevronIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 8.25l-7.5 7.5-7.5-7.5" />
    </svg>
  )
}
