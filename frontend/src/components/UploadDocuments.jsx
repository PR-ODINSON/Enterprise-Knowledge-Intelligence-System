import { useCallback, useRef, useState } from 'react'
import { uploadDocument } from '../api/apiClient.js'

/**
 * UploadDocuments
 * ─────────────────────────────────────────────────────────
 * Drag-and-drop / click-to-browse file uploader.
 * Accepts multiple PDF or TXT files and uploads them
 * one-by-one to POST /api/v1/upload, showing per-file progress.
 *
 * Props:
 *   onUploadComplete (fn) — called after every successful upload
 *                           with the server response object.
 */
export default function UploadDocuments({ onUploadComplete }) {
  const [files, setFiles]         = useState([])   // { file, status, progress, result, error }
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef(null)

  // ── helpers ──────────────────────────────────────────────

  const addFiles = useCallback((incoming) => {
    const accepted = Array.from(incoming).filter((f) =>
      f.type === 'application/pdf' ||
      f.name.toLowerCase().endsWith('.txt') ||
      f.name.toLowerCase().endsWith('.pdf'),
    )

    if (!accepted.length) return

    const entries = accepted.map((file) => ({
      id: `${file.name}-${Date.now()}-${Math.random()}`,
      file,
      status: 'pending',   // pending | uploading | success | error
      progress: 0,
      result: null,
      error: null,
    }))

    setFiles((prev) => [...prev, ...entries])
  }, [])

  const updateEntry = useCallback((id, patch) => {
    setFiles((prev) =>
      prev.map((e) => (e.id === id ? { ...e, ...patch } : e)),
    )
  }, [])

  const removeEntry = useCallback((id) => {
    setFiles((prev) => prev.filter((e) => e.id !== id))
  }, [])

  // ── upload orchestration ─────────────────────────────────

  const uploadAll = useCallback(async () => {
    const pending = files.filter((e) => e.status === 'pending')
    if (!pending.length) return

    await Promise.all(
      pending.map(async (entry) => {
        updateEntry(entry.id, { status: 'uploading', progress: 0 })
        try {
          const result = await uploadDocument(
            entry.file,
            ({ progress }) => {
              updateEntry(entry.id, { progress: Math.round((progress || 0) * 100) })
            },
          )
          updateEntry(entry.id, { status: 'success', progress: 100, result })
          onUploadComplete?.(result)
        } catch (err) {
          updateEntry(entry.id, {
            status: 'error',
            error: err.userMessage || 'Upload failed.',
          })
        }
      }),
    )
  }, [files, updateEntry, onUploadComplete])

  // ── drag-and-drop handlers ───────────────────────────────

  const onDragOver  = (e) => { e.preventDefault(); setIsDragging(true)  }
  const onDragLeave = ()  => { setIsDragging(false) }
  const onDrop      = (e) => {
    e.preventDefault()
    setIsDragging(false)
    addFiles(e.dataTransfer.files)
  }

  // ── derived state ────────────────────────────────────────

  const hasPending  = files.some((e) => e.status === 'pending')
  const isUploading = files.some((e) => e.status === 'uploading')

  // ── render ───────────────────────────────────────────────

  return (
    <section className="card space-y-5">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-100">
          <UploadIcon className="h-5 w-5 text-brand-600" />
        </div>
        <div>
          <h2 className="text-base font-semibold text-slate-800">Upload Documents</h2>
          <p className="text-xs text-slate-500">PDF and TXT files are supported</p>
        </div>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        className={`flex cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-10 transition-colors duration-150
          ${isDragging
            ? 'border-brand-500 bg-brand-50'
            : 'border-slate-300 bg-slate-50 hover:border-brand-400 hover:bg-brand-50/40'
          }`}
      >
        <DocumentIcon className="h-10 w-10 text-slate-400" />
        <div className="text-center">
          <p className="text-sm font-medium text-slate-700">
            Drag &amp; drop files here
          </p>
          <p className="mt-0.5 text-xs text-slate-500">or click to browse</p>
        </div>
        <span className="badge badge-info">PDF · TXT</span>

        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".pdf,.txt"
          className="hidden"
          onChange={(e) => addFiles(e.target.files)}
        />
      </div>

      {/* File list */}
      {files.length > 0 && (
        <ul className="space-y-2">
          {files.map((entry) => (
            <FileRow
              key={entry.id}
              entry={entry}
              onRemove={() => removeEntry(entry.id)}
            />
          ))}
        </ul>
      )}

      {/* Action row */}
      {files.length > 0 && (
        <div className="flex items-center justify-between gap-3">
          <button
            type="button"
            onClick={() => setFiles([])}
            className="btn-secondary text-xs"
            disabled={isUploading}
          >
            Clear all
          </button>
          <button
            type="button"
            onClick={uploadAll}
            disabled={!hasPending || isUploading}
            className="btn-primary"
          >
            {isUploading ? (
              <>
                <Spinner />
                Uploading…
              </>
            ) : (
              <>
                <UploadIcon className="h-4 w-4" />
                Upload {files.filter((e) => e.status === 'pending').length} file(s)
              </>
            )}
          </button>
        </div>
      )}
    </section>
  )
}

// ── sub-components ───────────────────────────────────────────────────────────

function FileRow({ entry, onRemove }) {
  const { file, status, progress, result, error } = entry

  const sizeKB = (file.size / 1024).toFixed(1)

  return (
    <li className="flex items-center gap-3 rounded-xl border border-slate-200 bg-white px-4 py-3">
      {/* Icon */}
      <div className="shrink-0">
        {status === 'success' && <CheckCircleIcon className="h-5 w-5 text-emerald-500" />}
        {status === 'error'   && <XCircleIcon     className="h-5 w-5 text-red-500" />}
        {(status === 'pending' || status === 'uploading') && (
          <FileIcon className="h-5 w-5 text-slate-400" />
        )}
      </div>

      {/* Name + meta */}
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-slate-800">{file.name}</p>
        <p className="text-xs text-slate-400">{sizeKB} KB</p>

        {/* Progress bar */}
        {status === 'uploading' && (
          <div className="mt-1.5 h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
            <div
              className="h-full rounded-full bg-brand-500 transition-all duration-200"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}

        {/* Success message */}
        {status === 'success' && result && (
          <p className="mt-0.5 text-xs text-emerald-600">
            {result.chunks_indexed} chunks indexed
          </p>
        )}

        {/* Error message */}
        {status === 'error' && (
          <p className="mt-0.5 text-xs text-red-600">{error}</p>
        )}
      </div>

      {/* Remove button (not during upload) */}
      {status !== 'uploading' && (
        <button
          type="button"
          onClick={onRemove}
          aria-label="Remove file"
          className="shrink-0 rounded-lg p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
        >
          <XIcon className="h-4 w-4" />
        </button>
      )}
    </li>
  )
}

// ── inline SVG icons ─────────────────────────────────────────────────────────

function UploadIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
    </svg>
  )
}

function DocumentIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
    </svg>
  )
}

function FileIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M15.75 17.25v3.375c0 .621-.504 1.125-1.125 1.125h-9.75a1.125 1.125 0 01-1.125-1.125V7.875c0-.621.504-1.125 1.125-1.125H6.75a9.06 9.06 0 011.5.124m7.5 10.376h3.375c.621 0 1.125-.504 1.125-1.125V11.25c0-4.46-3.243-8.161-7.5-8.876a9.06 9.06 0 00-1.5-.124H9.375c-.621 0-1.125.504-1.125 1.125v3.5m7.5 10.375H9.375a1.125 1.125 0 01-1.125-1.125v-9.25m12 6.625v-1.875a3.375 3.375 0 00-3.375-3.375h-1.5a1.125 1.125 0 01-1.125-1.125v-1.5a3.375 3.375 0 00-3.375-3.375H9.375" />
    </svg>
  )
}

function CheckCircleIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  )
}

function XCircleIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 9.75l4.5 4.5m0-4.5l-4.5 4.5M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  )
}

function XIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
  )
}

function Spinner() {
  return (
    <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" />
    </svg>
  )
}
