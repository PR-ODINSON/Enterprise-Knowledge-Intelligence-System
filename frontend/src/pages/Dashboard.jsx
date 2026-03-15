/**
 * Dashboard
 * ─────────────────────────────────────────────────────────
 * Full-page layout combining:
 *   Header  → collection selector + stats pill
 *   Section 1 → UploadDocuments (document ingestion)
 *   Section 2 → ChatInterface   (streaming Q&A)
 *   Section 3 → AnswerViewer    (latest RAG answer)
 *
 * New state:
 *   selectedCollection  — active collection
 *   collections         — list of all collections from GET /collections
 *   conversationId      — UUID for multi-turn memory
 */
import { useCallback, useEffect, useState } from 'react'
import AnswerViewer from '../components/AnswerViewer.jsx'
import ChatInterface from '../components/ChatInterface.jsx'
import CollectionSelector from '../components/CollectionSelector.jsx'
import UploadDocuments from '../components/UploadDocuments.jsx'
import {
  listCollections,
  listDocuments,
  startConversation,
} from '../api/apiClient.js'

export default function Dashboard() {
  const [latestResponse, setLatestResponse]       = useState(null)
  const [docStats, setDocStats]                   = useState(null)
  const [statsError, setStatsError]               = useState(false)
  const [collections, setCollections]             = useState([{ collection_id: 'default', total_documents: 0, total_vectors: 0 }])
  const [collectionsLoading, setCollectionsLoading] = useState(true)
  const [selectedCollection, setSelectedCollection] = useState('default')
  const [conversationId, setConversationId]       = useState(null)

  // Fetch collections list
  const refreshCollections = useCallback(async () => {
    try {
      setCollectionsLoading(true)
      const data = await listCollections()
      setCollections(data.collections || [])
    } catch {
      // Keep existing list
    } finally {
      setCollectionsLoading(false)
    }
  }, [])

  // Fetch per-collection doc stats
  const refreshStats = useCallback(async (coll) => {
    try {
      const data = await listDocuments(coll || selectedCollection)
      setDocStats(data)
      setStatsError(false)
    } catch {
      setStatsError(true)
    }
  }, [selectedCollection])

  // Start a conversation session on mount
  useEffect(() => {
    startConversation()
      .then(({ conversation_id }) => setConversationId(conversation_id))
      .catch(() => {/* conversation memory works without a server-issued ID */})
  }, [])

  useEffect(() => {
    refreshCollections()
    refreshStats(selectedCollection)
  }, [selectedCollection]) // eslint-disable-line

  const handleUploadComplete = useCallback(() => {
    refreshStats(selectedCollection)
    refreshCollections()
  }, [selectedCollection, refreshStats, refreshCollections])

  const handleCollectionChange = (coll) => {
    setSelectedCollection(coll)
    setLatestResponse(null)
    // Start a fresh conversation for the new collection
    startConversation()
      .then(({ conversation_id }) => setConversationId(conversation_id))
      .catch(() => {})
  }

  const currentColl = collections.find((c) => c.collection_id === selectedCollection)
  const hasDocuments = (docStats?.total_vectors_indexed ?? 0) > 0

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-brand-50/30">
      {/* ── Navbar ─────────────────────────────────────────── */}
      <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/80 backdrop-blur-md">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-3.5">
          <div className="flex items-center gap-3">
            <LogoIcon className="h-8 w-8" />
            <div>
              <h1 className="text-sm font-bold text-slate-900 leading-none">
                Enterprise Knowledge Intelligence
              </h1>
              <p className="text-[11px] text-slate-500 leading-none mt-0.5">
                Hybrid RAG · FAISS · Mistral-7B-Instruct
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Collection selector */}
            <CollectionSelector
              collections={collections}
              selected={selectedCollection}
              onChange={handleCollectionChange}
              loading={collectionsLoading}
            />

            {/* Stats pill */}
            {docStats && !statsError && (
              <div className="hidden sm:flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3.5 py-1.5 text-xs text-slate-600 shadow-sm">
                <PulseIcon className={`h-2 w-2 rounded-full ${hasDocuments ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`} />
                <span>
                  {docStats.total_documents} doc{docStats.total_documents !== 1 ? 's' : ''} ·{' '}
                  {docStats.total_vectors_indexed.toLocaleString()} chunks
                </span>
              </div>
            )}

            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="btn-secondary text-xs py-1.5 px-3"
            >
              API Docs
            </a>
          </div>
        </div>
      </header>

      {/* ── Main content ───────────────────────────────────── */}
      <main className="mx-auto max-w-5xl space-y-6 px-6 py-8">

        {/* Hero — hidden once documents are indexed */}
        {!hasDocuments && (
          <section className="card flex flex-col items-center gap-4 py-10 text-center">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-brand-500 to-violet-500 shadow-lg">
              <BrainIcon className="h-8 w-8 text-white" />
            </div>
            <div className="max-w-md space-y-1">
              <h2 className="text-xl font-bold text-slate-800">Welcome to Your Knowledge Base</h2>
              <p className="text-sm text-slate-500">
                Upload PDF or TXT documents below. Once indexed, you can ask
                natural-language questions powered entirely by a local Mistral-7B LLM.
              </p>
            </div>
            <div className="flex flex-wrap justify-center gap-2">
              {[
                'Hybrid BM25 + FAISS retrieval',
                'Cross-encoder reranking',
                'Streaming responses',
                'Multi-collection support',
                'Conversation memory',
                'Query caching',
              ].map((tag) => (
                <span key={tag} className="badge badge-info">{tag}</span>
              ))}
            </div>
          </section>
        )}

        {/* Collections overview bar */}
        {collections.length > 1 && (
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {collections.slice(0, 4).map((c) => (
              <button
                key={c.collection_id}
                type="button"
                onClick={() => handleCollectionChange(c.collection_id)}
                className={`rounded-xl border p-3 text-left transition-all ${
                  c.collection_id === selectedCollection
                    ? 'border-brand-300 bg-brand-50 shadow-sm'
                    : 'border-slate-200 bg-white hover:border-slate-300'
                }`}
              >
                <p className={`text-xs font-semibold truncate ${c.collection_id === selectedCollection ? 'text-brand-700' : 'text-slate-700'}`}>
                  {c.collection_id}
                </p>
                <p className="text-[10px] text-slate-400 mt-0.5">
                  {c.total_documents} docs · {c.total_vectors.toLocaleString()} chunks
                </p>
              </button>
            ))}
          </div>
        )}

        {/* Section 1: Upload */}
        <div>
          <SectionLabel icon={<UploadSectionIcon />} label="1 · Ingest Documents" />
          <UploadDocuments
            onUploadComplete={handleUploadComplete}
            collection={selectedCollection}
          />
        </div>

        {/* Section 2: Question */}
        <div>
          <SectionLabel icon={<QuestionSectionIcon />} label="2 · Ask a Question" />
          <ChatInterface
            onAnswer={setLatestResponse}
            hasDocuments={hasDocuments}
            collection={selectedCollection}
            conversationId={conversationId}
          />
        </div>

        {/* Section 3: Answer */}
        <div>
          <SectionLabel icon={<AnswerSectionIcon />} label="3 · Generated Answer" />
          <AnswerViewer response={latestResponse} />
        </div>

      </main>

      {/* ── Footer ─────────────────────────────────────────── */}
      <footer className="mt-12 border-t border-slate-200 bg-white py-5 text-center text-xs text-slate-400">
        Enterprise Knowledge Intelligence System · Hybrid RAG · Local LLM ·{' '}
        <a
          href="http://localhost:8000/docs"
          target="_blank"
          rel="noopener noreferrer"
          className="text-brand-500 hover:underline"
        >
          API Docs
        </a>
      </footer>
    </div>
  )
}

// ── helpers ───────────────────────────────────────────────────────────────────

function SectionLabel({ icon, label }) {
  return (
    <div className="mb-2.5 flex items-center gap-2 px-1">
      <span className="text-slate-400">{icon}</span>
      <span className="text-xs font-semibold uppercase tracking-widest text-slate-400">{label}</span>
    </div>
  )
}

// ── icons ─────────────────────────────────────────────────────────────────────

function LogoIcon({ className }) {
  return (
    <svg className={className} viewBox="0 0 32 32" fill="none">
      <rect width="32" height="32" rx="8" fill="url(#logo-grad)" />
      <defs>
        <linearGradient id="logo-grad" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
          <stop stopColor="#2563eb" />
          <stop offset="1" stopColor="#7c3aed" />
        </linearGradient>
      </defs>
      <path stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
        d="M9 10h14M9 14h10M9 18h12M9 22h8" />
    </svg>
  )
}

function BrainIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23-.693L5 14.5m14.8.8l1.402 1.402c1 1 .03 2.798-1.442 2.798H4.24c-1.47 0-2.441-1.798-1.442-2.798L4.2 15.3" />
    </svg>
  )
}

function PulseIcon({ className }) {
  return <span className={`inline-block ${className}`} />
}

function UploadSectionIcon() {
  return (
    <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
    </svg>
  )
}

function QuestionSectionIcon() {
  return (
    <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M9.879 7.519c1.171-1.025 3.071-1.025 4.242 0 1.172 1.025 1.172 2.687 0 3.712-.203.179-.43.326-.67.442-.745.361-1.45.999-1.45 1.827v.75M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9 5.25h.008v.008H12v-.008z" />
    </svg>
  )
}

function AnswerSectionIcon() {
  return (
    <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
    </svg>
  )
}
