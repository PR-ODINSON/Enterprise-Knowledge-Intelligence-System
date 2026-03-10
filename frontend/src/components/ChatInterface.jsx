import { useCallback, useEffect, useRef, useState } from 'react'
import { queryDocuments } from '../api/apiClient.js'

/**
 * ChatInterface
 * ─────────────────────────────────────────────────────────
 * Displays a scrollable conversation history and a text input
 * for submitting questions to the RAG pipeline.
 *
 * Each message is stored locally as:
 *   { id, role: 'user'|'assistant', text, timestamp }
 *
 * Props:
 *   onAnswer (fn) — called with the full API response object
 *                   so siblings (AnswerViewer) can render details.
 *   hasDocuments (bool) — whether any documents have been indexed.
 */
export default function ChatInterface({ onAnswer, hasDocuments = true }) {
  const [messages, setMessages]   = useState([])
  const [question, setQuestion]   = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError]         = useState(null)
  const bottomRef  = useRef(null)
  const inputRef   = useRef(null)

  // Scroll to the latest message whenever messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  // ── submit handler ───────────────────────────────────────

  const handleSubmit = useCallback(
    async (e) => {
      e?.preventDefault()
      const trimmed = question.trim()
      if (!trimmed || isLoading) return

      setError(null)
      setQuestion('')

      // Append user message optimistically
      const userMsg = {
        id: Date.now(),
        role: 'user',
        text: trimmed,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, userMsg])
      setIsLoading(true)

      try {
        const data = await queryDocuments(trimmed)

        const assistantMsg = {
          id: Date.now() + 1,
          role: 'assistant',
          text: data.answer,
          timestamp: new Date(),
          metadata: {
            chunks: data.retrieved_chunks,
            processingTime: data.processing_time_seconds,
          },
        }
        setMessages((prev) => [...prev, assistantMsg])
        onAnswer?.(data)
      } catch (err) {
        setError(err.userMessage || 'Failed to get an answer. Please try again.')
        // Remove the optimistic user message on error so the user can retry
        setMessages((prev) => prev.filter((m) => m.id !== userMsg.id))
        setQuestion(trimmed)
      } finally {
        setIsLoading(false)
        inputRef.current?.focus()
      }
    },
    [question, isLoading, onAnswer],
  )

  // Submit on Enter (Shift+Enter inserts newline)
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const clearConversation = () => {
    setMessages([])
    setError(null)
  }

  // ── render ───────────────────────────────────────────────

  return (
    <section className="card flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-violet-100">
            <ChatIcon className="h-5 w-5 text-violet-600" />
          </div>
          <div>
            <h2 className="text-base font-semibold text-slate-800">Ask a Question</h2>
            <p className="text-xs text-slate-500">Powered by Mistral-7B-Instruct</p>
          </div>
        </div>
        {messages.length > 0 && (
          <button
            type="button"
            onClick={clearConversation}
            className="btn-secondary text-xs"
          >
            Clear
          </button>
        )}
      </div>

      {/* No-documents notice */}
      {!hasDocuments && (
        <div className="flex items-start gap-2.5 rounded-xl border border-amber-200 bg-amber-50 p-3.5">
          <InfoIcon className="mt-0.5 h-4 w-4 shrink-0 text-amber-500" />
          <p className="text-xs text-amber-700">
            No documents indexed yet. Upload documents above to enable question answering.
          </p>
        </div>
      )}

      {/* Conversation scroll area */}
      <div className="flex max-h-72 min-h-[120px] flex-col gap-3 overflow-y-auto pr-1">
        {messages.length === 0 && hasDocuments && (
          <div className="flex flex-1 flex-col items-center justify-center py-8 text-center">
            <SparklesIcon className="mb-2 h-8 w-8 text-slate-300" />
            <p className="text-sm text-slate-400">
              Ask anything about your uploaded documents.
            </p>
          </div>
        )}

        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}

        {/* Thinking indicator */}
        {isLoading && (
          <div className="flex items-start gap-2.5">
            <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-violet-100">
              <BotIcon className="h-4 w-4 text-violet-600" />
            </div>
            <div className="rounded-2xl rounded-tl-sm bg-slate-100 px-4 py-2.5">
              <ThinkingDots />
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Error banner */}
      {error && (
        <div className="flex items-start gap-2.5 rounded-xl border border-red-200 bg-red-50 p-3.5">
          <AlertIcon className="mt-0.5 h-4 w-4 shrink-0 text-red-500" />
          <p className="text-xs text-red-700">{error}</p>
        </div>
      )}

      {/* Input area */}
      <form onSubmit={handleSubmit} className="flex items-end gap-2">
        <textarea
          ref={inputRef}
          rows={2}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            hasDocuments
              ? 'Type your question and press Enter…'
              : 'Upload documents first…'
          }
          disabled={!hasDocuments || isLoading}
          className="input resize-none leading-relaxed"
        />
        <button
          type="submit"
          disabled={!question.trim() || isLoading || !hasDocuments}
          className="btn-primary shrink-0 self-end"
          aria-label="Send question"
        >
          {isLoading ? <Spinner /> : <SendIcon className="h-4 w-4" />}
        </button>
      </form>

      <p className="text-center text-[11px] text-slate-400">
        Press <kbd className="rounded border border-slate-200 bg-slate-100 px-1 py-0.5 font-mono text-[10px]">Enter</kbd> to send ·
        <kbd className="ml-1 rounded border border-slate-200 bg-slate-100 px-1 py-0.5 font-mono text-[10px]">Shift+Enter</kbd> for new line
      </p>
    </section>
  )
}

// ── sub-components ───────────────────────────────────────────────────────────

function MessageBubble({ message }) {
  const isUser = message.role === 'user'

  return (
    <div className={`flex items-start gap-2.5 ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div
        className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-full
          ${isUser ? 'bg-brand-100' : 'bg-violet-100'}`}
      >
        {isUser
          ? <UserIcon className="h-4 w-4 text-brand-600" />
          : <BotIcon  className="h-4 w-4 text-violet-600" />
        }
      </div>

      {/* Bubble */}
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm leading-relaxed
          ${isUser
            ? 'rounded-tr-sm bg-brand-600 text-white'
            : 'rounded-tl-sm bg-slate-100 text-slate-800'
          }`}
      >
        <p className="whitespace-pre-wrap">{message.text}</p>
        {message.metadata?.processingTime != null && (
          <p className={`mt-1.5 text-[11px] ${isUser ? 'text-brand-100' : 'text-slate-400'}`}>
            {message.metadata.processingTime.toFixed(2)}s
          </p>
        )}
      </div>
    </div>
  )
}

function ThinkingDots() {
  return (
    <div className="flex items-center gap-1 py-0.5">
      {[0, 1, 2].map((i) => (
        <span
          key={i}
          className="inline-block h-2 w-2 animate-bounce rounded-full bg-slate-400"
          style={{ animationDelay: `${i * 0.15}s` }}
        />
      ))}
    </div>
  )
}

// ── icons ────────────────────────────────────────────────────────────────────

function ChatIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M8.625 12a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H8.25m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0H12m4.125 0a.375.375 0 11-.75 0 .375.375 0 01.75 0zm0 0h-.375M21 12c0 4.556-4.03 8.25-9 8.25a9.764 9.764 0 01-2.555-.337A5.972 5.972 0 015.41 20.97a5.969 5.969 0 01-.474-.065 4.48 4.48 0 00.978-2.025c.09-.457-.133-.901-.467-1.226C3.93 16.178 3 14.189 3 12c0-4.556 4.03-8.25 9-8.25s9 3.694 9 8.25z"/>
    </svg>
  )
}

function SendIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
    </svg>
  )
}

function UserIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
    </svg>
  )
}

function BotIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M9.75 3.104v5.714a2.25 2.25 0 01-.659 1.591L5 14.5M9.75 3.104c-.251.023-.501.05-.75.082m.75-.082a24.301 24.301 0 014.5 0m0 0v5.714c0 .597.237 1.17.659 1.591L19.8 15.3M14.25 3.104c.251.023.501.05.75.082M19.8 15.3l-1.57.393A9.065 9.065 0 0112 15a9.065 9.065 0 00-6.23-.693L5 14.5m14.8.8l1.402 1.402c1 1 .03 2.798-1.442 2.798H4.24c-1.47 0-2.441-1.798-1.442-2.798L4.2 15.3" />
    </svg>
  )
}

function SparklesIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round"
        d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zm8.906-9.5L18 6.75l-.719-1.294A3.375 3.375 0 0015 3.964V3.75A3.375 3.375 0 0011.625.375h-.75A3.375 3.375 0 007.5 3.75v.214a3.375 3.375 0 00-2.281 1.492L4.5 6.75l-.719-1.294A3.375 3.375 0 001.5 4.25V3.75A3.375 3.375 0 00-1.875.375h-.75" />
    </svg>
  )
}

function InfoIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M11.25 11.25l.041-.02a.75.75 0 011.063.852l-.708 2.836a.75.75 0 001.063.853l.041-.021M21 12a9 9 0 11-18 0 9 9 0 0118 0zm-9-3.75h.008v.008H12V8.25z" />
    </svg>
  )
}

function AlertIcon({ className }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z" />
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
