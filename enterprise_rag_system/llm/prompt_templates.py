"""
Prompt Templates Module
Defines and builds the prompt strings used throughout the RAG generation pipeline.

Two template variants are provided:
  - INSTRUCTION_FORMAT  — Mistral / Llama instruction-tuned format using [INST] tags
  - PLAIN_FORMAT        — A generic format suitable for base/completion models
"""

# ---------------------------------------------------------------------------
# Raw template strings
# ---------------------------------------------------------------------------

# Instruction-tuned format (Mistral-7B-Instruct, Llama-2-Chat, etc.)
# The [INST] / [/INST] tags signal the boundaries of the user instruction.
INSTRUCTION_PROMPT_TEMPLATE = """\
<s>[INST] You are a helpful enterprise knowledge assistant. Answer the user's \
question based strictly on the provided context. Do not use any external knowledge \
or information outside the context below.

Context:
{context}
{history_block}
Question: {question}

Provide a clear, factual, and concise answer based only on the context above. \
If the context does not contain enough information to answer the question, respond \
with: "I don't have enough information in the provided documents to answer this." [/INST]"""

# Plain completion format for base (non-instruction-tuned) models
PLAIN_PROMPT_TEMPLATE = """\
You are an assistant answering questions using provided context.

Context:
{context}

Question:
{question}

Answer using only the context. If the context does not contain sufficient \
information to answer the question, say "I don't have enough information in \
the provided documents to answer this question."

Answer:"""

# Summary prompt
SUMMARY_PROMPT_TEMPLATE = """\
<s>[INST] Please provide a concise summary of the following document excerpt. \
Focus on the key points and main ideas.

Text:
{text}

Summary: [/INST]"""


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------

def build_rag_prompt(
    context: str,
    question: str,
    use_instruction_format: bool = True,
    history: str = "",
) -> str:
    """
    Construct a complete RAG prompt from retrieved context and a user question.

    Args:
        context:                The retrieved document context.
        question:               The user's natural-language question.
        use_instruction_format: ``True`` for Mistral [INST] format.
        history:                Optional formatted conversation history string.

    Returns:
        A fully formatted prompt string ready for LLM inference.
    """
    template = (
        INSTRUCTION_PROMPT_TEMPLATE
        if use_instruction_format
        else PLAIN_PROMPT_TEMPLATE
    )
    history_block = (
        f"\nPrevious conversation:\n{history}\n"
        if history.strip() else ""
    )
    return template.format(
        context=context,
        question=question,
        history_block=history_block,
    )


def build_summary_prompt(text: str) -> str:
    """
    Build a prompt for summarising a piece of text.

    Args:
        text: The document excerpt to summarise.

    Returns:
        Formatted summary prompt string.
    """
    return SUMMARY_PROMPT_TEMPLATE.format(text=text)
