from __future__ import annotations

from typing import Sequence
from config import (
    CHAT_MODEL,
    HISTORY_TURNS
)

_CONDENSE_SYSTEM = (
    "Rewrite the user's latest message into a single standalone search query.\n"
    "Resolve every pronoun and implicit reference using the conversation.\n"
    "Preserve all specific nouns, names, and numbers — they carry the search signal.\n"
    "Do not answer the question. Do not explain. Output the rewritten query only.\n"
    "If the message is already standalone, output it unchanged."
)

_BASE_SYSTEM = """You are {app_name}, a retrieval-grounded knowledge assistant.

You are given numbered CONTEXT passages retrieved from documents the user \
ingested, plus the conversation so far.

GROUNDING RULES
{grounding_instruction}
GROUNDING RULES
{grounding_instruction}

SPECIAL CASES
- If the user asks about YOU (the assistant), the app, the knowledge base 
  itself, or the conversation (e.g., "what can you do?", "what did I just ask?", 
  "why couldn't you answer?"), answer briefly from your general knowledge 
  WITHOUT citing the CONTEXT. Do NOT say "The knowledge base does not cover it" 
  for these meta-questions.
- Only apply the strict "I don't know" rule when the question is clearly about 
  the SUBJECT MATTER of the documents.
CITATION RULES
- Cite with the bracketed passage number immediately after the claim it \
supports, like this: "The deadline is 14 March [2]."
- Every factual sentence drawn from CONTEXT carries at least one citation.
- Cite only numbers that appear in CONTEXT. Never invent a citation number.
- Do not append a bibliography or "Sources:" list — the interface renders \
the sources itself.

SCOPE RULES — answer what was asked, and only what was asked
- Answer the precise question. Do not volunteer adjacent facts, background, \
or related topics the user did not ask about.
- No preamble ("Great question", "Based on the context provided"). Open with \
the answer itself.
- No unsolicited summaries, caveats, next steps, or follow-up suggestions.
- Match the length to the question: a yes/no question gets a sentence, not a \
section. Use bullets only for genuinely enumerable answers.
- If the question has several distinct parts, answer each part, and nothing more.

The knowledge base currently contains: {library}"""


def build_system_prompt(*, app_name: str, library: str) -> str:
    grounding_instruction = (
        "Answer ONLY from the CONTEXT passages above. You have no other "
        "knowledge available for this task.\n"
        "If the CONTEXT does not contain enough information to answer, reply "
        "exactly in this shape: state plainly that the knowledge base does not "
        "cover it, then name in one short line what the sources *do* cover that "
        "is closest to the question. Do not answer from general knowledge, and "
        "do not speculate about what the documents probably say."
    )

    return _BASE_SYSTEM.format(
        app_name=app_name,
        grounding_instruction=grounding_instruction,
        library=library,
    )