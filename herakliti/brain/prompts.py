"""Prompt templates — the highest-leverage text in the project.

Every token here is paid for on every turn, at ~40ms of prefill each on this machine.
A 300-token system prompt is ~12 seconds before the user sees anything. So these are
deliberately terse: where a longer prompt would buy a marginal quality win, it is not
worth the wall-clock.

The grounded prompt's real job is to make refusing *cheap*. A 4B model under pressure
to be helpful will invent a fluent, confident answer rather than admit a gap — so the
refusal is framed here as the correct outcome and given exact wording, rather than
left as an apologetic fallback the model will avoid.

Imports nothing heavy — safe to import from anywhere.
"""

from __future__ import annotations

from typing import Sequence

from herakliti.knowledge.types import Chunk

# The refusal clause is worded, not quoted, and that is deliberate — measured on Qwen3.5-4B:
# pinning a literal ("reply exactly \"The sources don't say.\"") makes the model emit that
# exact string whatever language it was asked in, because "exactly" beats "in the user's
# language". Quoting the Albanian leaked Albanian into English answers (2/4); quoting the
# English did the reverse (3/4). Naming the *idea* instead scored 12/12 across both
# languages, and costs fewer tokens.
#
# "no information in the sources" is also not arbitrary: it steers both languages onto
# phrasing `grounding.is_refusal` already matches ("no information" / "nuk ka informacion"),
# which is what tells an honest refusal apart from a hallucination when scoring confidence.
# Reword this and check that function still fires.
SYSTEM_GROUNDED = """You are Herakliti. Answer only from the numbered context.

- Reply in the same language as the question. Always.
- Use the context and nothing else. Do not use your own knowledge.
- Mark every fact with the block it came from: [1], [2].
- If the context does not answer the question, reply that there is no information in the
  sources. Refusing is a correct answer and a job well done — never guess, never fill a
  gap from memory.
- Copy names, numbers and dates exactly as written. Keep to 1-3 sentences."""

SYSTEM_CHAT = """You are Herakliti, a local AI assistant running privately on the user's \
own machine — no cloud, no tracking. For factual questions you look things up and cite \
your sources.
Be brief, warm and direct. Reply in the user's language.
If you are not sure of a fact, say so rather than guess."""

# For work the model does from its OWN skill, not from sources: arithmetic, reasoning,
# writing, translating, summarising text the user pasted. Here "answer only from context"
# would be actively wrong — there is no context, the whole point is the model's own ability.
# Kept terse (prefill tax) but with the two clauses that matter on a 4B: compute step by
# step (it drops operations otherwise) and give the final result plainly at the end.
# Chain-of-thought is load-bearing here, not decoration: measured on Qwen3.5-4B, telling it
# to "just give the result" for arithmetic produced 30 for 5+3-3*5 and 72 for 12*(3+4) —
# both wrong. The same model that works step by step gets -7 and 84. A small model computes
# by writing the steps, so we ask for them (brief, plain text) and never for the bare answer.
SYSTEM_REASON = """You are Herakliti, a sharp, capable assistant running locally.
Answer directly using your own knowledge and reasoning — there are no sources to quote here.
Write plain text only: no LaTeX, no markdown, no symbols like \\times or $ — use x, *, / and =.
For maths, work through it in short steps (this is how you get it right), then put the final
answer on its own line. Do not skip the steps.
For writing, translating or summarising, do exactly what is asked, nothing more.
Be correct and concise. Reply in the user's language."""


def build_context(chunks: Sequence[Chunk]) -> str:
    """Numbered context blocks, as tight as they go.

    The whitespace collapse is not cosmetic: Wikipedia extracts arrive full of newlines
    and runs of spaces, and the tokenizer charges for every one of them. Squeezing them
    out is a few percent of prefill for free.

    Block numbers are 1-based to match the [1], [2] citations SYSTEM_GROUNDED asks for.
    """
    return "\n\n".join(
        f"[{i}] {c.title}\n{' '.join(c.text.split())}" for i, c in enumerate(chunks, 1)
    )


# Sent as a lone user message (see grounding.llm_support), so it carries its own
# instructions. Judges support only — an answer can be perfectly grounded and still wrong,
# which is the sources' problem, not ours. The refusal clause stops the checker from
# flagging a correct "the context is silent" answer as unsupported.
GROUNDING_CHECK = """Does the context support the answer? Judge only support, not truth.
Grounded means: every claim in the answer is stated in the context.
An answer that declines because the context is silent is grounded.

Context:
{context}

Answer:
{answer}"""

# --------------------------------------------------------------------------
# Templates for the routing/planning layer. All are used with llm.json(), so they
# describe *content* only — llama.cpp's grammar enforces the shape, and repeating
# "reply with JSON" at it would just be tokens we pay for twice.
# --------------------------------------------------------------------------

# Two-way on purpose. FACT is not offered: it is decided by rules that structurally
# guarantee an entity to look up, and anything reaching this prompt matched none of them.
# Measured on Qwen3.5-4B over the fragments that actually reach the fallback, offering
# "fact" as a third option scored 4/9 — it labelled every bare name ("Tirana",
# "Ismail Kadare") a fact, each one a wasted ~3s Wikidata miss. Dropping the option
# scored 8/9 and costs 12 fewer tokens. The lone miss errs toward retrieve, which is the
# direction we want to be wrong in.
ROUTER_SYSTEM = """Classify the message.
chat = greeting, thanks, small talk, or a question about you.
retrieve = anything else: any request for real information about the world.
When unsure, answer retrieve."""

ROUTER_SCHEMA = {
    "type": "object",
    "properties": {"route": {"type": "string", "enum": ["chat", "retrieve"]}},
    "required": ["route"],
}

# The last two clauses each fix a failure caught on Qwen3.5-4B. Without "do not replace it
# with an earlier question" the model answers the *previous* turn — "what about its
# population?" came back as "What is the capital of Albania?", pronoun resolved, question
# thrown away. And there is deliberately no "if it already stands alone, return it
# unchanged" escape hatch: planner.rewrite only calls this once it has detected a pronoun,
# so offering that option just invites the model to no-op (3/5 -> 5/5 without it).
REWRITE_SYSTEM = """Rewrite the follow-up question so it stands alone without the conversation.
Replace its pronouns with the name they refer to. Keep everything else it asks for.
Keep its language. Do not answer it. Do not replace it with an earlier question."""

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}},
    "required": ["question"],
}

DECOMPOSE_SYSTEM = """Split the question into the ordered sub-questions needed to answer it.
Each must stand alone: no pronouns referring to another sub-question.
Most questions need only one — return it unchanged then. Never more than 3.
Keep the original language."""

DECOMPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 3,
        }
    },
    "required": ["questions"],
}
