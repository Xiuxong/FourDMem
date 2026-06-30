# FourDMem — Cognitive Agent Memory

You have a persistent memory system called FourDMem that stores your conversations and knowledge across sessions.

## What happens automatically

- Every tool interaction is auto-archived to L0 (raw evidence)
- The post-turn hook checkpoints your session automatically
- L0→L1 fact extraction runs periodically on high-salience content
- You do NOT need to call `log_turn` — it's optional

## What you should do

1. **Start of session**: Call `wake_up` to restore context
2. **Before answering**: Call `search_memory` with the user's query
3. **After search**: Call `submit_feedback` (+1 if useful, -1 if not)
4. **Important facts**: Call `save_memory` when user says "remember this"

## Memory architecture

- **L0**: Raw conversation evidence (SQLite, append-only)
- **L1**: Atomic facts with version trees (graph.json)
- **L2**: Scenario blocks (Markdown, human-readable)
- **L3**: Core persona/rules (YAML)
- **L4**: Meta-cognition observer (self-referential)

## Key tools

| Tool | When to use |
|------|-------------|
| `search_memory` | Every user query |
| `wake_up` | Session start |
| `save_memory` | Explicit "remember this" |
| `submit_feedback` | Rate search results |
| `abandon_branch` | Mark failed approach as counterfactual |
| `get_entity_context` | Inspect memory history/conflicts |
| `extract_deep` | Manually extract important facts |
| `synthesize_l2` | Combine facts into scenarios |
| `reflect_and_synthesize` | Resolve knowledge conflicts |
