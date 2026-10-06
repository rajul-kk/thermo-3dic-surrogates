# Agent roles

- **Opus**: plans changes, proposes ambitious ideas, and reviews whether the code and the ideas behind it are going in the right direction.
- **Sonnet (medium effort)**: handles the simpler work automatically: building already-planned changes, writing and running almost all tests, simple reviews, and writing summaries and docs and simple question answer statements.

# Context management

- Compact automatically once the conversation reaches 200k-250k tokens; do not let it run past 250k.
- Before compacting, write anything a later session would need (running jobs, unpushed branches, pending decisions, result paths) to `notes/` or the project memory, so the summary does not have to carry it.
- Compact at a natural break (between tasks, after a commit), never in the middle of an edit or while waiting on a job whose ID is not recorded.
