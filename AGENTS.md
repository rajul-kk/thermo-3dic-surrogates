# Agent roles

- **Opus**: plans changes, proposes ambitious ideas, and reviews whether the code and the ideas behind it are going in the right direction. Reviews every agent result before it reaches the report, and reruns the key claims itself. Owns anything destructive or outward-facing: history rewrites, pushes, Kaggle launches, messages to maintainers.
- **Sonnet (medium effort)**: work that needs judgement: investigation and debugging, building planned changes, writing new tests, scoop-check verdicts, and writing docs and report sections from results.
- **Haiku 5.5 (low effort)**: mechanical work: running existing test suites and reporting pass/fail, checking job and kernel status, downloading outputs, grep and file inventories, checking that numbers in docs match the result files, checking commits against the rules (no `notes/`, no attribution trailer, explicit paths), and first-pass web searches that return links and snippets for Sonnet to judge.
- A Haiku agent that finds anything unexpected (a failing test, a mismatched number) stops and reports; it does not fix it.

# Context management

- Compact automatically once the conversation reaches 200k-250k tokens; do not let it run past 250k.
- Before compacting, write anything a later session would need (running jobs, unpushed branches, pending decisions, result paths) to `notes/` or the project memory, so the summary does not have to carry it.
- Compact at a natural break (between tasks, after a commit), never in the middle of an edit or while waiting on a job whose ID is not recorded.
