# Writer-output parse failures ate the run's most novel ideas

5 of 58 candidates (~9%) never reached evaluation because the writer's output could not be parsed into code — and they disproportionately carried the run's most original orchestration ideas.

## How it was tried
- c0009, c0010, c0037, c0054: `parse_error`, code length 0 — an idea/reasoning text existed but no extractable code block. c0037 and c0054 held the untested persistent-C-state / final-polish-only ideas (now in new-ideas/persistent-c-search-state.md); c0010's recreate tie-break is in new-ideas/unevaluated-variants.md.
- c0006: pseudo-parse failure — the emitted "code" was 277 bytes containing `exec(compile('Idea: Rewrite the C engine...'` — the writer wrapped its own prose as code; SyntaxError at line 1.
- Pattern: parse failures cluster on ambitious full-rewrite prompts (novelty 1.0 entries), where the writer emits long prose + partial code and the extractor finds nothing.

## Why it failed
- These are generation failures, not algorithm results: the ideas were never refuted. Treating them as "tried and failed" (as a score of -inf suggests) silently buries untested directions.

## Verdict
promising to handle explicitly: a parse_error candidate's idea is OPEN, not refuted — record it in new-ideas/ instead of dropping it. Any future writer picking up c0054's idea starts from zero evidence against it. When re-attempting an ambitious rewrite prompt, emit the code block first and keep prose out of it.
