# Writer-output parse failures and silent truncations

Parse failures (no extractable code block or unterminated string literals) recur heavily and disproportionately affect ambitious full-rewrite prompts. They do not evaluate the underlying idea.

## How it was tried
- Code extraction failures: c0009, c0010, c0037, c0054, c0059, c0085, c0088, c0092, c0094, c0098, c0100.
- Pseudo-parse failure (emitted "code" was heavily mangled): c0006, c0052.
- c0001r2 (cvrp-s11-66566581): parse_error death. Code was completely empty.
- c0016r1/c0017 (cvrp-s11-66566581): SyntaxError: unterminated triple-quoted string literal. Diff deleted massive C blocks and truncated the file.
- c0006/c0007/c0010 (cvrp-s13-71671014): NameError/SyntaxError from failed full file rewrites. c0010 explicitly used `exec(compile(...))` wrappers, causing syntax failures. 
- c0022 (cvrp-s13-71671014): NameError (`_unpack` not defined). Emitted a snippet instead of a unified diff.
- c0002r2/c0013r2 (run cvrp-s17-78412700): Empty code string provided.
- c0010 (run cvrp-s17-78412700): NameError (`solve` not defined). Truncated diff leaving orphaned functions.
- c0001 (run cvrp-s19-83885116): SyntaxError: unterminated triple-quoted string literal. Massive deletion truncation.
- c0005 (run cvrp-s19-83885116): SyntaxError: closing parenthesis `]` does not match opening parenthesis `(`. Massive deletion truncation.
- c0002r2, c0004, c0005r1 (cvrp-s23-89572315): SyntaxError/parse_error. c0004 attempted sweep construction but truncated `_valid(c` at the end. c0005r1 tried to remove `try_intra_3opt` but deleted the entire C kernel block, truncating the file. c0002r2 provided an empty code string.
- c0025r2 (cvrp-s23-89572315): parse_error death. Output was completely empty (code block truncated). Attempted inter-route Or-opt(4).
- c0003r1 (run cvrp-s29-99479842): NameError death. Emitted a 3-line snippet instead of a full diff, resulting in `net_fwd` being undefined globally.
- c0006r2 (run cvrp-s29-99479842): parse_error death. Output was empty. Attempted bug fixes + intra-route Or-opt.
- c0022r2 (run cvrp-s31-5170619): parse_error death (empty code string).
- c0023 (run cvrp-s31-5170619): SyntaxError death. Emitted a raw snippet of C code instead of a unified diff.
- c0016 (run cvrp-s31-5170619): NameError (`net_fwd` undefined). Emitted a 2-line snippet instead of a full diff.
- c0010 (run cvrp-s31-5170619): Truncated diff leaving `_py_cross_exchange_2` orphaned and undefined.
- c0012 (run cvrp-s31-5170619): Truncated `_py_two_opt_star` definition and omitted overlay wiring.
- c0036/c0038/c0040 (run cvrp-s31-5170619): ValueError (`too many values to unpack`). Dropped `bk = ck` / `bk = len(...)` assignments while keeping 3-value returns from `_pack`.

## Why it failed
- These are generation failures, not algorithm results. Treating them as "tried and failed" silently buries untested directions.
- Writers repeatedly attempted massive structural insertions or full rewrites, leading to diff truncation.
- Snippet emission (c0003r1, c0016, c0023) leaves orphaned variables, breaking the solver globally.
- **Lethal trap:** Writers routinely rewrite `_run_c_lns` / `_pack` wrappers but forget that `_pack` returns 3 values `(seq, rend, k)`. Assigning `bseq, brend = _pack(...)` causes `ValueError`. 

## Verdict
promising to handle explicitly: a parse_error candidate's idea is OPEN, not refuted — record it in new-ideas/ instead of dropping it. Ensure complete blocks before closing diffs! DO NOT rewrite helper function signatures without full context.
