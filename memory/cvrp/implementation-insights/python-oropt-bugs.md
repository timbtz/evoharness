# Critical Bugs in the Python Or-opt / Segment-Shift Overlays

The incumbent Python overlay contains two silent bugs that repeatedly trap writers. Both are in `_py_segment_shift` or related overlay functions. Writers must apply both fixes silently as a baseline, but neither will yield algorithmic gradient.

## Bug 1: Forward Segment-Shift NameError (`net` vs `net_fwd`)
- **Location:** `_py_segment_shift`, forward-insertion gain comparison.
- **Code:** The diff has `if net_fwd > best_net:` correctly, but then immediately sets `best_net = net` (using undefined `net` instead of `net_fwd`).
- **Impact:** If an improving forward segment relocation is found, the overlay crashes with a NameError instead of applying the move. Because the branch only triggers on improving moves, it can lie dormant for entire runs if the search trajectory doesn't find an improvement, making it look "safe".
- **Fix:** `best_net = net_fwd; ...`

## Bug 2: Dead Code / Unreachable Time Break
- **Location:** `_py_segment_shift` outer loop, immediately after `route_i = routes[ri]; break`.
- **Code:** 
  ```python
  route_i = routes[ri]
  break
  if time.monotonic() > t_end:
      break
  ```
- **Impact:** The `break` statement terminates the loop before the time check, leaving the `if time.monotonic()` block completely unreachable. This causes `_py_segment_shift` to ignore its time budget if the route list is massive, risking timeout and SA starvation.

## Verdict
**Exhausted as algorithmic levers.** In runs across cvrp-s11 to cvrp-s31, over 10 candidates focused entirely on fixing Bug 1 (e.g., c0004r2, c0007, c0009r1, c0015, c0024r1, c0025, c0027r1) with high hopes of unlocking "silently disabled" improvements. Fixing the bugs correctly **does not improve the score** (c0024r1 train -0.157, c0007 train -0.097 vs baselines ~-0.075 / -0.094). 
**Verdict for future writers:** If you are touching the Python overlay, apply both fixes silently as a baseline, but **do not propose "fixing the segment-shift bug" as your main idea**. It provides zero algorithmic gradient.
