# ctypes / C kernel pattern

## Worked example — compile once, cache the CDLL
```c
#include <time.h>
static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec;
}
#define D(i, j) dist[(long)(i) * n + (j)]
int route_cost(const int *dist, int n, const int *route, int len) {
    int total = 0, prev = 0;
    for (int i = 0; i < len; i++) { total += D(prev, route[i]); prev = route[i]; }
    return total + D(prev, 0);
}
void two_opt(const int *dist, int n, int *tour, int len, double tlimit) {  /* in-place */
    double tend = now() + tlimit;
    for (int improved = 1; improved && now() < tend; ) {
        improved = 0;
        for (int i = 0; i < len - 1; i++)
          for (int j = i + 1; j < len; j++) {
            int a = i ? tour[i - 1] : 0, b = j == len - 1 ? 0 : tour[j + 1];
            if (D(a, tour[j]) + D(tour[i], b) >= D(a, tour[i]) + D(tour[j], b)) continue;
            for (int x = i, y = j; x < y; x++, y--) { int t=tour[x]; tour[x]=tour[y]; tour[y]=t; }
            improved = 1;
          }
    }
}
```
## Python side
```python
I = ctypes.POINTER(ctypes.c_int)
_lib = None                                    # module global, compiled once
def get_lib(compile_c):
    global _lib
    if _lib is None:
        _lib = compile_c(C_SOURCE)
        _lib.route_cost.argtypes = [I, ctypes.c_int, I, ctypes.c_int]
        _lib.route_cost.restype = ctypes.c_int
        _lib.two_opt.argtypes = [I, ctypes.c_int, I, ctypes.c_int, ctypes.c_double]
        _lib.two_opt.restype = None
    return _lib
def polish(compile_c, dist, route, tlimit):
    lib = get_lib(compile_c)
    dist = np.ascontiguousarray(dist, dtype=np.int32)
    route = np.ascontiguousarray(route, dtype=np.int32)   # mutated in place
    dp, rp, n, m = dist.ctypes.data_as(I), route.ctypes.data_as(I), dist.shape[0], len(route)
    lib.two_opt(dp, n, rp, m, ctypes.c_double(tlimit))
    return route, lib.route_cost(dp, n, rp, m)
```

## Pitfalls
- **int32, C-contiguous only**: `np.ascontiguousarray(a, dtype=np.int32)` — a
  plain int64 array (numpy default) reinterprets as garbage under `c_int*`,
  silently, no crash. Index with `(long)i*n+j`, not `i*n+j` — overflow insurance.
- **Bind arrays to a name that outlives the call**: an inline temporary
  (`f(x().ctypes.data_as(I))`) can be freed mid-call — use-after-free.
- **Never print from a kernel**: the harness parses stdout for the JSON result
  line; stray output risks corrupting the parse.
- **VLAs/malloc fine, `-lm` linked**, no other libs, no `-fopenmp`, 1 core;
  compile once per process and cache the `CDLL` (anytime.md).
