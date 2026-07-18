Anytime CVRP: one depot (index 0), n-1 customers, identical vehicles of capacity Q.
Minimize total distance of depot-to-depot routes serving every customer once within
capacity, under a FIXED per-instance wall-clock budget (3-6 s). Write ONE Python
module (stdlib + numpy + ctypes only; no files/network) exporting exactly:

    def solve(coords, dist, demand, capacity, deadline, compile_c) -> list[list[int]]

coords: (n,2) float64; dist: (n,n) int32, CVRPLIB-rounded euclidean int(d+0.5) —
the ONLY metric scored; demand: (n,) int32 with demand[0]=0; capacity: int;
deadline: absolute time.monotonic() value — return BEFORE it (+1 s grace, later
means -inf); compile_c(c_source: str) -> ctypes.CDLL compiles arbitrary C with
gcc -O3 -march=native inside the sandbox. Everything else is yours: define C
kernels as string constants (compile once, cache in a module global), any
construction / local search / metaheuristic in Python. A faster C kernel buys
more search iterations inside the fixed budget; a smarter anytime policy (keep
the best-so-far, perturb, reoptimize) spends them better — both matter.

Minimal ctypes bridge (flat int32 arrays; more in the wiki):

    lib = compile_c("int cost(const int*d,int n,const int*t,int m){...}")
    I = ctypes.POINTER(ctypes.c_int)
    lib.cost.argtypes, lib.cost.restype = [I, ctypes.c_int, I, ctypes.c_int], ctypes.c_int
    lib.cost(dist.ctypes.data_as(I), n, tour.ctypes.data_as(I), len(tour))

Return routes as lists of customer indices 1..n-1; depot 0 must not appear; order
within a route = visit order. Validation happens OUTSIDE the sandbox: every
customer exactly once, per-route demand <= capacity — invalid or late => -inf.
Score = -mean(gap%) vs best-known cost (0 = match BKS, >0 = beat it); the seed
lands near -0.2 on train and -2 on the larger hidden X instances — the headroom
is there. Instances: CVRPLIB A/B/P (n=32-65) and hidden X (n=101-303).
