# QCBB-LABS: algorithm

QCBB (quantum-classical hybrid branch and bound, arXiv 2511.19501) for the QOBLIB LABS instances
(labs002-labs100). Part A states the framework problem-agnostically, with one correctness
theorem (Theorem A). Part B is the LABS instance and proves that it satisfies the theorem's
conditions (i)-(v). Every component used in the proof of correctness (bounds used for pruning,
propagation, symmetry reduction, stopping, the reported gap, independence from the sampler) has
a complete proof in Section 3 or an exact literature reference. Numbers are labeled MEASURED
(our measurements, hardware and script named), ESTIMATE (derived from measured numbers and
stated assumptions) or EMPIRICAL (observed on prototypes or taken from the literature, no
proof). Statements without a proof steer only the heuristic side (branching variable, sampler
angles); they are collected in Section 6.

What the quantum part does: the sampler is called at every node of the tree. It supplies the
incumbent UB (which sets the proof threshold) and the conflict values that choose the branching
variable. The optimality certificate is the bound machinery and holds for every sampler
(Theorem A). Leaves also update UB. For instances that are not proven within the budget, UB is
the reported answer and sets the gap. The quantum sampler is fixed-angle QAOA (B.7); simulated
annealing is the other backend behind the same interface, and Section 4.8 compares the two end to end.

---------------------------------------------------------------------------------------------------

## Part A. The QCBB framework (problem-agnostic)

### A.1 Objects

| QCBB object | Generic form |
|---|---|
| Master problem (MP) | min f(x) over x in X = {-1,+1}^n, f integer-valued. |
| Master Hamiltonian (MH) | diagonal H with H(sigma) = f(sigma) on basis states (QCBB: penalty form of a BLP; here: any polynomial in the spins). |
| Node | N = (Phi, U): fixed set Phi with values, free set U, F = \|U\|. Completions comp(N) = the x agreeing with the fixed values. |
| Subproblem construction (eqs. 10-16) | node Hamiltonian H_node(z) = f(x(Phi, z)), z in {-1,+1}^F, by substituting the fixed values into H; constant tracked. |
| Sampler (3.1) | returns bitstrings z of the free variables with counts omega; each is completed and its f recomputed exactly before use. |
| Incumbent | UB = f(x_UB), the best value seen, always attained. |
| Constraints | a finite family g_j(x) <= b_j (or = b_j) per node that every solution of interest satisfies (QCBB: rows of Ax = b). |
| Conflict values (eqs. 5-9) | V_jl = [g_j violated by sample l]; nu_j = (1/q) sum_l V_jl omega_l; P_ji = [variable i appears in constraint j]; gamma = nu P; k* = argmax over a candidate set of free variables. |
| Lower bound (3.4) | LB(N) <= f(x) for all x in comp(N); monotone by max with the parent (eq. 19), valid since comp(child) is a subset of comp(parent). |
| Lattice and threshold | f(X) is contained in f_0 + Delta Z. Then f(x) < UB implies f(x) <= T := UB - Delta. Prune when LB > T (3.5 with the lattice step; Delta = 1 for an integer objective). |
| Propagation (3.2.4), infeasibility (App. C) | maps (N, T) to "infeasible" or to a sub-node N' such that every x in comp(N) with f(x) <= T lies in comp(N'); "infeasible" only if no such x exists. |
| Symmetry (optional) | a group G with f(gx) = f(x) and a set K meeting every orbit; a rule may discard N only if comp(N) and K are disjoint. |
| Branching (3.2) | children x_i = +1 and x_i = -1 of a free variable i; they partition comp(N). |
| Node selection (3.3) | lowest bound first (QCBB 3.3); ties go to the deeper node, then to the older one. |
| Stopping (3.6) | proof when the lowest open key exceeds T (3.6.1); node or time limit with the open-node bound and gap (3.6.2). A completed search at a lattice threshold T' certifies min f >= T' + Delta (Theorem A(d)); the driver runs only T = UB - Delta. |

### A.2 Node loop (generic)

One loop evaluates every node in the same way (QCBB Sec. 3, Fig. 2). The open nodes are kept in
a priority queue keyed by their lower bound; the master is the first node. Each iteration takes
the open node N with the lowest key and evaluates it:
  1. the symmetry rule and propagation until no spin changes at T = UB - Delta; a node with
     no completion in K or with no completion of value <= T is dropped;
  2. bound: LB(N); if LB(N) > T prune;
  3. leaf (U empty): f exact; f < UB -> UB := f;
  4. otherwise sample H_N with the run's sampler (the same backend for every node of a run);
     the best sample becomes the incumbent if it beats UB;
  5. branching: the children of the master are the class roots (symmetry reduction, A.1); every
     other node branches on k* = argmax gamma over the candidate set, with gamma from the node's
     own samples (eqs. 5-9; nu = 1 if error detection rejected every shot). Each child enters the queue
     with key max(LB(N), LB(child)) unless that exceeds T.
Before the first incumbent T is +infinity in effect. The loop ends when the lowest key exceeds T
(UB is optimal) or at a node or time limit (UB with a certified gap).
Trace and primal-dual integral. After every node the node loop records (wall
time t, UB(t), LB(t)) whenever UB or LB changed. UB(t) is undefined before the first incumbent.
LB(t) is min(UB, lowest key in the queue) rounded up to the lattice f_0 + Delta Z, the open-node
bound A(e) with (iv). The primal-dual gap is gamma(t) = 1 if UB or LB is undefined, 0 if
UB = LB, else (UB - LB) / max(|UB|, |LB|); it is a step function between records. The
primal-dual integral PDI = integral of gamma(t) over [0, t_end] (Berthold, Oper. Res. Lett. 41
(2013) 611-614) is reported with every run, together with the sampler time (sum of the wall
times of the sampler calls). Backends are compared on the same settings and seeds, never mixed
inside a run.

### A.3 Theorem A (correctness of the framework)

Assume
 (i)  the bound is valid: LB(N) <= f(x) for every node N and every x in comp(N);
 (ii) propagation removes only completions with f > T, and returns "infeasible" only when no
      completion has f <= T;
 (iii) the symmetry rule discards a node only if no completion lies in K, where K meets every
      orbit of a group under which f is invariant;
 (iv) f(X) is contained in f_0 + Delta Z and all thresholds are on this lattice;
 (v)  the two children of a node partition its completions.
Then, for every sampler:
 (a) the search tree is finite and each node is evaluated once;
 (b) if some x has f(x) <= T, the search at threshold T reaches a leaf with f <= T (a search
     that does not stop at its first leaf reaches the orbit-mate x' of x in K, f(x') = f(x));
 (c) if the search at T = UB - Delta ends with no leaf, UB = min f;
 (d) if a search at a lattice threshold T' ends with no leaf, min f >= T' + Delta;
 (e) during a search at T, with O the set of open nodes (created, not yet evaluated; an open
     node carries the key max(parent bound, own bound)),
     min f >= min(T + Delta, min over N in O of LB(N));
 (f) LB_cert = max(highest completed T' + Delta, bound (e)) satisfies LB_cert <= min f <= UB, so
     gap = (UB - LB_cert)/UB is valid.
Sample data enter only as incumbent candidates (re-evaluated exactly) and as the branching
variable; none of (a)-(f) depends on them.

Proof. (a) Every branching fixes at least one variable and propagation only fixes variables, so
the depth is at most n; each node has at most two children. (b) Induction on depth. Let f(x) <= T
and let x' in K be an orbit-mate; f(x') = f(x) <= T by invariance. The root contains x'. If a node
N contains x', it is not pruned by the bound ((i): LB(N) <= f(x') <= T), not removed by
propagation ((ii): x' has f <= T, so x' in comp(N') and N' is not infeasible), not discarded by the
symmetry rule ((iii): x' in comp(N) and in K), and by (v) exactly one child contains x'. So x' is
a completion of a surviving node at every depth, hence of a leaf. (c) UB is attained. If some x had
f(x) < UB then f(x) <= UB - Delta = T by (iv), and (b) would give a leaf. (d) By (b) no x has
f(x) <= T', and by (iv) the next lattice value is T' + Delta. (e) Let f(x) <= T and x' as in (b).
Now x' is either a completion of an open node N, and then f(x) = f(x') >= LB(N) by (i) (the parent's
bound is valid for an unevaluated child because its completions are a subset of the parent's),
or of an already closed subtree; a closed subtree contains no completion with f <= T except leaves
already seen, whose values set UB > T, so f(x) > T and f(x) >= T + Delta by (iv). (f) follows from
(d), (e) and UB = f(x_UB). The last sentence holds because (i)-(v) do not refer to samples, (b)
holds for every choice of branching variable and node order, and (c)-(f) hold for every attained
UB. ∎

### A.4 The original instance

For a binary linear program min c^T x, Ax = b, with integer c, QCBB's instance is: a quadratic
Ising master Hamiltonian with penalty M; the certified form min H >= W - 2 z_SDP of the Max-Cut
bound as (i) (eq. 17 with the Goemans-Williamson value z_GW holds in expectation, since
Theorem 1 of QCBB App. B, which restates the Goemans-Williamson bound for negative weights,
bounds E[z_GW]); variable bound propagation on Ax = b (3.2.4) and the infeasibility test
LB + C >= M (App. C) as (ii); the trivial group as (iii); Delta = 1 as (iv); binary branching as
(v). No new claim is made about that instance.

---------------------------------------------------------------------------------------------------

## Part B. The LABS instance

Notation. s in {-1,+1}^n, C_k(s) = sum_{i=1}^{n-k} s_i s_{i+k}, E(s) = sum_{k=1}^{n-1} C_k^2.
Delta = 8 if n = 2 (mod 4), else 4 (Lemma 4). T = UB - Delta.

### B.1 Objects

| QCBB object | LABS version |
|---|---|
| Master problem (MP) | min E(s) over s in {-1,1}^n. No linear rows. |
| Master Hamiltonian (MH) | H(sigma) = sum_k C_k(sigma)^2. 4-local Ising on n qubits, integer coefficients, no big M, no penalty. |
| Node Hamiltonian | H_node(z) = sum_k (c_k + sum_i b_ki z_i + sum z_i z_{i+k})^2 on F qubits (B.2). |
| Constraints | lag budgets C_k(s)^2 <= beta_k, k = 1..n-1, beta_k = T - LB_{-k}(node) (B.2). |
| Samples | bitstrings of the F free spins, counts; SA or fixed-angle QAOA behind one `Sampler` interface (B.7). |
| Incumbent | best exact E over all completions seen (samples, leaves). |
| Lower bound LB | per-lag bound from the exact ranges R_k plus the pair congruence mod 4 (B.3). Integer. |
| Prune test | LB > T (= UB - Delta). |
| Infeasibility | a budget set A_k empty, or a spin forced to both values (B.4). |
| Symmetry | group of order 8; K = {s_1 = s_2 = +1} and the reversal lex rule on class roots (B.5). |
| Stopping 3.6.1 | the lowest open key exceeds T = UB - Delta. |
| Stopping 3.6.2 | the open-node bound (A(e)) rounded to the lattice; gap on energy. |

### B.2 Master Hamiltonian, subproblem construction, constraints

Substitute the fixed spins into the polynomial. For lag k:
  C_k(s) = c_k + sum_{i in U} b_{ki} z_i + sum_{i, i+k in U} z_i z_{i+k},
  c_k = sum_{i, i+k in Phi} s_i s_{i+k},
  b_{ki} = [i-k in Phi] s_{i-k} + [i+k in Phi] s_{i+k}.
H_node(z) = sum_k C_k(z)^2 = E(completion). Collecting monomials (z_i^2 = 1) gives 1-, 2-, 3-,
4-body terms with integer coefficients and a constant (eq. 16 analog). The column elimination D
of eq. 11 becomes monomial substitution; sample energies are the true E. Every bound of B.3 is a
bound on min_z H_node(z) (Theorem 1). Term counts: Section 4.
Not included (and why): penalty terms for the budgets. Any z with some C_k^2 > beta_k has
H_node(z) > T already, so a penalty is redundant; it would also raise coefficients and cost.
Constraints (N2): lag budgets C_k(s)^2 <= beta_k with beta_k = T - LB_{-k}(node). Roles: (a)
conflict values (B.6), (b) propagation and infeasibility (Theorem 3), (c) implicit in H_node.
They are node-specific and tighten with UB, as the paper's b tightens through eq. 10.
Alternatives tried in prototypes and not kept: sector rows (sum of spins) as root equalities,
branching on lag budgets, long-lag equalities, nogoods, probing. Bound-tightness constraints
\|C_k\| = mu_k are the budgets with beta_k = mu_k^2, the tightest special case; the budget form
keeps the slack T - LB, so it subsumes them.

### B.3 Bound (condition (i))

LB(node) = sum_{k < n/2} m_k + [n even] mu_{n/2}^2, with
  m_k = min { a^2 + b^2 : a in R_k, b in R_{n-k}, a + b = n (mod 4) }  (Theorem 2),
  mu_k = min \|R_k\|  (Theorem 1), R_k the exact range of C_k at the node.
Integer arithmetic only; O(n^2) per node. m_k from at most 4 residue classes. LB is monotone along
the tree (Theorem 1), so eq. 19 holds without extra work. LB = floor(n/2) (mod 4) like E (for LB_0
by Theorem 1; for the pair sum because a^2 + b^2 mod 4 depends only on the parities of a, b, which
are those of n-k and k by Lemma 1); nothing in the proof of correctness uses this.
Remark on the paper's bound. QCBB eq. 17 holds in expectation; the certified form is
min H >= W - 2 z_SDP (A.4). Neither is used: an SDP per node costs seconds and the GW gap on the
quartic LABS polynomial is far above E*.

### B.4 Propagation (condition (ii))

For each lag k: beta_k = T - LB_{-k} with LB_{-k} = LB - m_k + mu_{n-k}^2 (middle lag: LB - mu_k^2);
A_k = {v in R_k : v^2 <= beta_k}. A_k empty -> infeasible. A_k = {lo_k} or {hi_k} (single extreme;
exact test lo^2 <= beta and (lo = hi or (lo + g)^2 > beta)) -> every free chain of lag k whose extreme
is attained by a unique assignment is forced (free chains with one fixed end always; with two fixed
ends only if s_first s_last = sign^e; with no fixed end never), Theorem 3; forced spins are fixed in place.
Recompute from the bound until no spin changes. A contradiction (a spin forced to both values)
or LB > T after forcing -> prune. The kernel forces only free spins and recomputes everything
after the first lag that forces, so a contradiction shows up as LB > T, and A_k is never empty
while LB <= T (beta_k >= mu_k^2 then); both tests are kept for completeness in the proof.

### B.5 Symmetry (condition (iii))

Fix s_1 = s_2 = +1 (negation and alternation, Theorem 4(a)): 4x fewer leaves. Classes: all
assignments of the 2 m0 boundary spins s_1..s_m0, s_{n-m0+1}..s_n with s_1 = s_2 = +1
(m0 = min(7, floor(n/2)) by default, 4,096 assignments at m0 = 7). Each class root is a child of
the master with its own LB and enters the queue like every other node (QCBB 3.3). The reversal
lex rule (Theorem 4(b)) depends only on the boundary spins, so `class_roots()` applies it once: 2,112 of
the 4,096 roots are kept at m0 = 7, and no node below a root pays for it. The class-count formula
of Packebusch-Mertens is not used; classes are enumerated explicitly.

### B.6 Conflict values (eqs. 5-9)

Given the node's samples X (s distinct samples, counts omega, q = sum omega; each sample is
completed to a full sequence) and the node's budgets beta_k:
  V_{k l} = 1 if C_k(x^l)^2 > beta_k else 0                       (eq. 5)
  nu_k = (1/q) sum_l V_{k l} omega_l  (nu_k = 1 for all k if there are no samples)   (eq. 6)
  P_{k i} = 1 if spin i in U has a lag-k product with a FIXED partner (i-k in Phi or i+k in Phi) (eq. 7 analog)
  gamma = nu P                                                     (eq. 8)
  k* = argmax_{i in candidates} gamma_i                            (eq. 9)
P is our analog of eq. 7 for a polynomial constraint: QCBB sets P_ji = 1 when variable i
appears in row j at all; C_k depends on every spin with a lag-k partner inside the sequence, and
we count only the partners that are fixed, i.e. the spins in a non-constant linear term of C_k
(the analog of A-hat_ji != 0). The literal choice would make gamma even flatter.
Candidate set (configuration option `candidates`, default "outermost"): "all" (eq. 9 over all
free spins) or "outermost" (eq. 9 over the two outermost free spins). Ties: the free spin nearest to
an end; equal distances: the left one when the number of fixed spins is even, else the right one
(deterministic). Both children (+1 first, then -1) enter the queue; the queue order, not the
samples, decides which is evaluated first.
How gamma behaves on LABS. With a prefix and suffix of length m each, P_{.i} has 2m ones for
every free spin except those near the middle, where the lag ranges [i-m, i-1] and [n-m+1-i, n-i]
overlap and the count drops (n = 30, m = 5: gamma(nu = 1) = 10 for 16 spins, 8, 6, 6, 8 for the
middle four). Proposition 5 (Section 3, PROVEN): on every node whose fixed spins are a prefix and a
suffix (any lengths), eq. 9 with nu = 1 over all free spins picks an outermost free spin, so "outermost" is
eq. 9 itself on such nodes, evaluated on 2 spins instead of F; and when the prefix and suffix have
equal length the two outermost free spins tie for every nu, so the tie rule decides. With sample nu,
"all" may pick an interior spin; long lags have few free products, tight budgets and high nu_k,
and their mass lands on near-end spins. Both modes are complete (Theorem A(v) holds for any free
spin). Alternatives tried in prototypes and not kept: a sample-dependent P (P_{kil} = 1 if
flipping spin i in sample l reduces \|C_k\|), and the visiting order of the two children from
the samples.
Summary: N2 is implemented as eqs. 5-9 with the analog P above, at every sampled node.
Proposition 5 says where the outermost-pair rule is eq. 9 itself, and Proposition 6 says exactly what
the samples can change in a proof: when UB reaches E*, and which outermost spin is fixed at each
branched node. The node counts of the backends in 4.8 differ through these two channels only.

### B.7 Sampler (N1)

Where samples are load-bearing: the incumbent and the branching variable. Every node that
survives steps 1-3 of A.2 is sampled once with `shots` shots (default 1,000); the best sample is
an incumbent candidate, and the samples give nu for the branching variable. For an instance that
is not proven within the budget, UB is the answer and the numerator of the gap. Proposition 6
(Section 3, PROVEN) makes exact what the samples change in a proof. The same structure holds in
the tree-IQO branch and bound of Hadfield, Braydwood and Rieffel (arXiv:2611.00678, MaxCut):
QAOA output ranks branches and supplies candidates, pruning is classical. Read from a sample: the
F bits -> a completion -> exact E (UB); V, nu, gamma.

Sampler: fixed-angle QAOA (Shaydulin et al. 2024). Start in |+>^F, apply p layers of
exp(-i gamma_l H_node / 2) and exp(-i beta_l sum_j X_j), measure in the Z basis. There is no
variational loop: gamma_l = c g_l / n and beta_l = b_l, where (g, b) are the length-free angles
of the master Hamiltonian (QOKit, p = 1..33). Here n is the sequence length, not F; c = 1 for the
master (the published rule), c = 0.875 at p = 1 and 0.625 at p >= 2 for node Hamiltonians, and
c = 0.5 for the skew restriction (EMPIRICAL, Section 4.6). The depth is p = 12 or p = 1 on the
simulator (4.8). One backend per run: every sampler call of a run goes to the same backend. A
Hamiltonian with more qubits than the device takes (`backend.max_size`, 26 for the exact
emulation) raises `SizeError`. Only the master may be that large: its children, the class roots,
do not depend on samples, so the node loop then branches it without samples. Every other node
must fit, which holds for n <= 40 with 7 boundary spins (class roots have n - 14 free spins). For
odd n
the master is also sampled through its skew-symmetric restriction on (n + 1) / 2 qubits (B.12;
on by default with QAOA in the CLI), which fits the emulation up to n = 51 while the master fits
up to n = 26. The sampler contract (bitstrings of the free spins) is enforced: `collect` rejects
samples with a value other than +1 or -1 or with the wrong width. An empty sample set is
harmless: the first incumbent then comes from a later node or a leaf (Theorem A).

### B.8 Pruning stack (all proven, all integer)

1. LB > T (Theorems 1, 2; Lemma 4 lattice). 2. A_k empty (Theorem 3). 3. Propagation
contradiction (Theorem 3). 4. s_1 = s_2 = +1 (Theorem 4(a)) and the reversal lex rule on the
class roots (Theorem 4(b)).

### B.9 Stopping and gap

Proof: the loop runs at T = UB - Delta and ends when the lowest open key exceeds T; then UB is
optimal (Theorem A(c)). Limits: on a node or time limit the node loop reports UB, the certified
LB = min(UB, lowest open key) rounded up to the lattice (A(e) with (iv)), and the gap on energy
(A(f)). The staircase of thresholds (Theorem A(d)) is proven but not run by the node loop.

### B.10 Node loop for LABS

The loop of A.2 with: propagation B.4 and bound B.3 (numba), leaves with E exact in int64, the
node Hamiltonian of B.2 sampled at every node (for odd n the master also through its skew
restriction, B.12), the class roots of B.5 as the children of the master, and eq. 9 over the
outermost free spins with nu from the node's samples (B.6). Only the master may have more qubits
than the device takes; it is then branched into the class roots without samples, and any other
node that does not fit raises `SizeError`. Result: proven optimum when the queue is exhausted at T, else UB, the certified LB
and the gap; the trace and the primal-dual integral of A.2 are part of the result.

### B.11 Component table

| QCBB component | LABS version | What changed, why |
|---|---|---|
| MP: BLP with Ax = b | unconstrained quartic polynomial | linearization needs ~1.5 n^2 binaries, big M ~ n^3/3, GW gap >> E* |
| MH: 2-local Ising, big M | H = sum_k C_k^2, 4-local, n qubits, no M | exact and penalty-free; LABS is the penalty form of C_k = 0 |
| Constraints for V, nu, P | lag budgets C_k^2 <= beta_k, beta_k = T - LB_{-k} | the only constraints implied by "improve UB"; they feed eqs. 5-8, propagation, App. C |
| 3.1 VQA, samples | fixed-angle QAOA (transferred angles, no variational loop) behind the Sampler interface, or SA behind the same interface, one backend per run; every node sampled once; only the master may exceed the device size, and it is then branched without samples; same H_node | N1; optimizing angles per node would cost more than the node; reads bits, exact E, V, nu |
| 3.2.1-3.2.2 conflict values, argmax | eqs. 5-9 with the analog P of eq. 7; candidate set the two outermost free spins (default) or all free spins | the branching order of Mertens and Packebusch-Mertens, from both ends inward (Proposition 5); the samples decide between the two outermost spins |
| 3.2.3 subproblem (10)-(16) | substitution of fixed spins into the polynomial; constant tracked | polynomial instead of linear rows |
| 3.2.4 propagation | budget propagation with exact ranges R_k (Theorem 3) | DPLL-like forcing on C_k^2 <= beta_k |
| 3.3 node selection | lowest key first over all open nodes (priority queue), ties to the deeper then the older node | QCBB 3.3 as stated |
| 3.4 bound (17), (19) | per-lag exact + pair mod-4, integer; monotone | GW is expectation-only and far too loose; (19) automatic |
| 3.5 pruning | LB > UB - Delta | lattice of E (Lemma 4) sharpens LB > UB by one step |
| 3.5.1, App. C infeasibility | A_k empty, propagation contradiction | same role (contradiction in constraints), no M |
| 3.6.1 optimality | lowest open key > T | same |
| 3.6.2 gap | open-node bound min(UB, lowest key) on the lattice | valid anytime LB; gap on energy |
| App. D term counts | Section 4 | 4-local terms ~ F^3/12 |
| App. E convergence | Theorem A(a), (b) | finite tree |
| symmetry (not in paper) | group of order 8, s_1 = s_2 = +1; reversal lex rule on class roots | 4x fewer leaves; the lex rule keeps 2,112 of 4,096 class roots at m0 = 7; Theorem 4 |

### B.12 Skew-symmetric restriction (sampler option, odd n)

For n = 2q - 1, a skew-symmetric sequence satisfies s_{q+l} = (-1)^l s_{q-l} (l = 1..q-1). Such a
sequence has C_k = 0 for every odd k and is fixed by s_1..s_q. Its energy is a quartic Ising
Hamiltonian on q = (n+1)/2 qubits with ~6.5x fewer CNOTs per layer than the master (n = 21: 536 vs
3,890). When the master is sampled, `SkewSampler` samples this Hamiltonian as well, completes
each sample to n spins and merges these samples with the samples of the master itself
(`with_master=True`, default), so the incumbent is never worse than without the restriction.
Every sample is a feasible sequence, so the best is a valid incumbent. The restriction never
enters a bound, a prune or a node Hamiltonian.
EMPIRICAL (exhaustive over the 2^q skew sequences): the skew optimum equals the QOBLIB optimum
for 14 of 24 odd n in 19..65 (21, 27, 29, 39-59) and differs for 19, 23, 25, 31, 33, 35, 37, 61,
63 and 65; this is why the master is sampled as well. MEASURED (exact noiseless emulation, 4.6): at
q = 20-26 (n = 39-51, where the skew optimum is E*), p = 12 QAOA with c = 0.5 reaches E* with
probability 9.2e-4 to 1.0e-2 per shot, i.e. 100-1,100 expected shots, on q <= 26 qubits where the
master does not fit the simulator. So with the QAOA sampler the restriction is on by default in
the CLI for odd n (`--no-skew` turns it off).

---------------------------------------------------------------------------------------------------

## 2. Requirement compliance

N1 (quantum sampler in place, load-bearing, swappable, math written for the quantum sampler).
  Hamiltonian: H_node on F qubits (B.2), the master Hamiltonian at the root. Qubits: n at the
  root, n - 2 m0 at the class roots, fewer below. Read from samples: bitstrings -> exact E -> UB;
  V, nu, gamma. Bound on the same Hamiltonian: LB <= min H_node (Theorem 1). Swappable: one
  `Sampler` protocol (`qcbb.sampler`). Load-bearing: every node is sampled; the sampler is the
  source of the incumbent and of the branching variable; for instances beyond proven reach its
  UB is the reported answer and sets the gap. Verdict: MET, with honest limits. Leaves also lower
  UB, and once UB = E* the samples change the search tree only through the choice between the two
  outermost free spins (Proposition 6, PROVEN); the certificate is classical by construction
  (Theorem A); the sampler is fixed-angle QAOA on the exact emulation (B.7, 4.6, 4.8; only the
  master may exceed 26 qubits, and it is then branched without samples), and hardware runs are demonstrations only.
N2 (branching by conflict values from wisely chosen constraints, eqs. 5-9 spirit).
  Constraints: dynamic lag budgets. V, nu, gamma, argmax as eqs. 5-9 and P as the analog of
  eq. 7 for a polynomial constraint (B.6), over the two outermost free spins (default) or all free
  spins, at every sampled node. Verdict: MET; with the honest fact that on nodes with a fixed prefix
  and suffix eq. 9 with nu = 1 picks an outermost spin (Proposition 5, PROVEN), so over the two
  outermost spins the samples decide
  between two spins.
N3 (bound may differ from GW, must be strictly guaranteed and really fast).
  Integer per-lag exact ranges + pair congruence mod 4, O(n^2) integer operations per node, in
  numba. Proofs: Theorems 1, 2. Verdict: MET.
Pruning essential, wise constraints.
  Stack of B.8, all proven. Verdict: MET.
Optimality guarantee independent of sampler quality. Theorem A (last sentence); the sampler
  contract (+-1 bitstrings) is enforced by `collect`. Verdict: MET.
Every change described (what, why, how). B.11 and the text. Verdict: MET.
Python kept, numba for the hot parts (bound, propagation, annealing, emulation); modular,
  docstrings. Layout: README.
Resource needs, proofs, reachable sizes. Sections 4, 3, 4.8.
Not every instance must be solved. MEASURED (4.8): with 1,000 shots per node, QAOA on the exact
  emulation proved labs002-labs030 at p = 12 and at p = 1, and SA proved labs002-labs026; above
  that, the incumbent with a certified gap.

---------------------------------------------------------------------------------------------------

## 3. Proofs (correctness)

Status key: PROVEN (proof here or exact reference), EMPIRICAL (heuristic side, no proof needed),
CONJECTURE (Section 6 only). Theorem A (Part A) is the framework theorem. The LABS results prove
its conditions: Lemma 4 -> (iv); Theorems 1, 2 -> (i); Theorem 3 -> (ii); Theorem 4 -> (iii);
binary branching on any free spin -> (v). Corollary: the LABS algorithm returns a proven optimum
or a valid gap for every sampler. Dependency: Thm 3 <- Thm 1, Thm 2; Thm 2 <- Thm 1, Lemma 2;
Lemma 4 <- Lemmas 1-3.

### Lemma 1 (parity). C_k = n - k (mod 2). PROVEN.
C_k is a sum of n - k terms, each +1 or -1. A sum of m terms in {+1, -1} with t terms equal to -1
is m - 2t = m (mod 2). ∎

### Lemma 2 (periodic congruence). C_k + C_{n-k} = n (mod 4) for 1 <= k <= n - 1. PROVEN.
Index positions mod n. The periodic autocorrelation P_k = sum_{i=1}^{n} s_i s_{i+k} (indices mod n)
splits into i <= n - k, giving C_k, and i > n - k, giving sum_{j=1}^{k} s_{j+n-k} s_j = C_{n-k}.
So P_k = C_k + C_{n-k}. The map i -> i + k (mod n) is a permutation of Z_n whose cycles have
length n / gcd(n, k). Along one cycle (i_1, ..., i_L), the product of the L terms
s_{i_t} s_{i_{t+1}} equals prod_t s_{i_t}^2 = 1, so the number of terms equal to -1 in that cycle
is even. Summing over cycles, the number N of -1 terms in P_k is even. P_k = n - 2N = n (mod 4). ∎
(For k = n/2, n even, this reads 2 C_{n/2} = n (mod 4), i.e. Lemma 1 for that lag.)
Reference: Packebusch and Mertens 2016, eq. (15). The mod-8 statement of Lemma 4(b) is NOT in
Packebusch-Mertens 2016 or in Mertens 1996 (J. Phys. A 29 L473); it is proven here.

### Lemma 3 (energy residue). E = floor(n/2) (mod 4). PROVEN.
An odd square is 1 (mod 8), an even square is 0 (mod 4). By Lemma 1, C_k is odd iff n - k is odd.
The number of k in {1, ..., n-1} with n - k odd is the number of odd integers in {1, ..., n-1},
which is floor(n/2). Hence E = sum C_k^2 = floor(n/2) (mod 4). ∎

### Lemma 4 (lattice step Delta; condition (iv)). PROVEN.
(a) For every n, all attainable E lie in one residue class mod 4 (Lemma 3).
(b) For n = 4m + 2, all attainable E lie in one residue class mod 8. Proof: pair the lags
(k, n-k) for k = 1..2m and keep the middle lag n/2 = 2m + 1. Odd k (m pairs): n - k is odd, both
C_k and C_{n-k} are odd (Lemma 1), their squares are 1 (mod 8), the pair contributes 2 (mod 8). Even k
(m pairs): both are even; write a = 2a', b = 2b'. By Lemma 2, a + b = 2 (mod 4), so a' + b' is odd, so
exactly one of a', b' is odd and a'^2 + b'^2 is odd; a^2 + b^2 = 4(a'^2 + b'^2) = 4 (mod 8).
Middle lag: n - (2m+1) = 2m + 1 is odd, so C_{n/2} is odd and its square is 1 (mod 8).
E = 2m + 4m + 1 = 6m + 1 (mod 8). ∎
(c) Hence E(X) is contained in f_0 + Delta Z with Delta = 8 for n = 2 (mod 4) and 4 otherwise:
condition (iv). In particular, if UB = E(s_UB) is attained and E(s) < UB then E(s) <= UB - Delta. ∎

### Theorem 1 (per-lag ranges and the bound LB_0). PROVEN.
Fix a node (Phi, U) and a lag k. Write the n positions as k chains r, r + k, r + 2k, ...
(r = 0..k-1 in 0-based indexing); every lag-k product s_a s_{a+k} is a product of two consecutive
positions of one chain. Call a product free if at least one of its two positions is in U, fixed
otherwise, and let c_k be the sum of the fixed products. Cut every chain at each of its fixed
positions. A free chain is a piece between two consecutive fixed positions of a chain, or between a
chain end and the nearest fixed position, or a whole chain without fixed positions, that contains
at least one free position, together with the products along it; equivalently, a maximal run of
consecutive free products with no fixed position in its interior. Products between two adjacent
fixed positions are fixed. Every interior position of a free chain is free, and only its end
positions can be fixed. Two different free chains share no free position: a free position a lies
in the products (a-k, a) and (a, a+k), which are consecutive with no fixed position between
them, hence in the same free chain. Therefore, as the free spins range over {-1,1}^U, the free chain sums vary over
independent sets, and the set of attainable C_k is
  R_k = c_k + (S_1 + S_2 + ... + S_q)   (Minkowski sum over free chains),
where S_j is the set of sums attainable by free chain j alone. For one free chain with e products:
 (i) both end positions fixed (at A and B), e - 1 free spins between them. The map from the
     free spins to the product vector (p_1, ..., p_e) in {-1,1}^e is injective, and its image is
     exactly {p : prod_i p_i = s_A s_B} (given p, the spins are s_{A + ik} = s_A p_1 ... p_i, and
     the last product then equals s_A s_B iff prod p = s_A s_B). The sum is e - 2t with t the number
     of -1 entries, t = [s_A s_B = -1] (mod 2), 0 <= t <= e. So S_j = {e - 2t : t of that parity}:
     an arithmetic progression of step 4, from lo_j = -e or -(e-2) to hi_j = e or e-2, as the
     parity allows.
 (ii) exactly one end fixed (the run reaches the chain end): e free spins, the map spins ->
     product vector is a bijection onto {-1,1}^e; S_j = {-e, -e+2, ..., e}, step 2.
 (iii) no end fixed (the whole chain is free): e products, e + 1 free spins; each product vector
     is attained (twice); S_j = {-e, ..., e}, step 2.
Minkowski sums: a sum of arithmetic progressions of step 4 is an arithmetic progression of step 4
(from the sum of the lo's to the sum of the hi's). If at least one summand is a step-2 progression
with at least two elements {a, a+2, ...} then the sum is the full step-2 progression from the sum
of the lo's to the sum of the hi's: given a target x of the right parity, pick z in the remaining
sum (a progression of step 2 or 4) as the largest element <= x - a; then x - z lies in [a, a + 4)
with the right parity, i.e. x - z in {a, a+2}, both in the step-2 summand (if x - a exceeds the
top of the remaining sum, take its top; then x - top is <= the summand's top and >= a).
These cases are exhaustive because every interior position of a free chain is free.
Hence R_k = c_k + {umin_k, umin_k + g_k, ..., umax_k} with g_k = 4 if every free chain is of type (i),
g_k = 2 otherwise (g_k irrelevant if there is no free chain). Every element of R_k is n - k (mod 2)
(Lemma 1 applied to any completion).
Bound. mu_k = min{\|v\| : v in R_k} is computed in O(1) from (c_k, umin_k, umax_k, g_k): if
c_k + umin_k >= 0 it is c_k + umin_k; if c_k + umax_k <= 0 it is -(c_k + umax_k); otherwise it is
min(x, g_k - x) where x is the smallest nonnegative element of the progression. For every
completion s, C_k(s) in R_k, so C_k(s)^2 >= mu_k^2, and E(s) >= sum_k mu_k^2 =: LB_0(node). ∎
Monotonicity (eq. 19). A child's completions are a subset of the parent's, so R_k(child) is a
subset of R_k(parent), mu_k does not decrease, LB does not decrease. ∎
Lattice of LB. mu_k = n - k (mod 2), so sum mu_k^2 = floor(n/2) (mod 4) by the argument of Lemma 3. ∎
Plain-range relaxation. R_k is contained in {c_k - f_k, ..., c_k + f_k} (step 2), f_k = number of
free products, so mu_k >= max(\|c_k\| - f_k, (n-k) mod 2), a weaker valid bound (Packebusch-Mertens
eq. (12) state it as max(b_k, \|c_k\| - f_k) with their b_k; Prestwich 2013). ∎
Reference. Packebusch and Mertens, J. Phys. A 49 (2016) 165001 (arXiv:1512.02475), Sec. 3.1.5,
eqs. (18)-(25): ranges with granularity 2 or 4 for the contiguous free block (their
induction covers odd run lengths; the even case is omitted there). The proof above is
self-contained, covers both parities and arbitrary fixed sets (needed because propagation fixes
interior spins). Cost: the chain scan is O(n - k) per lag, O(n^2 / 2) per node. Implementation:
`bound.lag_range` cuts every chain at every fixed position, as above; `tests/test_labs_bound.py`
checks R_k against the enumerated set of C_k values at random nodes with arbitrary fixed sets.

### Theorem 2 (pair mod-4 coupling, bound LB; condition (i)). PROVEN.
For k < n/2 and any completion s: C_k(s) in R_k, C_{n-k}(s) in R_{n-k} (Theorem 1), and
C_k(s) + C_{n-k}(s) = n (mod 4) (Lemma 2). Hence
  C_k(s)^2 + C_{n-k}(s)^2 >= m_k := min{ a^2 + b^2 : a in R_k, b in R_{n-k}, a + b = n (mod 4) },
the set being nonempty because the true pair is in it. Summing pairs and adding mu_{n/2}^2 for
even n (Theorem 1) gives LB >= LB_0, and LB satisfies (i). Computation: the objective is
separable, so for each residue rho (mod 4) the best a is the element of R_k of residue rho with
the smallest \|a\| (if any), and the best b is the element of R_{n-k} of residue n - rho with the
smallest \|b\| (if any); m_k is the minimum over the at most 4 residues. For an arithmetic
progression {lo, lo + g, ..., hi} the smallest \|a\| in a residue class is found in O(1) (g \| 4, so
each residue class is a progression of step 4 or the whole set). ∎
Root value for n = 2 (mod 4). R_k at the root is the full parity class {-(n-k), ..., n-k}.
Odd k: a, b odd, min a^2 + b^2 = 2 (a = b = 1, sum 2 = n (mod 4) OK). Even k: a, b even,
a + b = 2 (mod 4) forbids (0, 0); min is 4. Middle lag: odd, min 1. With n = 4m + 2:
LB_root = 2m + 4m + 1 = 3n/2 - 2. ∎ (Equals the optimum for n = 6, 10, 14, 18.)
Note: the pair bound with the chain-aware sets (step 4 classes) is stronger than with plain ranges
(EMPIRICAL, our prototype: 1.06-1.44x fewer nodes at n = 16-21).

### Theorem 3 (budget propagation and infeasibility; condition (ii)). PROVEN.
Let the node be N, threshold T, and let LB_{-k} be any valid lower bound on
sum_{j != k} C_j(s)^2 over completions s of N. Concretely, with the pair decomposition of Theorem 2:
for k paired with k' = n - k, LB_{-k} = LB - m_k + mu_{k'}^2 (drop the pair, re-add the exact
per-lag bound of the partner lag, valid because C_{k'}^2 >= mu_{k'}^2); for the middle lag
LB_{-k} = LB - mu_k^2. Set beta_k = T - LB_{-k} and A_k = {v in R_k : v^2 <= beta_k}.
Claim 1. Every completion with E(s) <= T has C_k(s) in A_k. Proof: C_k(s)^2 = E(s) - sum_{j != k} C_j^2
<= T - LB_{-k} = beta_k, and C_k(s) in R_k by Theorem 1. ∎
Claim 2 (infeasibility, App. C analog). If A_k is empty, N has no completion with E <= T. ∎
Claim 3 (forcing). Suppose A_k = {lo_k} where lo_k = c_k + umin_k is the smallest element of R_k.
Then every completion with E <= T has C_k(s) = lo_k, i.e. sum of free chain sums = umin_k = sum of
free chain minima, so every free chain attains its own minimum (each is >= its minimum). For a free chain
of type (ii) the minimum -e is attained only when all e products are -1, which fixes its free
spins uniquely from the fixed end: s_{A + ik} = (-1)^i s_A (or from the right fixed end). For a
type-(i) free chain whose minimum is -e (parity allows), all products are -1 and the free spins are
fixed uniquely as well. For a type-(i) free chain with minimum -(e - 2), or a type-(iii) free chain,
the minimizing assignment is not unique and nothing is forced (type (iii) forces only relations
between spins; not used). The symmetric statement holds for A_k = {hi_k} with all products +1.
Hence all forced values are necessary for E <= T, and fixing them removes no completion with
E <= T. ∎
Exact test for "A_k = {lo_k}": lo_k^2 <= beta_k and (lo_k = hi_k or (lo_k + g_k)^2 > beta_k).
Proof: if lo_k + g_k <= 0 then (lo_k + g_k)^2 < lo_k^2 <= beta_k and A_k has two elements, so the
test correctly fails; if lo_k + g_k > 0, every element above lo_k + g_k has a larger square than
(lo_k + g_k)^2, so A_k = {lo_k} iff (lo_k + g_k)^2 > beta_k. Symmetric for hi_k. (A rule of the form "(\|c_k\| - f_k + 2)^2 > beta_k
and c_k != 0" without the guard \|c_k\| >= f_k is unsound; the test above is exact.)
Claim 4 (repetition and contradiction). After forcing, the sets R_j shrink or stay (Theorem 1), so
recomputing LB, beta, A is valid; repeat until no spin changes. At most \|U\| rounds. If a forcing
assigns a spin already fixed to the other value, two necessary conditions for E <= T contradict,
so N has no completion with E <= T: prune. If LB rises above T: prune (Lemma 4(c)). ∎
Together, Claims 1-4 give condition (ii). The kernel restarts the pass after the first
lag that forces anything, so every forcing uses sets computed from the current assignment.

### Theorem 4 (symmetry reduction; condition (iii)). PROVEN.
Maps: N (negation) s_i -> -s_i; A (alternation) s_i -> (-1)^{i-1} s_i; R (reversal) s_i -> s_{n+1-i}.
Invariance: N leaves every product unchanged. A multiplies each lag-k product by (-1)^k, so
C_k -> (-1)^k C_k and E is unchanged. R: C_k(Rs) = sum_i s_{n+1-i} s_{n+1-i-k} = C_k(s) after
reindexing. All three are involutions; N commutes with A and R; RA = AR when n is odd and
RA = N AR when n is even. The group G they generate has order 8 (elements N^a A^b R^c), and E is
constant on orbits. Reference: Packebusch and Mertens 2016, Appendix A, eqs. (28)-(34): same
group, lex-min representative; their class count, eq. (26), is in their Sec. 3.2. Outside-in 4-way branching with tie
pruning: Mertens 1996, Algorithm 1.
(a) K = {s : s_1 = s_2 = +1}. For any s, exactly one element of the coset {N^a A^b s} lies in K:
a = [s_1 = -1] makes s_1 = +1 (N flips s_1, A does not), then b = [s_2 = -1 after the first step]
makes s_2 = +1 (A flips s_2 and keeps s_1). Call it kappa(s). The orbit of s meets K in exactly
{kappa(s), kappa(Rs)}. So K meets every orbit and restricting the search to K satisfies (iii). ∎
(b) Reversal lex rule. Fix a total order on {+1, -1} and let
K' = {s in K : s <=_lex kappa(Rs)}. K' meets every orbit: <N, A> is normal in G (RN = NR, and
RA = AR or N AR), so R maps the coset {N^a A^b s} onto {N^a A^b Rs}; hence for c = kappa(s),
kappa(Rc) = kappa(Rs), and with c' = kappa(Rs), kappa(Rc') = kappa(s). The smaller of the two
in the lex order therefore lies in K'. Let the node have s_1 = s_2 = +1, the prefix p = (s_1..s_m) and the suffix
q = (s_{n-m+1}..s_n) fixed, m >= 2. For any completion s, kappa(Rs)_i = eps_i s_{n+1-i} with
eps_i = s_n (-1)^{(i-1) b}, b = [s_n s_{n-1} = -1]: the sign pattern depends only on s_n, s_{n-1},
so the first m entries p' of kappa(Rs) depend only on q and are known at the node. Rule: discard
the node if p' <_lex p (strictly, decided within the first m entries). A completion c in K' would
need c <=_lex kappa(Rc), whose first m entries are p and p'; p' <_lex p contradicts this. So no
completion lies in K' and the rule satisfies (iii) with K' in place of K. ∎
The rule uses only positions 1..m and n-m+1..n, is monotone in m (once strictly decided it stays),
and is unaffected by interior spins fixed by propagation. Implementation: the order puts +1 before
-1; the rule is applied to the class roots (m = m0), where both blocks are fixed, so
`LabsProblem.class_roots()` drops the failing roots and no node below a root needs the test. (The class-count formula c(m) = 2^(2m-3) + 2^(m-2+(n mod 2)) of
Packebusch-Mertens is not used anywhere.)

### Corollary (LABS satisfies Theorem A). PROVEN.
With LB of Theorem 2 (i), the propagation of Theorem 3 (ii), the restriction s_1 = s_2 = +1 of
Theorem 4(a) (iii), the lattice of Lemma 4 (iv) and binary branching on any free spin (v), the
LABS algorithm of Part B returns a proven optimum or a valid gap for every sampler, with the
staircase and open-node bounds of Theorem A(d)-(f). The reversal rule of Theorem 4(b) may replace
K by K'. ∎

The two results below are not needed for optimality. They state exactly what the conflict
values and the samples can change.

### Proposition 5 (eq. 9 on nodes with a fixed prefix and suffix). PROVEN.
Let a node have the fixed set {0..a-1} and {b+1..n-1} (0-based), free block U = {a..b},
r = n-1-b, a + r >= 1, F = b-a+1. Let the tie key of a free spin i be (d(i), side) with
d(i) = min(i, n-1-i), as in `LabsProblem.tie_key`.
(a) With nu = 1, eq. 9 over all free spins picks a or b.
(b) If a = r, then gamma_a = gamma_b for every nu >= 0.
Proof. (a) For i in U, lag k gives i a fixed partner on the left iff i-k in [0, a), i.e.
k in K_L(i) = {i-a+1..i} (a values), and on the right iff i+k in (b, n-1], i.e.
k in K_R(i) = {b+1-i..n-1-i} (r values); all these k lie in 1..n-1. With nu = 1,
gamma_i = |K_L(i) u K_R(i)| = a + r - O(i), O(i) = max(0, min(i, n-1-i) - max(i-a+1, b+1-i) + 1).
Take an interior spin a < i < b (so F >= 3). Case i <= n-1-i. Then a < i <= (n-1)/2, so
d(a) = a < i = d(i): a has the strictly smaller tie key. O(a) = max(0, a - F + 1), since
max(1, F) = F and min(a, n-1-a) = a. For i, min(i, n-1-i) = i >= a+1 and both i-a+1 and b+1-i are
<= F-1, so O(i) >= max(0, a - F + 3) >= O(a), i.e. gamma_i <= gamma_a. Hence i is not chosen: if
gamma_i is below the maximum it is not a candidate, and if it is the maximum, a is a candidate too
and wins the tie. Case i > n-1-i: the reflection j -> n-1-j maps the node to one with prefix r and
suffix a, maps b to the left end and keeps d, so the first case applies with b in place of a.
(b) If a = r, the reflection j -> n-1-j maps the fixed set onto itself and a onto b, and
j - k is fixed iff (n-1-j) + k is fixed, so P_{k a} = P_{k b} for every k and
gamma_a = (nu P)_a = (nu P)_b. ∎
`tests/test_labs_bound.py` checks (a) for every prefix and suffix length at n <= 40 and (b).

### Proposition 6 (what the samples change in a proof). PROVEN.
Assume UB = E* is known when a node N is taken from the queue, so T = E* - Delta from then on.
(a) T does not change any more: a leaf or sample lowers UB only if its value is < E*, which is
impossible.
(b) Whether N is pruned depends only on N and T (steps 1-2 of A.2), not on any sample.
(c) If N is not pruned, its samples choose the branching variable among the outermost free spins
(eq. 9 with nu from the samples). Hence the samples change the search tree in two ways only: they
decide when UB reaches E* (every node evaluated earlier is judged at a larger T), and at every
branched node they choose which outermost spin is fixed next. If a node's fixed spins form a
prefix and a suffix of equal length with no fixed interior spin, Proposition 5(b) makes the two
outermost free spins tie for every nu, so the samples do not change that choice. ∎

---------------------------------------------------------------------------------------------------

## 4. Resource needs

### 4.1 Qubits
Every node that is not pruned and is not a leaf is sampled: the master on n qubits
(labs002-labs100: 2-100; skew option B.12, odd n: q = (n+1)/2), the class roots on
F = n - 2 m0 qubits (default m0 = 7: n - 14), and every node below a root on fewer. Only the master
may exceed the device size; it is then branched without samples (B.7). MEASURED (4.8, exact emulation, max_size = 26): the
largest sampled F of a run is n for n <= 26, (n + 1) / 2 for odd n = 27, 29 (skew restriction)
and n - 14 for even n = 28, 30, where the master returns no samples.

### 4.2 Term counts (term columns: computed exactly by collecting monomials, n = 50, random fixed context; CX and depth columns: MODEL, 6/4/2 CX per ZZZZ/ZZZ/ZZ, no routing, idealized parallelism)
| F | 1-body | 2-body | 3-body | 4-body | CX per QAOA layer | ~2q-depth all-to-all (F/2 gates in parallel) |
|---|---|---|---|---|---|---|
| 8 | 8 | 28 | 41 | 22 | 352 | 88 |
| 12 | 12 | 66 | 146 | 95 | 1,286 | 214 |
| 16 | 16 | 120 | 385 | 252 | 3,292 | 412 |
| 20 | 20 | 190 | 749 | 525 | 6,526 | 653 |
| 24 | 24 | 266 | 1,270 | 946 | 11,288 | 941 |
| 30 | <=30 | 318-386 | 2,241-2,579 | 1,925 | 21,000-23,000 | 1,400-1,500 |
| 40 | 40 | 540-690 | 3,183-5,417 | 4,750 | 42,000-52,000 | 2,100-2,600 |
| 66 (root) | 0 | 1,056 | 0 | 22,352 | 136,224 | 4,128 |
Scaling: 4-body = number of quadruples a < b < c < d with b - a = d - c ~ F^3/12 (the count n^3/6
often quoted is the number of (k, i, j) triples; two triples collapse into one monomial with
coefficient 4); 3-body ~ F^3/12 to F^3/6 depending on the fixed context (zero at the root);
2-body ~ F^2/4 .. F^2/2; 1-body F (zero at the root). CX per layer ~ F^3/2. Plus one Rz per term.

### 4.3 Depth, connectivity, hardware class
All-to-all (trapped ions, neutral atoms): depth per layer ~ F^2 two-qubit layers (table).
Heavy-hex (IBM): every ZZZZ on four arbitrary qubits needs routing; with F^3/12 such terms the
SWAP overhead is a further factor 3-10 (ESTIMATED), i.e. 10^4-10^5 two-qubit gates per layer at
F = 20-30. With two-qubit error ~10^-3 the usable budget is ~10^3 two-qubit gates: today a
QAOA layer on H_node is realistic only at F <= 12 (ESTIMATED), F <= 16 on all-to-all machines.
Root and class-root Hamiltonians of the instances that matter (n >= 30) do not fit this regime.
The two-qubit budget allows p = 1 at F <= 12-14 (nodes), n <= 18-20 (master) and q <= 14-16 (skew),
by gate count only. On trapped ions, the output of p = 1 QAOA is at the uniform-random level for
N >= 8 (QOBLIB LABS submission). A hardware run is therefore an interface demonstration. The
QAOA sampler runs on the exact noiseless emulation (4.6, 4.8). Classical simulation: our CPU
emulation reaches F ~ 28 (32 * 2^F bytes); QOKit reaches n ~ 40 on a GPU.

### 4.4 Sampler calls and device time per instance
One sampler call per sampled node. MEASURED (4.8, 1,000 shots per node, seed 1, p = 12): sampler
calls per proof n = 20: 130; 22: 730; 24: 2,598; 26: 3,392; 28: 76,237; 29: 256,140;
30: 110,868. Device time per shot, overhead included, is 0.03-1 s (QOBLIB LABS submissions, AQT
and IBM), so 1,000 shots take 0.5-15 min per call (ESTIMATE). At these rates the n = 20 proof
(130 calls) would take 1-33 h of device time and the n = 28 proof (76,237 calls) 26 days to
2.2 years (ESTIMATE). On hardware every node below the master must fit the device's `max_size`;
a larger one raises `SizeError` (B.7). Batching: `qaoa/device.py` packs the
programs of one sampler call on disjoint qubits up to a share of the device's qubits (4.7); the
node loop passes one Hamiltonian per call, so packing does not reduce the number of jobs of a
run.

### 4.5 Classical SA cost
numba HUBO SA: a single-spin flip changes C_k by -2 s_j (s_{j-k} + s_{j+k}) for every k, so
Delta E costs O(n) per flip and a sweep O(nF): MEASURED 1.8 ms per 2,000 sweeps at n = 30 (Apple
M4 Pro, one core, numba), i.e. ~0.9 us per sweep; ESTIMATE 2-4 us per sweep at n = 50-60.
MEASURED (`bnb_sa.log`, 4.8; the shots of a call run in parallel threads): a call of 1,000 shots
x 1,000 sweeps costs 14-26 ms averaged over the calls of a proof at n = 19-26, against 1-3 ms
for the QAOA emulation at p = 12 and 0.7-1.5 ms at p = 1 (4.8). Memetic tabu (Gallardo, Cotta,
Fernandez 2009; Boskovic, Brglez, Brest 2017 report a time-to-solution fit ~1.34^n) is the
stronger classical stand-in; it would plug in through the same `Sampler` interface (not
implemented).

### 4.6 QAOA sampler on the simulator (MEASURED, noiseless, Apple M-series 12 cores, numba)
Convention check (test): the ground-state probability of the master Hamiltonian equals QOKit's
stored value to all printed digits (n = 12, p = 12: 0.1735587; n = 16, p = 4: 0.0205713).
State preparation, p = 12, node Hamiltonians: F = 16: 0.03 s; 20: 0.09 s; 22: 0.27 s; 24: 1.1 s;
26: 4.7 s. The diagonal is the Hadamard transform of the coefficients (0.2 s at F = 26).

| n (master) | p_opt, p = 12 | expected shots 1/p_opt | SA 1,000 sweeps: p_opt per shot | SA ms per shot |
|---|---|---|---|---|
| 14 | 0.205 | 5 | 1.00 | 0.058 |
| 18 | 0.039 | 26 | 0.91 | 0.073 |
| 20 | 0.0068 | 147 | 0.25 | 0.092 |
| 24 | 0.0032 | 314 | 0.10 | 0.123 |

Values stored by QOKit for larger n: p_opt = 7.4e-4 (n = 28), 1.0e-5 (n = 40). At n <= 24, QAOA on
a CPU simulator is ~10^3 times slower than SA to the same incumbent.
Angle scale c (gamma_l = c g_l / n), noiseless, grid c = 0.25..1.5 (`qaoa_scale.py`,
`qaoa_optroots.py`, `qaoa_skew.py`). What matters for the incumbent is the class root whose ground
state is the global optimum ("optimum root"), so c is chosen there. Optimum roots, 22 Hamiltonians,
F = 8-24, m0 = 6 and 8, geometric mean of p_opt(c) / p_opt(best c): p = 12: c = 0.5: 0.66,
0.5625: 0.77, 0.625: 0.64, 0.75: 0.24, 1.0: 0.06 (best c 0.5 at F <= 14, 0.5625-0.6875 at
F = 16-24); p = 4: 0.5625 and 0.625: 0.91, 0.75: 0.75; p = 1: 0.875: 0.99, 1.0: 0.98, 0.75: 0.92.
F = 26 (n = 44, m0 = 9) and F = 28 (n = 46, m0 = 9), optimum roots, p = 12: best c = 0.625 with
p_opt = 3.9e-3 and 2.0e-3 (c = 0.75: 7.1e-4 and 4.8e-4; c = 1: 2.4e-5 and 3.1e-5; uniform 1.5e-8
and 7.5e-9); p = 1: best c = 1.0 and 0.875 (p_opt 5.9e-6, 4.7e-6). Random class roots (24
Hamiltonians, F = 12-24): c = 0.75 is best at p = 12 (geometric mean 0.96; 0.625: 0.72) and
c = 0.875-1.0 at p = 1. Skew restriction (q = 11-26): c = 0.5 is best at p = 12 for every
q >= 20 (c = 0.75 is 5-26x worse) and at p = 1 (geometric mean 0.96). Rule used
(`angles.node_scale`, `SKEW_SCALE`): nodes c = 0.875 at p = 1 and 0.625 at p >= 2, skew c = 0.5,
master c = 1. EMPIRICAL constants (C8); the proof does not depend on them. State preparation at
F = 28, p = 12: 21 s and about 8 GB (statevector 4 GB, diagonal 2 GB).
Noise (`qaoa_noise.py`, qiskit-aer statevector trajectories, 2,000 per point, depolarizing error
eps on every CX, no single-qubit or readout error, all-to-all, the package circuit transpiled to
CX + single-qubit gates, p = 1, c = 1 for the master and 0.75 for the skew restriction). Ground-state
probability, noiseless / eps = 2e-3 / uniform: master n = 10 (340 CX): 0.108 / 0.086 / 0.039;
n = 12 (630 CX): 0.023 / 0.013 / 0.0039; n = 14 (1,050 CX): 0.026 / 0.012 / 0.0044; skew n = 21
(q = 11, 536 CX): 0.0099 / 0.0060 / 0.0020; n = 25 (q = 13, 960 CX): 0.0024 / 0.0010 / 0.00049.
So p = 1 is 2.8-7.1x above uniform without noise and 2.1-3.5x above uniform at eps = 2e-3
(2.5-4.7x at 1e-3, 1.2-2.1x at 5e-3). In this idealized model the output is not at the uniform
level, but p = 1 itself gains only a small factor over uniform sampling at these sizes; a device
run remains an interface demonstration.
End to end: Section 4.8.

### 4.7 Devices (one choice per problem; local emulation MEASURED)
Interface: every backend serves the same `Sampler`; IBM and Quantinuum are devices behind
`QaoaSampler` (`qcbb.qaoa`). A real device runs only with `submit=True`, and the resource estimate (and on Nexus
the HQC cost, capped by `max_cost`) is logged before every submission. Credentials stay in each
SDK's saved login. Choices, from the literature and the providers' documentation:

| problem | IBM (Heron, heavy-hex) | Quantinuum (H-series, all-to-all) |
|---|---|---|
| two-qubit gate count | fractional gates (native RZZ, RX) | native ZZPhase |
| connectivity, qubit quality | preset pass manager level 3 (error-aware layout and routing) | none needed; device pytket pass |
| idle and crosstalk errors | dynamical decoupling XY4 | - |
| readout | measurement twirling (TREX); no M3 (samples, not expectation values) | - |
| error detection | Iceberg optional, off by default | Iceberg, on by default |
| throughput | spatial packing (Liaqat et al. 2026) | spatial packing |

Not used: gate twirling (incompatible with fractional gates), ZNE/PEC/PEA (they estimate
expectation values, not samples). Full error correction of QAOA's non-Clifford rotations is not
available on these devices. Helios (Guppy/HUGR programs) is not supported yet.
Iceberg (`iceberg.py`): logical gates are even-weight Pauli rotations, so the LABS master's phase
layer is unchanged; the syndrome is read once at the end (S_X by an ancilla, S_Z by the parity of
the final measurement). PROVEN by the tests in `tests/test_qaoa.py` (exact emulation): without
noise every shot is accepted and the logical distribution equals the unencoded one, and a single X or
Z error between logical gates is always detected. Not fault tolerant (errors inside CX ladders or on
the ancilla can pass).
MEASURED (local emulation, F = 10 class root of n = 24 whose ground state is the optimum, p = 1,
40,000 shots, seed 3; `paper/figures/data/scripts/hw_study.py`, log `hw_study.log`; uniform
ground-state probability 0.0010; IBM: fake_marrakesh calibrated noise, no fractional gates in
the fake backend; Quantinuum: depolarizing model 1e-3 / 3e-5 / 1e-3, rough H2-class rates).
p(ground state) is the fraction among the accepted shots with its Wilson 95 % interval; the
counts are ground states / accepted shots:

| device | accepted | ground states / accepted | p(ground state) [Wilson 95 %] | mean E | two-qubit gates | est. fidelity |
|---|---|---|---|---|---|---|
| exact emulation (noiseless) | 1.00 | 609 / 40,000 | 0.0152 [0.0141, 0.0165] | 207.2 | - | 1 |
| IBM | 1.00 | 120 / 40,000 | 0.0030 [0.0025, 0.0036] | 240.5 | 917 | 0.07 |
| IBM + Iceberg | 0.25 | 10 / 9,992 | 0.0010 [0.0005, 0.0018] | 254.0 | 1,702 | 0.004 |
| Quantinuum | 1.00 | 487 / 40,000 | 0.0122 [0.0112, 0.0133] | 213.5 | 279 | 0.74 |
| Quantinuum + Iceberg | 0.71 | 399 / 28,567 | 0.0140 [0.0127, 0.0154] | 211.2 | 450 | 0.62 |

Reading: on heavy-hex, Iceberg doubles the two-qubit gates and brings the output to the uniform
level (0.0010 [0.0005, 0.0018] against uniform 0.0010), so it stays off on IBM; without the
code the IBM model keeps 3x uniform. On all-to-all, the circuit keeps 0.74 estimated fidelity
without the code, and Iceberg raises p(ground state) among the accepted shots from 0.0122
[0.0112, 0.0133] to 0.0140 [0.0127, 0.0154] (the intervals touch: a small gain from
post-selection at 71 % acceptance, not a loss). The depolarizing model cannot show what the code
detects on a device (leakage, memory and correlated errors), so the Quantinuum default follows
the device results of He et al. and Jin et al.; the decision is to be checked on Quantinuum's
own emulator (H2-1E). None of this affects the proof (Proposition 6).

### 4.8 Backend comparison and coverage: QAOA (exact emulation) versus SA, one backend per run (MEASURED)
Setting (`paper/figures/data/scripts/bnb_study.py`, run by `run_bnb.sh`; Apple M4 Pro, one
process; package defaults: m0 = 7, reversal rule, candidates "outermost", propagation; 1,000 shots
per node; seed 1; at most one hour per run). QAOA: fixed angles at p = 12 and at p = 1 with the
angle scale of 4.6, on the exact noiseless emulation with max_size = 26 qubits; the
master above 26 qubits is branched without samples (B.7), so for even n >= 28 the master gets no
samples and the first incumbent comes from the first class root, and for odd n the master is
sampled through its skew restriction (B.12). SA: 1,000 sweeps per shot, every Hamiltonian sampled.
Both backends ran the same seed; the trace, the sampler calls, the time split and the PDI are
those of A.2; the emulation counts its circuits and shots. Logs: `bnb_qaoa_p12.log`,
`bnb_qaoa_p1.log`, `bnb_sa.log`; one record per run with the full trace and call log:
`bnb_traces.jsonl` (`tables.py bnb_traces.jsonl` prints the coverage tables; `make_figures.py
primal_dual` draws the bounds against time and the PDI against n).

Coverage (every run PROVEN with E = E*): labs002-labs030 at p = 12 and at p = 1 (n = 2..30),
labs002-labs026 with SA (n = 2..26). For n <= 15 every backend needs one sampler call and 1 node
(6 at n = 9, 3 at n = 15), at most 0.07 s. Rows for n >= 16 (seed 1; sampled = calls that
returned samples; the difference is the master above 26 qubits):

| n | E* | backend | nodes | sampler calls | sampled | sampler (s) | total (s) | PDI (s) |
|---|---|---|---|---|---|---|---|---|
| 16 | 24 | QAOA p=12 | 61 | 1 | 1 | 0.03 | 0.06 | 0.06 |
| 16 | 24 | QAOA p=1 | 61 | 1 | 1 | 0.01 | 0.04 | 0.04 |
| 16 | 24 | SA | 61 | 1 | 1 | 0.05 | 0.07 | 0.07 |
| 17 | 32 | QAOA p=12 | 319 | 13 | 13 | 0.04 | 0.07 | 0.06 |
| 17 | 32 | QAOA p=1 | 319 | 13 | 13 | 0.01 | 0.05 | 0.04 |
| 17 | 32 | SA | 319 | 13 | 13 | 0.12 | 0.15 | 0.11 |
| 18 | 25 | QAOA p=12 | 1 | 1 | 1 | 0.04 | 0.04 | 0.04 |
| 18 | 25 | QAOA p=1 | 53 | 1 | 1 | 0.01 | 0.04 | 0.03 |
| 18 | 25 | SA | 1 | 1 | 1 | 0.05 | 0.05 | 0.05 |
| 19 | 29 | QAOA p=12 | 411 | 73 | 73 | 0.10 | 0.14 | 0.10 |
| 19 | 29 | QAOA p=1 | 443 | 99 | 99 | 0.07 | 0.11 | 0.07 |
| 19 | 29 | SA | 411 | 73 | 73 | 1.01 | 1.04 | 0.38 |
| 20 | 26 | QAOA p=12 | 328 | 130 | 130 | 0.17 | 0.22 | 0.15 |
| 20 | 26 | QAOA p=1 | 328 | 130 | 130 | 0.11 | 0.15 | 0.09 |
| 20 | 26 | SA | 328 | 130 | 130 | 2.21 | 2.25 | 0.94 |
| 21 | 26 | QAOA p=12 | 381 | 129 | 129 | 0.25 | 0.29 | 0.21 |
| 21 | 26 | QAOA p=1 | 381 | 129 | 129 | 0.14 | 0.18 | 0.10 |
| 21 | 26 | SA | 381 | 129 | 129 | 2.28 | 2.32 | 0.79 |
| 22 | 39 | QAOA p=12 | 1,777 | 730 | 730 | 0.94 | 1.05 | 0.46 |
| 22 | 39 | QAOA p=1 | 1,968 | 915 | 915 | 0.86 | 1.00 | 0.31 |
| 22 | 39 | SA | 1,777 | 730 | 730 | 13.59 | 13.71 | 2.89 |
| 23 | 47 | QAOA p=12 | 14,604 | 7,519 | 7,519 | 7.50 | 8.42 | 3.14 |
| 23 | 47 | QAOA p=1 | 14,736 | 7,611 | 7,611 | 7.14 | 8.09 | 2.76 |
| 23 | 47 | SA | 14,604 | 7,519 | 7,519 | 147.23 | 148.21 | 47.88 |
| 24 | 36 | QAOA p=12 | 4,643 | 2,598 | 2,598 | 4.14 | 4.49 | 2.39 |
| 24 | 36 | QAOA p=1 | 13,348 | 7,575 | 7,575 | 8.21 | 9.24 | 3.56 |
| 24 | 36 | SA | 4,388 | 2,425 | 2,425 | 55.47 | 55.82 | 19.73 |
| 25 | 36 | QAOA p=12 | 6,144 | 3,259 | 3,259 | 6.38 | 6.84 | 3.64 |
| 25 | 36 | QAOA p=1 | 14,352 | 9,400 | 9,400 | 11.23 | 12.57 | 4.81 |
| 25 | 36 | SA | 5,522 | 2,853 | 2,853 | 69.78 | 70.20 | 19.61 |
| 26 | 45 | QAOA p=12 | 7,084 | 3,392 | 3,392 | 9.88 | 10.39 | 6.00 |
| 26 | 45 | QAOA p=1 | 9,341 | 5,572 | 5,572 | 7.89 | 8.77 | 3.15 |
| 26 | 45 | SA | 6,631 | 3,166 | 3,166 | 82.42 | 82.89 | 14.83 |
| 27 | 37 | QAOA p=12 | 12,433 | 7,025 | 7,025 | 11.11 | 12.17 | 3.68 |
| 27 | 37 | QAOA p=1 | 12,403 | 6,998 | 6,998 | 10.31 | 11.60 | 3.53 |
| 28 | 50 | QAOA p=12 | 134,039 | 76,237 | 76,236 | 109.70 | 120.58 | 43.82 |
| 28 | 50 | QAOA p=1 | 138,186 | 78,663 | 78,662 | 105.09 | 118.26 | 43.41 |
| 29 | 62 | QAOA p=12 | 443,140 | 256,140 | 256,140 | 376.76 | 414.75 | 146.69 |
| 29 | 62 | QAOA p=1 | 459,708 | 265,898 | 265,898 | 355.95 | 401.25 | 142.37 |
| 30 | 59 | QAOA p=12 | 182,737 | 110,868 | 110,867 | 197.46 | 214.48 | 65.56 |
| 30 | 59 | QAOA p=1 | 186,980 | 113,252 | 113,251 | 167.78 | 188.34 | 59.69 |

Reading. (a) Every run of every backend proves the QOBLIB optimum. (b) For n <= 14 and n = 18
(SA, p = 12) the master's samples reach E* and every class root has LB > E* - Delta (at n <= 14
the roots have at most one free spin; at n = 18, LB_root = 3n/2 - 2 = E*, Theorem 2), so the run
is one node and one sampler call; at p = 1 the n = 18 master sample missed E* and the run took
53 nodes. (c) Node counts. SA and QAOA p = 12 visit the same tree at n = 16-23 (61, 319, 1, 411,
328, 381, 1,777, 14,604 nodes) and differ at n = 24-26 (4,388 against 4,643; 5,522 against 6,144;
6,631 against 7,084); p = 1 differs at n = 18, 19, 22-26 (up to 2.9x at n = 24: 13,348). By
Proposition 6 these differences come from when UB reaches E* and from which outermost spin nu
picks. At n = 28-30, where QAOA does not sample the master, p = 12 and p = 1 differ by 2-4 %.
(d) Time. The sampler dominates: one call per node, 1.0-2.9 ms per call averaged over a proof at
p = 12 (n = 22-30), 0.7-1.5 ms at p = 1, 14-26 ms with SA (n = 19-26); the node loop outside
the sampler costs 60-130 us per node (Python loop, propagation and bound in numba). Totals:
n = 30: 214 s at p = 12 (sampler 197 s) and 188 s at p = 1 (168 s); n = 26: 10.4 s (p = 12),
8.8 s (p = 1), 82.9 s (SA). (e) The PDI follows the sampler time and is below the total because
UB reaches E* early and the gap then closes as the lowest open key rises: n = 30: 66 s (p = 12),
60 s (p = 1); n = 26: 6.0 / 3.2 / 14.8 s. (f) Device-time substitution: a run makes one circuit
per sampled node, plus one for the skew restriction when the master of odd n is sampled, 1,000
shots each (`circuits`, `device_shots` in the trace rows); device runtime = non-sampler time +
circuits x device time per circuit (4.4).
No speedup is claimed; the comparison shows what the exact emulation of the quantum sampler
costs end to end and that the proof does not depend on which backend supplied the samples.

---------------------------------------------------------------------------------------------------

## 5. Risks and open questions

Settled by proof or measurement (details in the sections cited).
1. Branching candidate set (B.6): default "outermost"; on nodes with a fixed prefix and suffix it is eq. 9 with nu = 1
   (Proposition 5, PROVEN); "all" is complete too (Theorem A(v)) and was not measured with the
   node loop.
2. Reversal lex rule (B.5): on by default, applied to the class roots; keeps 2,112 of 4,096 roots
   at m0 = 7, no per-node cost.
3. Skew restriction (B.12): merged with master samples, so never worse; on by default with the
   QAOA sampler for odd n. Backends are never mixed: above the device size a Hamiltonian is not
   sampled (B.7), and 4.8 measures what this costs against SA.
4. QAOA angle scale (4.6): c = 0.875 (p = 1) and 0.625 (p >= 2) for nodes, 0.5 for the skew
   restriction; noise model at p = 1 measured.

Open.
5. Whether any sample-driven branching rule beats the balanced order from both ends inward on LABS (B.6).
6. Staircase of thresholds (B.9): proven (Theorem A(d)), not run by the node loop; a node- or
   time-limited run reports the open-node bound A(e) only.

Risks.
7. Proof reach. MEASURED (4.8) up to n = 30 with QAOA on the exact emulation and n = 26 with SA,
   1,000 shots per node, at most 415 s per run; one sampler call per node sets the time. No
   extrapolation is made here.
8. UB reach. Beyond proven reach UB is the answer; it comes from the sampled nodes and the
   leaves within the budget. How close it gets to E* at n > 30 is not measured.
9. Gap size. The certified LB is the open-node bound A(e); how it grows with the node budget at
   n > 30 is not measured.
10. QAOA hardware. Today's two-qubit budget (~10^3 gates, ESTIMATE) admits only p = 1 on small
   Hamiltonians (4.3), where p = 1 gains a factor 2-7 over uniform sampling even without noise
   (4.6). A proof samples every node from the master (n qubits) down to F = 1, so on a device
   only the nodes within `max_size` are sampled. Hardware runs are interface demonstrations.

---------------------------------------------------------------------------------------------------

## 6. Conjectures and empirical statements (not used in the proofs)

C1. Growth of the search tree ~1.73^n (Packebusch-Mertens 2016 numerical fit 1.729/1.727; their
    n = 66 CPU-days are an extrapolation, "N <= 43 in less than an hour on a laptop" their
    wording). Not used by the algorithm.
C2. A coarser lattice for the bound is FALSE in general: at random nodes for n = 12, 13, 15, 16,
    LB_0 and LB take both residues mod 8 that the energies take. The prune test stays LB > T.
C3. Two-point magnetization bound (S = sum s_i, S' = sum (-1)^i s_i give sum_k C_k = (S^2 - n)/2
    and sum_k (-1)^k C_k = (S'^2 - n)/2; integer water-filling per parity class, minimum over the
    feasible (S, S') box): proven valid as a relaxation in our prototypes; not part of the
    algorithm.
C4. Lagrangian pair bound (per fixed magnetization S, E = sum_k (C_k^2 - mu C_k) + mu (S^2 - n)/2,
    min over S of max over mu of the pair-class minima): proven valid by weak duality in our
    prototypes, needs exact dyadic arithmetic for the prune test; not part of the algorithm.
C5. (empirical) Skew-symmetric optimum for odd n: equal to the QOBLIB optimum for 14 of 24 odd n
    in 19..65 (B.12). Used only as an extra source of samples.
C6. (empirical) SA / memetic-tabu time-to-best-known ~1.34^n (Boskovic et al. 2017).
C7. (empirical) The balanced order from both ends inward is within ~20 % of the optimal static branching order
    (exact DP over orders at n = 12). Used only for the branching heuristic.
C8. (empirical) QAOA angle transfer: gamma_l = c g_l / n with the parent length n; c = 0.875
    (p = 1) and 0.625 (p >= 2) for node Hamiltonians, 0.5 for the skew restriction (4.6). Used
    only by the sampler.
