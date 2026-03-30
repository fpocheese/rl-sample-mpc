# EXECUTION PROMPT FOR CLAUDE CODE / COPILOT
# Project: Add Game-Theoretic Tactical RL Layer on top of ACADOS local planner in `rl-sample-mpc`

You are modifying the repository `rl-sample-mpc`.

Your task is to convert the current project into a **two-layer tactical decision + ACADOS local planning system** for autonomous racing on **YAS North**, following the already-defined theory-guided game-theoretic RL method.

You must read the existing code carefully and implement the method in an **engineering-feasible but theory-consistent** way.

---

# 0. Core Goal

Transform the current repository into a system with the following architecture:

```text
Track / GG / Vehicle / Opponent setup
    ↓
Observation construction
    ↓
Tactical decision layer
    ↓
Safe tactical wrapper
    ↓
Tactical-to-planner mapping
    ↓
ACADOS local planner (only online planner)
    ↓
30-point trajectory over 3.75 s (dt = 0.125 s)
    ↓
Perfect tracking simulation
```

The final system must support:

1. **No-sampling online planning**: only ACADOS local planner is used online.
2. **Two three-car YAS North scenarios**:
   - normal corner cluster: before Turn 1 to after Turn 4
   - high-curvature corner cluster: before Turn 6 to after Turn 8
3. **Tactical decision layer** with interface consistent with the paper:
   - discrete tactical mode/intention
   - continuous aggressiveness / terminal preference
   - push-to-pass trigger
4. **Safe RL integration**:
   - even if RL is untrained and outputs random actions, the safe wrapper must keep tactical outputs within a planner-feasible region
   - planner must still run
   - vehicle must still move

---

# 1. Hard Rules

## 1.1 Do not use the sampling-based planner online
Keep old files, but do not use the sampling planner for new online planning.

The new online pipeline must use only:
- `Track3D`
- `GGManager`
- `GlobalRacinglinePlanner`
- `LocalRacinglinePlanner`
- ACADOS OCP

Do NOT generate final tracked trajectories with `LocalSamplingPlanner`.

---

## 1.2 Online output format is fixed
The online planner output must always be postprocessed to:

- horizon = **3.75 s**
- number of points = **30**
- dt = **0.125 s**

The final trajectory object used for propagation must contain at least:
- `t`
- `s`
- `n`
- `V`
- `chi`
- `ax`
- `ay`
- `x`
- `y`
- `z`

---

## 1.3 RL can never break planner feasibility
This is the most important rule.

The RL tactical policy is not allowed to directly control the planner in an unconstrained way.

All RL outputs must go through:
1. tactical action parser
2. safe tactical wrapper / projection / clipping
3. tactical-to-planner mapping
4. fallback if necessary

At every time step, planner feasibility and vehicle drivability have priority over RL freedom.

If the RL action is unsafe or planner-infeasible:
- clip it
- sanitize it
- or replace it with fallback action

---

# 2. Method to Implement: Theoretical Layer → RL Layer → Planner Layer

You must implement the tactical layer according to the following theory-consistent abstraction.

---

## 2.1 Tactical hybrid action

Use the hybrid tactical action:

\[
a_k = (d_k, c_k)
\]

where

\[
d_k = (m_k, \ell_k)
\]

is the discrete tactical branch, and

\[
c_k = (\alpha_k, \rho_k, b_k)
\]

is the continuous / binary tactical branch.

Here:

- \(m_k\): tactical mode  
  one of:
  - `follow`
  - `overtake`
  - `defend`
  - `recover`

- \(\ell_k\): lateral intention  
  one of:
  - `center`
  - `left`
  - `right`

- \(\alpha_k \in [0,1]\): aggressiveness

- \(\rho_k\): terminal/planner preference vector

- \(b_k \in \{0,1\}\): push-to-pass trigger

---

## 2.2 Continuous preference vector

Implement the continuous tactical preference vector as

\[
\rho_k =
\begin{bmatrix}
\rho_k^{v} \\
\rho_k^{n} \\
\rho_k^{s} \\
\rho_k^{w}
\end{bmatrix}
\]

with the following meaning:

- \(\rho_k^{v}\): desired speed bias
- \(\rho_k^{n}\): terminal lateral bias
- \(\rho_k^{s}\): safety-margin scale
- \(\rho_k^{w}\): interaction weight scale

Engineering ranges must be bounded explicitly. Use something like:

- \(\rho_k^{v} \in [-0.25, 0.25]\)
- \(\rho_k^{n} \in [-1.5, 1.5]\) meters
- \(\rho_k^{s} \in [0.7, 1.5]\)
- \(\rho_k^{w} \in [0.5, 2.0]\)

You may adjust exact bounds if needed for stability, but keep them explicit in config.

---

## 2.3 Tactical game-theoretic objective

The high-level tactical decision is based on the ego utility

\[
U_e = U_{\mathrm{prog}} + U_{\mathrm{race}} + U_{\mathrm{safe}} + U_{\mathrm{term}} + U_{\mathrm{ctrl}}
\]

Implement these terms in code as follows.

### 2.3.1 Progress utility
\[
U_{\mathrm{prog}} = w_{\mathrm{prog}} \, \Delta s_e
\]

Code interpretation:
- forward progress along track
- measured over the current tactical decision step or planning horizon

---

### 2.3.2 Racing interaction utility
Use a simplified but explicit racing gain relative to opponents:

\[
U_{\mathrm{race}} =
w_{\mathrm{race}}
\sum_{j \in \mathcal N_{-e}}
\left[
(\Delta s_{e,j}^{k+H} - \Delta s_{e,j}^{k})
\right]
\]

where

\[
\Delta s_{e,j}^{k} = s_e^k - s_j^k
\]

Code interpretation:
- if ego gains longitudinal position relative to opponent(s), reward increases
- this term should encourage overtake / defense behavior

Use a short-horizon approximation based on opponent prediction and planner rollout.

---

### 2.3.3 Safety utility
Use a barrier-like safety signal:

\[
U_{\mathrm{safe}}
=
- w_{\mathrm{safe}}
\sum_{j\in\mathcal N_{-e}}
\phi_{\mathrm{safe}}(d_{e,j})
\]

where \(d_{e,j}\) is a distance / TTC / Frenet safety metric.

Code interpretation:
- use negative penalty for unsafe closeness
- use large penalty for collision / overlap / severe proximity
- do NOT rely only on this reward for safety; safety must also be enforced by action sanitization

---

### 2.3.4 Terminal recoverability utility
This is important. Implement a simplified but explicit terminal recoverability signal:

\[
U_{\mathrm{term}}
=
w_{\mathrm{term}}
\Phi_{\mathrm{rec},e}\!\left(
x_e^N,\,
\mathcal R_e(X_k;\vartheta_k)
\right)
\]

Code interpretation:
- evaluate whether the resulting ACADOS trajectory ends in a state that is easy to continue planning from
- proxy features may include:
  - terminal lateral offset near feasible corridor center
  - terminal heading not too large
  - terminal speed not too aggressive for local curvature
  - planner solve status success
- if the planner becomes near-infeasible or fallback is used, reduce this term strongly

---

### 2.3.5 Control / tactical smoothness utility
\[
U_{\mathrm{ctrl}}
=
- w_{\mathrm{ctrl}} \|a_k - a_{k-1}\|^2
\]

Code interpretation:
- discourage violent tactical switching
- discourage oscillation between left/right or overtake/recover every step

For hybrid action:
- discrete difference: use 0/1 penalty
- continuous difference: use squared norm

---

## 2.4 Conditioned continuous Stackelberg-IBR approximation

Do NOT implement a full exact Stackelberg game solver online.  
Implement a **conditioned continuous approximation** consistent with the theory:

- discrete tactical candidates are finite and enumerated
- for each candidate discrete tactic, approximate opponent response with a lightweight IBR-style predictor
- then derive:
  - safe set approximation
  - tactical prior
  - auxiliary game value target

Use the following engineering approximation:

### Step A: discrete candidate set
Define a small finite discrete set:

```text
D = {
  follow_center,
  overtake_left,
  overtake_right,
  defend_left,
  defend_right,
  recover_center
}
```

You may prune invalid combinations.

### Step B: opponent response approximation
For a given ego candidate action, approximate opponent response by a cheap rule-based IBR-style function:

\[
\hat a_{-e}^{\mathrm{IBR}}(o_k, a_e)
\]

Engineering implementation:
- opponents use same vehicle dynamics and same ACADOS local planner
- but their tactical response can initially be heuristic
- response must depend on ego action approximately:
  - if ego attacks left, opponent may defend left or hold line
  - if high-curvature region, opponent may prefer follow/recover

### Step C: robust tactical value
For a candidate tactical action \(a_e\), define approximate robust tactical value:

\[
\underline{\mathcal V}_e(a_e \mid o_k)
\]

Code interpretation:
- planner-based tactical score under predicted opponent response
- combines progress, interaction, safety, terminal recoverability
- if planner fails or fallback triggered, value must be strongly reduced

You do NOT need exact min-max optimization.  
Use a practical approximation:
- evaluate ego tactical candidate against predicted opponent tactical response(s)
- optionally use worst of a few opponent candidate responses

---

# 3. How theory enters RL

You must implement the RL layer as a **theory-guided hybrid tactical RL** method.

---

## 3.1 RL algorithm choice

Use **PPO** as the base RL algorithm.

But because the action is hybrid, implement a **hybrid PPO policy**:

- discrete branch for \(d_k\)
- continuous branch for \((\alpha_k, \rho_k)\)
- binary branch for \(b_k\) or fold it into the discrete branch / separate Bernoulli head

You do NOT need to invent a new optimizer.  
Use PPO with a factorized policy architecture.

Recommended implementation structure:

```text
policy(obs):
    discrete_logits -> tactical discrete distribution
    continuous_mean, continuous_logstd -> Gaussian for alpha and rho
    p2p_logit -> Bernoulli
    value head -> state value
    auxiliary game-value head -> tactical game value
```

If you prefer cleaner engineering, you may represent P2P as an extra discrete action bit outside PPO’s Gaussian branch.

---

## 3.2 Safe action masking in RL

This must be implemented.

Define an approximate safe tactical set

\[
\widehat{\mathcal A}_e^{\mathrm{safe}}(o_k)
\]

based on:
- track bounds
- opponent occupancy approximation
- local curvature / speed compatibility
- planner feasibility heuristics
- P2P availability constraints

Then define the safe discrete set

\[
\mathcal D_e^{\mathrm{safe}}(o_k)
\]

as the discrete candidates that admit at least one feasible continuous tactical parameter.

In code:

- mask discrete logits before sampling
- if all actions are masked, fall back to conservative action:
  - `recover_center` or `follow_center`

Continuous branch:
- sample continuous parameters
- then clip/project them into admissible bounded set
- then run safe wrapper again

This is mandatory.

---

## 3.3 Theory-guided prior

Implement the factorized theory-guided prior:

\[
\pi_{\mathrm{th}}(a_k \mid o_k)
=
\pi_{\mathrm{th}}^{\mathrm d}(a_k^{\mathrm d}\mid o_k)\,
\pi_{\mathrm{th}}^{\mathrm c}(a_k^{\mathrm c}\mid a_k^{\mathrm d}, o_k)
\]

### 3.3.1 Discrete prior
Use a Boltzmann distribution over tactical game value:

\[
\pi_{\mathrm{th}}^{\mathrm d}(a_k^{\mathrm d}\mid o_k)
=
\frac{
\exp(\mathcal G_{\mathrm d}(o_k,a_k^{\mathrm d})/\tau_d)
}{
\sum_{\bar a^{\mathrm d}\in\mathcal D_e^{\mathrm{safe}}(o_k)}
\exp(\mathcal G_{\mathrm d}(o_k,\bar a^{\mathrm d})/\tau_d)
}
\]

where

\[
\mathcal G_{\mathrm d}(o_k,a_k^{\mathrm d})
=
\underline{\mathcal V}_e\!\left(
(a_k^{\mathrm d}, c_{\mathrm{th}}^\star(a_k^{\mathrm d},o_k))
\mid o_k
\right)
\]

Engineering implementation:
- for each discrete candidate, compute tactical score using planner and opponent approximation
- temperature `tau_d` configurable
- use masked safe discrete set only

### 3.3.2 Continuous prior
Use a local target around theory-guided continuous parameters:

\[
c_{\mathrm{th}}^\star(a_k^{\mathrm d},o_k)
\]

Engineering implementation:
- for each discrete tactical candidate, generate a handcrafted or search-based continuous tactical target:
  - aggressiveness target
  - speed bias target
  - terminal lateral bias target
  - safety / interaction scales
- this can initially be heuristic, but must depend on geometry and opponents

Then implement continuous prior regularization using regression, not full KL.

---

## 3.4 PPO loss with theory regularization

The learning objective must be PPO plus theory-guided regularization:

\[
\mathcal L
=
\mathcal L_{\mathrm{PPO}}
+
\lambda_d \mathcal L_{\mathrm{prior}}^{d}
+
\lambda_c \mathcal L_{\mathrm{prior}}^{c}
+
\lambda_g \mathcal L_{\mathrm{game}}
\]

Implement:

### 3.4.1 Discrete prior loss
\[
\mathcal L_{\mathrm{prior}}^{d}
=
D_{\mathrm{KL}}
\left(
\pi_\theta^{\mathrm d}(\cdot\mid o_k)
\parallel
\pi_{\mathrm{th}}^{\mathrm d}(\cdot\mid o_k)
\right)
\]

### 3.4.2 Continuous prior loss
\[
\mathcal L_{\mathrm{prior}}^{c}
=
\left\|
\mu_\theta^{\mathrm c}(a_k^{\mathrm d}, o_k)
-
c_{\mathrm{th}}^\star(a_k^{\mathrm d}, o_k)
\right\|_{W_c}^{2}
\]

### 3.4.3 Auxiliary game-value head loss
Define auxiliary head:

\[
\hat{\mathcal G}_\psi(o_k, a_k)
\approx
\underline{\mathcal V}_e(a_k \mid o_k)
\]

and train it with

\[
\mathcal L_{\mathrm{game}}
=
\left(
\hat{\mathcal G}_\psi(o_k,a_k)
-
\underline{\mathcal V}_e(a_k\mid o_k)
\right)^2
\]

---

# 4. RL reward to implement

The RL reward must reflect the tactical utility but in step-wise engineering form.

Implement:

\[
r_k
=
r_k^{\mathrm{prog}}
+
r_k^{\mathrm{race}}
+
r_k^{\mathrm{safe}}
+
r_k^{\mathrm{term}}
+
r_k^{\mathrm{ctrl}}
+
r_k^{\mathrm{p2p}}
\]

Use the following coding structure.

### 4.1 Progress reward
- positive reward for forward `delta_s`

### 4.2 Racing reward
- positive reward if ego improves relative position to opponents
- bonus for successful overtake completion
- optional small bonus for holding defense when threatened

### 4.3 Safety reward
- strong penalty for collision
- strong penalty for leaving track
- penalty for very small gap / unsafe TTC

### 4.4 Terminal recoverability reward
- reward planner success
- reward low terminal curvature mismatch / feasible terminal state
- penalize fallback use
- penalize planner failure

### 4.5 Tactical smoothness reward
- penalize discrete tactical switching
- penalize violent continuous action changes

### 4.6 P2P reward
- no unconditional bonus for simply pressing P2P
- reward only if P2P usage improves tactical outcome
- penalize wasting it in bad locations

---

# 5. Planner-side tactical coupling to implement

The tactical layer must influence the ACADOS planner through a planner-guidance mapping.

Implement a mapping

\[
\vartheta_k = \mathcal T(a_k, o_k)
\]

where `PlannerGuidance` must at least contain:

- target side / desired corridor side
- terminal lateral bias
- speed scale / speed cap
- safety margin
- interaction weight
- follow target or overtake target id
- p2p active flag

This guidance must modify the ACADOS planner via:

1. **corridor shaping**
2. **terminal lateral target**
3. **speed bias**
4. **interaction penalty or opponent clearance preference**
5. **fallback conservative mode**

---

## 5.1 Corridor shaping
Use opponent prediction to carve the feasible lateral corridor by stage.

This is the main practical substitute for complex nonconvex collision constraints.

### Required behavior
- if action is `overtake_left`, preserve left-side corridor if possible
- if action is `overtake_right`, preserve right-side corridor if possible
- if action is `follow_center`, corridor stays conservative and speed reduced if blocked
- if action is `recover_center`, enlarge safety margin and recenter

---

## 5.2 Terminal tactical bias
Use `rho^n` / `target_terminal_n` to bias the trajectory end laterally.

---

## 5.3 Speed bias
Use `rho^v`, `alpha`, and P2P state to adjust speed target / speed cap / progress pressure.

---

# 6. Opponent design

Opponents must:
- use same vehicle params
- use same ACADOS planner
- have simple tactical policy
- be capable of follow / occupy / defend / simple overtake

Do not leave them as static or raceline-only obstacles.

They may initially use:
- heuristic tactical policy
- same safe wrapper
- no RL

---

# 7. Push-to-pass implementation

Ego only.

Implement:
- +50 hp equivalent effect
- one activation only
- duration = 15 s

Do not model it as teleportation.
Implement it through bounded longitudinal performance enhancement.

P2P must be a tactical decision variable, but still pass through safety wrapper.

---

# 8. Scenarios

Create two three-car scenarios on YAS North:

## Scenario A
normal corner sector:
- from before Turn 1
- to after Turn 4

## Scenario B
high-curvature sector:
- from before Turn 6
- to after Turn 8

Use the smoothed YAS track data already in the repo setup or current YAS track variant used in your branch.

All scenario boundaries and spawn positions must be stored in YAML, not hardcoded.

---

# 9. Code implementation order

Follow this exact implementation order.

## Phase 1: ACADOS-only baseline
1. Remove online dependence on sampling planner
2. Build new ACADOS-only simulation entry
3. Output 30-point, 3.75 s trajectories
4. Ensure ego can complete a lap

## Phase 2: Tactical interface
5. Add tactical action data structure
6. Add tactical safe wrapper
7. Add tactical-to-planner mapping
8. Add heuristic tactical policy
9. Add random tactical policy
10. Ensure random policy still keeps planner feasible

## Phase 3: Opponent-aware racing
11. Add opponent tactical behavior
12. Add corridor shaping against opponents
13. Run two 3-car YAS scenarios

## Phase 4: RL integration
14. Add gym-like tactical environment
15. Add hybrid PPO policy
16. Add safe masking
17. Add factorized theory prior
18. Add auxiliary game-value head
19. Add training loop
20. Train first on Scenario A, then Scenario B

---

# 10. Deliverables

You must produce code that includes at least:

- new online simulation entry using ACADOS only
- tactical action class / dict / dataclass
- safe tactical wrapper
- tactical-to-planner mapping
- heuristic and random tactical policies
- opponent-aware scenario logic
- two YAS scenario yaml files
- RL environment
- hybrid PPO policy implementation
- training-ready logs and evaluation metrics

---

# 11. Final engineering rule

If at any point there is a conflict between:
- RL freedom
- tactical aggressiveness
- planner feasibility
- vehicle ability to keep moving

you must always choose:
1. planner feasibility
2. vehicle drivability
3. safety
4. tactical aggressiveness
5. RL freedom

The RL layer must only improve performance on top of a working safe planning stack.

---

# 12. What to do now

Start by reading the current repository structure carefully and then implement the system incrementally.

Do NOT rewrite everything from scratch.

Prefer:
- adding new files
- extending current files
- keeping old code for reference
- building a clean new ACADOS-only tactical pipeline
