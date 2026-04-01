# A2RL Tactical Planning System: Technical Overview

This document provides a technical summary of the `tactical_acados` planning module, its architecture, and current known issues to facilitate expert review and further development.

## 1. Module Objective
The `tactical_acados` folder implements a **Hierarchical Tactical Planner**. Its goal is to bridge the gap between high-level racing strategy (overtaking, defending, following) and low-level optimization-based control (ACADOS MPC). It serves as the primary "decision-making" engine for the vehicle.

## 2. Core Architecture
The system follows a three-layer hierarchy:
1.  **Behavioral Layer** (`heuristic_policy.py`): Decides the mode (e.g., `FOLLOW`, `OVERTAKE`) based on the current racing situation.
2.  **Guidance Layer** (`planner_guidance.py` / `a2rl_obstacle_carver.py`): Translates the mode into numerical constraints (corridors) and targets (speed, lateral bias) for the solver.
3.  **Optimization Layer** (`acados_planner.py`): Solves the constrained optimal control problem.

---

## 3. File-by-File Breakdown

### Primary Planning Logic
- **`acados_planner.py`**: The solver wrapper. Manages the ACADOS OCP instance, sets stage-wise bounds (`lbx`, `ubx`) for the corridor, and handles solver failures with various fallback strategies.
- **`planner_guidance.py`**: The "Tactical-to-Planner" mapper. It computes the dynamic left/right boundaries of the corridor and speed scales based on the active tactical mode.
- **`a2rl_obstacle_carver.py`**: A baseline "hard-blocking" logic. It pushes track boundaries to essentially "remove" the side of the track occupied by an opponent.

### Tactical Components
- **`tactical_action.py`**: Data structures for Modes, Intentions, and the `PlannerGuidance` object.
- **`follow_module.py`**: Specialized logic for the `FOLLOW` state, including gap-keeping and speed matching.
- **`safe_wrapper.py`**: A terminal-state safety checker that validates if a chosen tactic is plausible before execution.

### Support & Visuals
- **`visualizer_tactical.py`**: Renders the 300m planning horizon, predicted trajectories, and the dynamic corridor (Cyan dashed lines).
- **`observation.py`**: Standardizes opponent and ego telemetry into a format the planners understand.
- **`config.py`**: Central repository for all tuning parameters.

---

## 4. Interaction with External Modules
The `tactical_acados` folder is self-contained but relies on the parent project for:
- **`Track3D`**: Geometry, curvature, and racing-line data.
- **`LocalRacinglinePlanner`**: Generation of curvature-aware reference velocities ($V_{max}$ profiles).
- **`CasADi` models**: Physics/Dynamics models used by the solver.

---

## 5. Current Unresolved Issues

### A. Corridor Infeasibility (The "Pinch" Problem)
The primary stability bottleneck is the **Hard-Constraint Side Blocking**. If the combined effect of track narrowing, opponent safety margins (currently 1.5m), and vehicle width (1.93m) exceeds the physical track width, the ACADOS solver enters `status 4 (Infeasible)`.
- **Observation**: At start-up or in hairpins, selecting a side can result in a "negative" width corridor. 
- **Compromise**: We currently reset to full track if a pinch is detected, but this creates a "flicker" that causes solver divergence.

### B. Prediction/Spatial Horizon Mismatch
We plan for a spatial 300m horizon. If the car is slow, this corresponds to a long temporal duration. However, opponent predictions only last ~3.75s. Handling the extrapolation of opponents at long horizons without creating "Ghost Obstacles" is still a delicate balance.

### C. Initialization during Start-up
At $V=0$, the kinematic time-steps are poorly defined. The system relies on a `V_start` fallback (5.0m/s) to compute node times. If an opponent is very close at start, the initial decision can lock the solver into an infeasible state before the car even moves.

## 6. Development Direction
The system would benefit from:
1.  **Soft Constraints**: Moving from hard `lbx/ubx` boundaries to soft constraints (Slacks) for opponent avoidance to allow "slaying" of the safety margin instead of solver collapse.
2.  **Temporal Hysteresis**: Preventing the side-selection decision from flipping between Left/Right in successive frames.
3.  **Width-Aware Selection**: A strategy that prioritizes the side with the maximum available room rather than just following the previous lateral position.
