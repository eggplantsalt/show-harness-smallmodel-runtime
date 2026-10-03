# Runtime V3 Evidence Ledger

Claims are separated from the numerical measurements that originally
motivated them. `VERIFIED` means the stated contract was directly exercised
within its described scope; `PARTIAL` means only some cases or layers passed;
`INVALIDATED` means later visual or semantic evidence disproved the claim;
`NOT TESTED` means no evidence is available.

| Claim | Status | Evidence and boundary |
|---|---|---|
| Single Authority | VERIFIED | Runtime V3 routes each physical action through Arbiter and Executor. Existing suite and one-approval regression test remain in force; the M3.8 semantic components have no controller, Arbiter, or Executor interface. |
| Bounded micro-motion | VERIFIED | Existing M3.4/M3.7 calibration and execution evidence; one approved same-direction motion is bounded and re-observed before another option is selected. M3.8 does not change this contract. |
| SceneMotionReady | VERIFIED | M3.7 passed on all six tested tasks. M3.8 waits for this gate before semantic region proposal when using the generalized observer. |
| EntityObservationReady | VERIFIED | M3.7 passed in its observed, correctly associated cases; stable observations are required after selection. M3.8 additionally tracks selected masks using SAM point prompts without repeating Qwen selection each tick. |
| Salad dressing ALIGN | VERIFIED | Existing task-2 ALIGN evidence remains valid for its reviewed target binding and unchanged physical contract. M3.8 reset-frame selection chose an orange distractor, while the separately captured Runtime readiness frame selected the green-capped target and established a valid reference. |
| Milk cross-object ALIGN | VERIFIED | M3.7 task 7 passed 6/6 episodes with 54 positive ALIGN effects and unchanged physical code. |
| Alphabet soup correct-target ALIGN | INVALIDATED | M3.6's stable selected mask was the neighboring Milk carton, not the requested soup can. The historical pixel improvements are retained as measurements but do not establish soup ALIGN transfer. M3.8 abstained in both reset-only and Runtime readiness diagnostics; the reset-only pool included a soup-can candidate, while the Runtime readiness pool missed it. |
| Generic semantic grounding | PARTIAL | On six reset-only development frames, the generic K=4 pool included a human-confirmed target mask in 4/6, versus 3/6 for direct text-SAM; Qwen selected an incorrect entity on both SELECT outputs. The separate Runtime readiness pass selected the correct salad dressing and milk masks and safely abstained on four tasks. On the frozen heldout set, target masks were visually present in 12/12, but Qwen selected the correct one in only 4/12 and selected the wrong entity in 8/12. All 12 reached stable Runtime references, which shows that temporal validity alone cannot verify semantic identity. |
| Qwen semantic compiler | PARTIAL | M3.7 compiled salad dressing and milk correctly. M3.8 Stage C compiled the two eligible heldout instructions (`ketchup`, `orange juice`) in two calls; candidate-ID re-grounding blocked the ketchup episode before ALIGN. |
| Agent as semantic coprocessor | PARTIAL | M3.7 and M3.8 assign Qwen semantic compilation/selection only; M3.8 records zero Qwen physical actions. Stage C completed three orange-juice episodes, while the heldout selector chose the wrong entity in 8/12 Stage A episodes. |
| NearTarget | NOT TESTED | No deployable NearTarget evidence was implemented or evaluated in M3.8. |
| Grasp | NOT TESTED | No grasp behavior or holding contract was implemented or evaluated in M3.8. |

## M3.8 formal grounding and ALIGN results

The frozen heldout manifest covered tasks 3 (BBQ sauce), 4 (ketchup), 5
(tomato sauce), and 9 (orange juice), each at init states 0–2, seed 0. Stage A
completed all 12 episodes with four candidates per frame, two Qwen calls per
initialization, 12/12 `EntityObservationReady`, 12/12 valid Runtime references,
and zero ALIGN actions. Post-hoc RGB visual review found target masks in 12/12
pools, 4/12 correct semantic selections, 8/12 wrong-entity selections, and no
`NO_MATCH` outputs. Per task: BBQ sauce 0/3 correct, ketchup 1/3, tomato sauce
0/3, and orange juice 3/3. The four episodes eligible for ALIGN were ketchup
init state 2 and orange juice init states 0, 1, and 2.

Stage B executed three eligible episodes: ketchup init 2 and orange juice init
0 and 2. The orange juice init 1 rerun failed closed because the selected
candidate ID did not reproduce the visually audited ID. All 18 executed steps
had positive effects; all 3 executed episodes were monotonic; mean normalized
error reduction was 17.37%. Ketchup improved 15.35% over 6 RIGHT × 9 mm steps;
orange juice improved a mean 18.37% over 12 DOWN × 9 mm steps.

Because Stage B had positive effects, Stage C ran the compiled semantic route.
The compiler used 2 calls for the two task instructions. Orange juice completed
all 3 episodes with 18/18 positive steps, 3/3 monotonic episodes, and 18.46%
mean normalized error reduction using DOWN × 9 mm. Ketchup init 2 failed its
candidate-ID re-grounding gate and executed zero steps. Stage B/C each recorded
zero physical-contract changes and no privileged runtime inputs.

| Failure class | M3.8 observation |
|---|---|
| `GROUNDING_PROPOSAL_MISS` | 0/12 heldout candidate pools; 2/6 Runtime development frames (soup, cream cheese) |
| `SEMANTIC_SELECTION_WRONG` | 8/12 heldout Stage A episodes |
| `SEMANTIC_SELECTION_NO_MATCH` | 0/12 heldout; 4/6 Runtime development frames |
| `ENTITY_OBSERVATION_NOT_READY` | 0/12 heldout; four development abstentions stayed unready |
| `IDENTITY_FAILURE` | One fail-closed candidate-ID mismatch in each of Stage B and Stage C |
| `SCENE_MOTION_NOT_READY` | 0/12 heldout episodes |
| `PHYSICAL_ALIGNMENT_FAILURE` | 0 among executed steps; identity-gate failures executed no steps |

Full machine-readable reports and the reviewed candidate/reference images are
archived in `experiments/runtime_v3/m3_8/`.
