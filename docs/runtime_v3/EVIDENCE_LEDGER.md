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
| Alphabet soup correct-target ALIGN | INVALIDATED | M3.6's stable selected mask was the neighboring Milk carton, not the requested soup can. The historical pixel improvements are retained as measurements but do not establish soup ALIGN transfer. M3.8 development now abstains on this task, although a soup-can candidate is present. |
| Generic semantic grounding | PARTIAL | On six reset-only development frames, the generic K=4 pool included a human-confirmed target mask in 4/6, versus 3/6 for direct text-SAM; Qwen selected an incorrect entity on both SELECT outputs. The separate Runtime readiness pass selected the correct salad dressing and milk masks and safely abstained on four tasks. On the frozen heldout set, target masks were visually present in 12/12, but Qwen selected the correct one in only 4/12 and selected the wrong entity in 8/12. All 12 reached stable Runtime references, which shows that temporal validity alone cannot verify semantic identity. |
| Qwen semantic compiler | PARTIAL | M3.7 Stage C compiled the salad-dressing and milk instructions correctly. M3.8 heldout compiler results depend on Stage B eligibility and will be recorded below after evaluation. |
| Agent as semantic coprocessor | PARTIAL | M3.7 showed semantic-only task compilation for tasks 2 and 7 with zero Qwen physical choices. M3.8 uses a strict semantic-region proposer and candidate selector, while the Runtime owns SAM tracking/readiness and has recorded zero physical actions in development; reset-frame selection errors mean generalization is not established. |
| NearTarget | NOT TESTED | No deployable NearTarget evidence was implemented or evaluated in M3.8. |
| Grasp | NOT TESTED | No grasp behavior or holding contract was implemented or evaluated in M3.8. |

## M3.8 formal grounding and ALIGN results

The frozen heldout manifest covered tasks 3 (BBQ sauce), 4 (ketchup), 5
(tomato sauce), and 9 (orange juice), each at init states 0–2, seed 0. Stage A
completed all 12 episodes with four candidates per frame, two Qwen calls per
initialization, 12/12 `EntityObservationReady`, 12/12 valid Runtime references,
and zero ALIGN actions. Post-hoc RGB visual review found 4/12 correct semantic
selections, 8/12 wrong-entity selections, and no `NO_MATCH` outputs. The four
episodes that qualify for Stage B are task 4 init state 2 and task 9 init states
0, 1, and 2. Only those episodes may count as correctly grounded ALIGN evidence.
