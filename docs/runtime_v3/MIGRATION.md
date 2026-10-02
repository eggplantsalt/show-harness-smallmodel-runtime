# Runtime V3 Migration Map

Legacy files remain available at the frozen `legacy-full-harness-0928` tag.
The V3 active path is a separate package and does not import the policy modules
below.

| Legacy capability | V3 responsibility |
| --- | --- |
| VisualHarness perception and evidence | Observer adapter; it may reuse image/SAM/geometry utilities but does not authorize actions |
| VerifiedRuntime geometry and residual utilities | OptionGenerator input utilities; only verified candidates become options |
| VLM controller / Qwen client | Selector adapter; receives compact state and option IDs |
| Multiple stage guards, option gate, commit guard | Arbiter's single precondition/freshness/boundedness check |
| RecoveryPlugin action proposals | Future recovery options, subject to the same Arbiter |
| VisualRoute | Optional geometry evidence provider; no route intent or direct action control in V3 |
| Placement review | Future evidence producer; cannot modify a selected action |
| Recursive reflection | Not imported or called by V3 |
| Legacy textual memory and RSI lessons | Future structured experience records; no automatic learning in this phase |
| LIBERO environment and atomic controller | Reused through `LiberoPrimitiveBackend`, which is called only by Executor |
| EpisodeLogger and image acquisition | Reusable after a V3 adapter supplies evidence and records V3 events |

Legacy policy/state modules are intentionally not imported by the V3 package or
runner. Reuse of a low-level utility does not grant that utility control
authority.
