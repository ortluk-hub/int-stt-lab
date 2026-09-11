# Mapping Async Delegation Delivery vs Kanban Creation

## Overview
This document maps the current **async delegation** mechanism (`delegate_task`) against the **Kanban task creation** workflow (`kanban_create` and related tools) within the Hermes/Int‑STT‑Lab system.

## 1. Async Delegation (`delegate_task`)

| Aspect | Details |
|--------|---------|
| **Invocation** | `delegate_task` is called from a running worker (e.g., Ada, Nemo). It spawns one or more **sub‑agents** with isolated contexts, each receiving its own `task_id`‑like identifier that lives only for the lifespan of the sub‑process.
| **Execution Model** | Sub‑agents run **asynchronously**. The parent worker continues after the call returns a *subagent transcript ID* and can later `steer` or `stop` them. Results are delivered as a **single final transcript** that the parent can read.
| **Result Persistence** | Results are **ephemeral** – stored only in the transcript log. They are not automatically persisted in the Kanban DB. The parent must manually record any important outcome (e.g., write a file, create a Kanban task, or update memory).
| **Failure & Recovery** | Sub‑agents can be stopped, and the parent can `steer` them with new prompts. If a sub‑agent crashes, the parent discovers it via the transcript `error` field.
| **Review** | No explicit review step. The parent decides whether the result is acceptable and may request a review manually by creating a Kanban task.
| **Heartbeat** | Not required – the sub‑agent is managed by the Hermes agent runtime. The parent can monitor its status through the `delegate_task` API.

## 2. Kanban Task Creation (`kanban_create`)

| Aspect | Details |
|--------|---------|
| **Invocation** | `kanban_create` creates a **persistent task** in the shared SQLite Kanban DB. The task gets an ID, status (`todo`, `running`, `review`, etc.), and an assigned profile.
| **Execution Model** | The task is claimed by a worker, which runs until it **completes** (`kanban_complete`) or **blocks** (`kanban_block`). The lifecycle is **state‑driven** and persisted across worker restarts.
| **Result Persistence** | All state changes, comments, and attached artifacts are **persisted**. Files referenced in `artifacts` are uploaded for downstream consumption.
| **Failure & Recovery** | Workers can `kanban_block` to request human input. The dispatcher can re‑spawn the worker later. Heartbeats are required for long‑running jobs.
| **Review** | Explicit review step via `kanban_request_review` (or `kanban_complete` if no review needed). Reviewers (e.g., Morgan) receive a structured handoff.
| **Heartbeat** | Required for operations longer than ~1 hour to avoid stale reclamation.

## 3. Key Differences

- **Persistence**: Kanban tasks live in the DB; delegation results live only in transient transcripts.
- **Async Handling**: Delegation spawns sub‑agents that run in parallel and later return a single transcript. Kanban tasks are tracked by status flags and heartbeats – the system knows they are still active.
- **Review Process**: Kanban has a built‑in review column and explicit handoff; delegation relies on the parent to create a review task if needed.
- **Auditing**: Kanban provides an immutable audit trail (events, comments, attachments). Delegation audit must be reconstructed from transcript logs.
- **Scoping**: Delegation is best for **short‑lived, fine‑grained** work (e.g., code generation, data lookup). Kanban is for **coordinated, multi‑step** work that may involve many contributors and requires persistence.

## 4. Similarities

- Both create a **unit of work** with an assigned owner.
- Both support **asynchronous execution** – the worker can continue after spawning.
- Both produce a **result** that downstream components can consume.

## 5. Open Questions & Recommendations

1. **Persisting Delegation Outcomes**: Should we automatically create a Kanban child task for every `delegate_task` result to guarantee auditability? This would bridge the durability gap.
2. **Integration Hook**: Implement a lightweight wrapper `delegate_task_with_kanban` that:
   - Calls `delegate_task`.
   - On successful result, writes a concise summary file.
   - Calls `kanban_create` with the summary as the task body, linking the parent via `parents=[<parent‑task‑id>]`.
3. **Review Alignment**: Standardize that any sub‑agent that performs a *decision‑making* step must result in a Kanban review task so a human can verify the outcome.
4. **Heartbeat Analogy**: Consider adding a “heartbeat” comment to long‑running delegation transcripts to mimic Kanban’s liveness guarantees.

## 6. Conclusion
Async delegation provides a powerful way to parallelize work within a single worker, but it lacks the persistence, audit, and review features baked into the Kanban workflow. By consciously creating Kanban follow‑up tasks for important delegation outcomes, the team can reap the benefits of both systems while maintaining a clear, auditable history.

---
*Prepared by Ada on $(date)*
