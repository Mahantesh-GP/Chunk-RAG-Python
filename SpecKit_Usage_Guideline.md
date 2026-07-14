# Spec-Driven Development (Spec Kit) — Usage Guideline
**Prepared by:** Mahantesh, EA GenAI Team
**Audience:** Engineers piloting Spec Kit on a real project
**Status:** Draft — ready to apply once a pilot project is nominated

---

## 1. Purpose

This guideline exists so that anyone on the team — not just the person who explored the tool — can pick up Spec Kit and apply it consistently on a real project. It covers the standard workflow, when to deviate from it, and how to recover when reality drifts away from the documented spec (a scenario Bijesh specifically flagged).

---

## 2. When to Use This

- **New (greenfield) features or projects** — the primary, well-supported case.
- **Brownfield features** — applying Spec Kit to an existing codebase (as done with the GHCP Rollout Dashboard). Works, but requires an extra step: capturing the *current* state as a baseline before specifying new work (see Section 5).

---

## 3. Standard Workflow

Run once per project:
```
/constitution   → defines non-negotiable rules (architecture style, testing policy, layering)
```

Run once per feature:
```
/specify   → spec.md              (what the feature does, no implementation detail)
/plan      → plan.md, research.md, data-model.md   (how it will be built, with rationale)
/tasks     → tasks.md             (ordered, numbered implementation steps)
/implement → executes tasks.md, writes actual code
             quickstart.md used to validate the result against spec.md
```

**Rule:** Always confirm the first 1–2 tasks in `tasks.md` are project/module *setup* tasks before jumping to feature-level `/implement` calls. Skipping these produces code fragments with no project shell to run them in — a mistake worth calling out explicitly in training, since it's easy to make.

---

## 4. Credit / Token Efficiency (org-specific constraint)

Since Copilot Business bills chat interactions as credits with a monthly cap:
- `specify`, `plan`, and `tasks` run locally via CLI — no credits consumed.
- Only `/implement` calls consume Copilot credits — do these layer by layer, not as one giant prompt.
- Reference files with `#filename` instead of pasting content.
- Start a new chat session when switching context (e.g., backend → frontend) to avoid re-sending stale history as tokens.

---

## 5. Handling Architectural Drift (Bijesh's scenario)

**The scenario:** A team starts a feature using Spec Kit, then — mid-project — makes a significant architectural change *outside* the Spec Kit process (e.g., under deadline pressure, or during an incident fix). Later, they want to resume using Spec Kit for the next feature. The problem: `constitution.md`, `plan.md`, and `tasks.md` no longer reflect what was actually built.

**Why this matters:** If left unreconciled, every subsequent `/plan` and `/tasks` call will generate output based on a stale, incorrect picture of the codebase — producing plans that don't match reality and tasks that assume structures that don't exist. This is exactly the failure I hit personally on the Todo app (Iteration 2), when `plan.md` still described Clean Architecture layers and a reducer that were never actually built.

**Recovery procedure:**

1. **Audit before resuming.** Before running any new Spec Kit command, do a manual (or Copilot-assisted) diff between what the docs claim and what the code actually contains. Ask directly:
   ```
   Compare the current codebase structure against plan.md and data-model.md.
   List every place they disagree.
   ```
2. **Reconcile `constitution.md` first.** If the architectural change was intentional and permanent (e.g., dropped Clean Architecture layering for a simpler structure), update the constitution to reflect the *new* standard going forward — don't leave it describing a style you've abandoned.
3. **Regenerate `plan.md` and `data-model.md`, don't hand-edit them from memory.** Ask Spec Kit to re-derive the plan against the current codebase, referencing the updated constitution:
   ```
   /plan
   Re-derive the technical plan from the current codebase structure and the 
   updated constitution.md. Flag any spec.md requirements that no longer 
   have a clear implementation path.
   ```
4. **Reconcile `tasks.md` status honestly.** Don't blanket-mark tasks complete. For each task, verify the described file/approach actually exists as described — if the implementation diverged (different file names, different pattern), mark it separately as "implemented differently" rather than blindly checked off, and update the task text to match reality.
5. **Log the drift.** Keep a short note (in the spec folder or a project changelog) of *why* the architectural change happened outside the process. This isn't bureaucracy — it's what lets the next person (or the next AI-assisted planning pass) understand *why* the docs and code diverged, instead of assuming someone made a mistake.

**General principle to teach pilot teams:** Spec Kit's artifacts (constitution, plan, tasks) are only as trustworthy as their last reconciliation with the real codebase. Treat "docs vs. code drift" the same way you'd treat any other technical debt — visible, tracked, and paid down deliberately, not ignored.

---

## 6. Brownfield-Specific Guidance

When applying Spec Kit to an existing project (not starting fresh):
1. Run `/constitution` based on the project's *actual* existing conventions, not aspirational ones — otherwise every subsequent plan will conflict with the real code from day one.
2. Before `/specify` on the first feature, ask Spec Kit to generate a baseline `research.md`-style summary of the current architecture, so the "alternatives considered" reasoning has an accurate starting point.
3. Expect the first 1–2 features to surface constitution mismatches — this is normal and is the calibration period, not a failure of the tool.

---

## 7. What "Good" Looks Like for a Pilot

By the end of a pilot, the team should be able to show:
- A `constitution.md` that accurately reflects real project conventions
- At least one feature taken end-to-end through specify → plan → tasks → implement → quickstart validation
- Evidence that `tasks.md` status reflects actual code state, not optimistic assumptions
- At least one documented instance of drift detection/reconciliation, if applicable

---

## 8. Open Questions to Track During Pilot

- How does Spec Kit behave when two engineers work on the same spec folder concurrently?
- What's the practical token/credit cost per feature at real project scale (not a Todo app demo)?
- Does the drift-reconciliation process above hold up, or does it need refinement after a real pilot?

---

*This document is a living guideline — update it after the first real pilot with Mani's team, incorporating what actually happened versus what was anticipated here.*
