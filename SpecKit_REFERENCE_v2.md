# Spec Kit REFERENCE
## Situational Guide — Dip In When a Specific Situation Comes Up

**Maintained by:** Mahantesh, EA GenAI Team — Spec Kit SPOC
**Who this is for:** you've completed **SpecKit_QUICKSTART.md** (or are already comfortable with the basic workflow) and now face a specific situation: an existing codebase, a requirement change, a broken build, a team rollout question.
**How to use this:** don't read it linearly. Find your situation in the routing table and jump there.

## Find Your Situation

| You need to... | Go to |
|---|---|
| Understand what each generated file is for | Part 1 |
| See the standard daily loop | Part 2 |
| Apply Spec Kit to an existing codebase | Part 3 |
| Make a change spanning many files | Part 4 |
| Handle a requirement change (three models) | Part 5 |
| Set up team review workflow | Part 6 |
| Check best practices / common mistakes | Parts 7–8 |
| Fix a specific error | Part 9 |
| Copy a ready-made prompt | Part 10 |
| Grab a template or checklist | Part 11 |
| Plan an org-wide rollout | Part 12 |
| Understand where the industry is heading | Part 13 |
| Get a quick answer | Part 14 (FAQ), Part 15 (Glossary) |
| Decide between options quickly | Part 16 (Decision Trees) |
| Know when a feature is truly "done" | Part 17 |

## Table of Contents

- [Part 1: Understanding the Generated Artifacts](#part-1-understanding-the-generated-artifacts) — *8 min*
- [Part 2: Daily Development Lifecycle](#part-2-daily-development-lifecycle) — *4 min*
- [Part 3: Existing Enterprise Applications](#part-3-existing-enterprise-applications) — *10 min*
- [Part 4: Cross-Cutting and Repository-Wide Changes](#part-4-cross-cutting-and-repository-wide-changes) — *4 min*
- [Part 5: Changing Requirements and Persistence Models](#part-5-changing-requirements-and-persistence-models) — *8 min*
- [Part 6: Working in Teams](#part-6-working-in-teams) — *3 min*
- [Part 7: Best Practices](#part-7-best-practices) — *3 min*
- [Part 8: Common Mistakes](#part-8-common-mistakes) — *3 min*
- [Part 9: Troubleshooting](#part-9-troubleshooting) — *5 min*
- [Part 10: Prompt Cookbook](#part-10-prompt-cookbook) — *4 min*
- [Part 11: Templates and Checklists](#part-11-templates-and-checklists) — *3 min*
- [Part 12: Enterprise Adoption and Governance](#part-12-enterprise-adoption-and-governance) — *10 min*
- [Part 13: The Long-Term Vision](#part-13-the-long-term-vision) — *12 min*
- [Part 14: FAQ](#part-14-faq) — *5 min*
- [Part 15: Glossary](#part-15-glossary) — *3 min*
- [Part 16: Decision Trees](#part-16-decision-trees) — *2 min*
- [Part 17: What Good Looks Like](#part-17-what-good-looks-like) — *2 min*
- [One-Page Quick Reference](#one-page-quick-reference)

---

# Part 1: Understanding the Generated Artifacts
*Reading time: ~8 minutes*

**What you will learn:** what to check *before* moving from one artifact to the next — a review checklist per transition, distinct from QUICKSTART Part 2.7's "what each file is."

## 1.1 What to review before moving forward

**Before `/plan`:** spec is correct · scope bounded · no unresolved ambiguity · requirements measurable and testable.

**Before `/tasks`:** architecture acceptable · interfaces/boundaries clear · security/performance/data concerns addressed · technical decisions justified.

**Before `/implement`:** tasks cover the entire approved scope · setup ordered correctly · testing work included · `/analyze` reports no unresolved critical inconsistency.

**Before merge:** implementation passes tests · quickstart steps actually work · `/converge` findings resolved · artifacts and code describe the *same* system.

```
spec.md ──[gate]──► plan.md ──[gate]──► tasks.md ──[gate]──► code ──[gate]──► merge
```

---

# Part 2: Daily Development Lifecycle
*Reading time: ~4 minutes*

```
Business/product request
        ↓
Understand problem and boundaries
        ↓
Create or select feature branch
        ↓
/specify → review → /clarify
        ↓
/plan → review
        ↓
/tasks → /analyze
        ↓
/implement (incremental) → test
        ↓
/converge
        ↓
Pull request → normal review process
```

This is simply the QUICKSTART Part 4 walkthrough, compressed into the loop you'll repeat for every real feature going forward.

---

# Part 3: Existing Enterprise Applications
*Reading time: ~10 minutes — the most important part if you're not starting from a blank folder*

**What you will learn:** why Spec Kit can't see your existing codebase by default, how to fix that with `ProjectContext.md`, and the full brownfield workflow.

## 3.1 Why existing projects are different

Spec Kit has **no knowledge of your existing codebase.** Running `/specify: Introduce Result<T> across all APIs` on day one of a brownfield project means the AI doesn't know your architecture, controllers, schema, standards, or constraints — it will guess, and guesses conflict with what already exists.

## 3.2 The fix: `ProjectContext.md`

A **recommended team practice, not a built-in Spec Kit feature.** Create `docs/ProjectContext.md` once per repo, before your first `/specify` call.

```markdown
# Project Context
## Overview
## Technology Stack
## Architecture
## Existing Features
## APIs
## Database
## External Integrations
## Coding Standards
## Constraints
## Non-functional Requirements
## Known Technical Debt
```
Keep it to 2–5 pages. Pull content from README, solution structure, controllers, DB schema, existing architecture docs.

> 💡 **Optional shortcut:** if your org has it, the **`project-discovery`** plugin (EA-Marketplace) auto-generates a first-pass, evidence-based discovery report — citing real files instead of guessing. Distill its output into the concise template above before your first `/specify`:
> ```
> Using <discovery-report>.md as the source, generate a concise
> docs/ProjectContext.md following this template. Keep it to 2-5 pages —
> summarize, don't copy sections verbatim.
> ```

## 3.3 The brownfield workflow, full picture

```
Business Requirement
        ↓
Project Discovery              ← read the actual codebase first
        ↓
Create/Update ProjectContext.md
        ↓
/constitution (one-time)       ← base this on REAL existing conventions
        ↓
/specify                       ← now grounded in ProjectContext.md
        ↓
Review (architect/tech lead sign-off)
        ↓
/plan → /tasks → /analyze → /implement → /converge
```

## 3.4 Good vs. bad prompts on existing projects

**Bad:** `/specify: Add caching to the API.` — no context on what already exists.
**Good:**
```
/specify
Use docs/ProjectContext.md as context.
Add response caching to the /api/todos GET endpoint only.
Constraint: must not change the existing response schema.
```

## 3.5 Handling architectural drift

Someone changes the architecture outside the process under deadline pressure — now `constitution.md`/`plan.md` no longer match reality.

**Recovery:**
1. Audit: *"Compare the current codebase structure against plan.md and data-model.md. List every place they disagree."*
2. Update `constitution.md` if the change is now permanent.
3. Regenerate `plan.md`/`data-model.md` — don't hand-edit from memory.
4. Update `tasks.md` honestly — mark divergent tasks "implemented differently."
5. Log *why* the drift happened.

### Common Questions — Part 3

**"Why can't I just run `/specify` cold on our existing system?"** You can — but the AI will confidently guess at architecture it can't see, and those guesses will often actively conflict with your real code. `ProjectContext.md` is the fix.

**"Do I need `ProjectContext.md` for a brand new project too?"** No — QUICKSTART Part 2's `specify init` (new project) doesn't need it. It's specifically for when code already exists before Spec Kit shows up.

---

# Part 4: Cross-Cutting and Repository-Wide Changes
*Reading time: ~4 minutes*

For a change spanning many files (e.g., introducing `Result<T>` across every controller), don't assume the agent infers the full scope alone — be explicit:
```
/plan
Scan all controllers, middleware, frontend API consumers, and tests.
Identify every location that needs to change to adopt the Result<T>
response pattern. Generate repository-wide tasks covering all of them.
```
This gets real repository-wide impact analysis — but only because you asked for it explicitly, not automatically.

---

# Part 5: Changing Requirements and Persistence Models
*Reading time: ~8 minutes*

**What you will learn:** the three official ways Spec Kit lets a spec evolve, and a decision tree for picking the right one every time.

Per Spec Kit's own docs ([evolving-specs.md](https://github.com/github/spec-kit/blob/main/docs/guides/evolving-specs.md)):

## 5.1 Flow-Forward Spec
New requirement → new `specs/00X-.../` folder. Old ones stay untouched, permanent history.
*Analogy: a diary — you don't rewrite yesterday's entry, you write a new one.*

## 5.2 Living Spec
A requirement *changes existing behavior* → edit the existing `spec.md` in place, regenerate `plan.md`/`tasks.md`, re-validate with `/analyze`.
*Analogy: a Wikipedia page — continuously edited to reflect current truth.*

## 5.3 Flow-Back Spec
An *insight discovered mid-work* (during `/plan`, `/tasks`, or `/implement`) should reshape an earlier artifact. Capture it wherever it happened → decide what it actually changes → update whichever artifacts now disagree → run `/analyze` → only continue once the artifact set is trustworthy again.
*Analogy: editing a novel — a Chapter 10 realization sends you back to fix Chapter 3.*

**Official caveat, worth repeating verbatim:** *"Flow-back is flexible, but it requires discipline. Do not leave a lower-level change in tasks.md or code if spec.md still says something different."* — this is the formal name for the drift failure mode in Part 3.5; Flow-Back is the *disciplined, intentional* version of that scenario.

## 5.4 All Three, Side by Side

| | Flow-Forward | Living Spec | Flow-Back |
|---|---|---|---|
| Trigger | Genuinely new capability | A planned requirement change | A mid-work discovery |
| Starting point | New `specs/00X/` folder | Always `spec.md` first | Wherever the insight landed |
| Direction | Forward only | Forward only, re-run per change | Can move **backward** |
| Validation | `/analyze` less critical | `/analyze` before resuming | `/analyze` is essential — highest drift risk |

> ⚠️ **Separate risk:** these three models govern your *feature artifacts*. `specify init --here --force` (refreshing Spec Kit's own project files) is different — it can **overwrite your customized `constitution.md`** unless backed up first.

### Common Questions — Part 5

**"Why can't I just always use Flow-Forward — isn't it the simplest?"** You can, but old specs pile up disconnected from each other, and a feature that's genuinely still evolving ends up scattered across several folders instead of one coherent document.

**"Isn't Flow-Back just... fixing bugs?"** Not quite — Flow-Back specifically means the fix reshapes an *earlier artifact* (the spec or plan), not just the code. If you only touch code and never reconcile the spec, that's uncontrolled drift, not disciplined Flow-Back.

---

# Part 6: Working in Teams
*Reading time: ~3 minutes*

**Solo developer:** self-review the spec before `/plan` — read it as a stranger would.
**2–5 developers:** one prepares the spec, one teammate reviews before `/plan` runs.
**Larger/cross-team:** add an architecture review step before `/plan` for shared infrastructure; consider a shared `constitution.md` preset (Part 12.4) so every team enforces the same standards.

---

# Part 7: Best Practices
*Reading time: ~3 minutes*

✅ One feature per specification · reference files with `#filename`, don't paste content · review every generated spec/plan before proceeding · run `/clarify` and `/analyze` — cheap, catches problems early · create `ProjectContext.md` once, maintain after major changes · implement layer by layer · run `/converge` — "it compiled" ≠ "it matches the spec" · periodically test constitution compliance directly on a small feature.

---

# Part 8: Common Mistakes
*Reading time: ~3 minutes*

❌ Running `/specify` on an existing project without `ProjectContext.md` · jumping to `/implement` before setup tasks are done · combining unrelated features into one spec · pasting whole files into chat instead of referencing them · treating AI output as production-ready without review · letting `plan.md`/`tasks.md` go stale after out-of-process changes · assuming a "MUST" constitution rule guarantees compliant code.

---

# Part 9: Troubleshooting
*Reading time: ~5 minutes*

**"Could not execute... dotnet-ef... command not found"**
```powershell
dotnet tool install --global dotnet-ef
```
If already installed but not found: `dotnet tool update --global dotnet-ef`, then reopen the terminal.

**"No migrations were applied. Database already up to date." — but you never made one**
Misleading message — means zero migrations *exist*, not that schema is set up:
```powershell
dotnet ef migrations add InitialCreate
dotnet ef database update
```

**After `/implement`, no `.csproj`/`package.json` anywhere**
Setup tasks (T001/T002) were skipped. Ask explicitly: *"Execute only the setup tasks from tasks.md — create the backend and frontend project structure first."*

**Frontend runs but data doesn't persist**
Likely still using local in-memory state. Ask Copilot to wire it to the real API with proper CORS configuration, then restart both servers.

**Generated code doesn't match a constitution rule**
Not a bug — constitution rules aren't compiler-enforced (see the QUICKSTART Part 4.3 "why" box). Use `/converge` to check compliance directly, then request a targeted fix.

**Direct DB calls inside a controller, or non-REST endpoints**
Same root cause as above — tighten the specific constitution wording (e.g., "no exceptions, including simple CRUD") and regenerate just that layer.

---

# Part 10: Prompt Cookbook
*Reading time: ~4 minutes*

**New spec:** `/specify [plain-language feature description, no tech detail]`

**Extend existing feature (brownfield):**
```
/specify
Use docs/ProjectContext.md as context.
Extend the existing [feature] to support [capability].
Constraint: [backward-compatibility requirement].
```

**Cross-cutting change with impact analysis:**
```
/specify
Use docs/ProjectContext.md as context.
Introduce [pattern] across [scope]. Generate: functional requirements,
migration strategy, impact analysis, acceptance criteria.
```

**Layer-by-layer implementation:**
```
/implement
Generate only: [specific file/layer]. Reference #tasks.md. Nothing else yet.
```

**Drift audit:** `Compare the current codebase against plan.md and data-model.md. List every disagreement.`

**Constitution compliance check:**
```
/converge
Verify [file] against constitution.md's [specific principle].
Report every disagreement.
```

---

# Part 11: Templates and Checklists
*Reading time: ~3 minutes*

## `docs/ProjectContext.md`
```markdown
# Project Context
## Overview
## Technology Stack
## Architecture
## Existing Features
## APIs
## Database
## External Integrations
## Coding Standards
## Constraints
## Non-functional Requirements
## Known Technical Debt
```

## Specification Review Checklist
```markdown
- [ ] No implementation details leaked into the spec
- [ ] Written for a non-technical stakeholder to understand
- [ ] All requirements are testable and unambiguous
- [ ] Success criteria are measurable
- [ ] Edge cases identified
- [ ] Scope clearly bounded
- [ ] No [NEEDS CLARIFICATION] markers remain
```

## Pre-Merge Checklist
```markdown
- [ ] Automated tests pass
- [ ] Manual acceptance checks pass
- [ ] Database migration validated
- [ ] API compatibility validated
- [ ] Security-sensitive changes reviewed
- [ ] /converge findings resolved
- [ ] Artifacts match the final implementation
- [ ] Rollback/recovery approach understood
```

---

# Part 12: Enterprise Adoption and Governance
*Reading time: ~10 minutes*

## 12.1 Start with a pilot

Use a bounded but realistic feature: more than one layer/component, meaningful acceptance criteria, a real DB/API change, automated tests, at least one requirement change *during* the pilot. **Don't judge enterprise fit from a blank Todo project alone** — the Todo tutorial validates mechanics; a brownfield pilot validates real organizational fit.

## 12.2 Pilot success measures

Clarification issues found before coding · rework avoided/introduced · spec/plan review time · implementation cycle time · defect/regression rate · % of acceptance criteria with test evidence · artifact drift found by `/analyze`/`/converge` · stakeholder usability feedback · AI token/credit consumption.

## 12.3 Organization-level standards worth considering

Constitution baseline · required spec sections · security/compliance sections · ADR format · task traceability expectations · PR evidence requirements · artifact ownership · retention/versioning approach · permitted tools and data-handling rules.

## 12.4 Extensions and Presets

**Extensions** add entirely new commands/capabilities (Jira sync, security review command, project discovery, test traceability). `specify extension search` / `specify extension add <name>`.

**Presets** override existing templates to enforce org standards (mandatory NFR sections, compliance-oriented acceptance criteria, standard task ordering). `specify preset search` / `specify preset add <name>`.

| | Extensions | Presets |
|---|---|---|
| Adds new commands? | Yes | No |
| Changes existing templates? | No | Yes, overrides in place |
| Analogy | Installing a new app | Changing settings on an existing app |

Validate available commands against your installed Spec Kit version before org-wide rollout.

## 12.5 Multi-repository pattern

```
Shared business outcome
        ↓
Cross-repository architecture and dependency analysis
        ↓
Repository-specific specifications or implementation plans
        ↓
Per-repository tasks and pull requests
        ↓
Shared contract and end-to-end validation
```
The business outcome stays stable even if the technical decomposition changes later — separating "what we promised" from "how we're currently building it."

## 12.6 Preventing specification sprawl

Keep historical Flow-Forward specs as an untouched audit trail · maintain a separate, periodically-regenerated "current system overview" (not hand-maintained like a BMAD-style PRD) · link capabilities to relevant specs/code · never manually merge dozens of old specs into one unreviewable document.

## 12.7 Classify failures correctly

**Intent-to-spec gap** — the requirement conversation or spec missed something. Fix: discovery, clarification, spec templates, stakeholder review, constitution wording (this is exactly what the QUICKSTART Part 4.3 REST/DbContext example was).

**Spec-to-implementation gap** — the spec was right, code diverged anyway. Fix: task precision, smaller increments, tests, `/converge` validation, PR review.

## 12.8 Recommended rollout stages

```
Stage 1: Learn        → complete the Todo tutorial + orientation
Stage 2: Pilot         → one bounded brownfield feature
Stage 3: Standardize   → shared templates, constitution baseline, checklists
Stage 4: Integrate     → connect to backlog, CI, testing, PR governance
Stage 5: Scale         → cross-team adoption, metrics, ongoing template improvement
```

---

# Part 13: The Long-Term Vision
*Reading time: ~12 minutes*

**What you will learn:** where the industry believes this category is heading — useful vocabulary for strategic conversations, clearly separated from what Spec Kit actually does today.

## 13.1 "SpecFall" — the core organizational risk

Named after "Scrumerfall" (adopting Agile ceremonies without changing real collaboration). **SpecFall** is the same failure applied to SDD: running `/specify → /plan → /tasks` mechanically, without changing how product/architecture/engineering/QA actually work together, produces a **"markdown monster"** — technically valid files nobody actually uses as a real collaboration surface.

A solo pilot (like the Todo app) proves the *mechanics* work. It doesn't yet prove the *collaboration* value — which is the bigger prize at enterprise scale.

## 13.2 Seven real tooling gaps at enterprise scale

1. **Developer-centric tooling** — Git/CLI-based, real friction for PMs/BAs who should own the "what."
2. **Mono-repo assumption** — most tools, Spec Kit included, don't have a clean answer for specs spanning multiple repos.
3. **No separation by audience/lifecycle** — strategic architecture decisions and tactical task lists share one folder.
4. **Unclear starting point** — most orgs already have refined Jira/Azure DevOps backlogs; integration isn't built in.
5. **Undefined collaboration patterns** — no built-in clarity on who approves what, when.
6. **No standard spec style across tools** — Spec Kit's format, Amazon Kiro's EARS format, etc. all diverge.
7. **Unclear brownfield path** — spec the whole legacy system, or build incrementally? **Incremental is right** — comprehensively specing a large legacy system risks exceeding context limits, and even if it succeeds, the result is too large for a human to meaningfully review, defeating the purpose. This directly validates Part 3's `ProjectContext.md`-first, incremental approach.

## 13.3 Multi-repo orchestration, the fuller version

```
Product Owner  → articulates business "what" (stays stable long-term)
        ↓
Architect      → breaks it into repo-specific sub-issues, scoped boundaries
        ↓
Engineer (per repo) → implementation tasks scoped to just that repo
```
Architects don't manually decompose every story by hand — document repo boundaries and constraints *once*, as a reusable harness, and let the agent apply it to new incoming stories.

## 13.4 Role-specific "harnesses" beyond the architect's constitution

Infrastructure specialists → deployment constraints · Performance specialists → optimization requirements · Security specialists → compliance requirements — each becomes an additional automatically-applied "constitution," catching concerns *before* implementation instead of in a review gate afterward.

## 13.5 The provocative claim: should even tiny fixes go through the spec?

The stronger argument: yes, even small bug fixes — because a direct code patch can be **silently overwritten the next time that code is regenerated by AI**, since the spec never captured the fix and AI generation is non-deterministic. Same logic as why teams stopped allowing direct production server changes: not a rule for its own sake, but because direct changes get silently overwritten by the next deployment.

**[VALIDATE ON PILOT]** — stronger discipline than anything in Parts 1–16. Worth testing deliberately: does skipping the spec for a "trivial" fix actually cause recurrence, or does this matter more at higher AI-autonomy scale than one pilot will reveal?

## 13.6 Harness governance — classify where bugs actually come from

Same as Part 12.7, applied at the philosophical level: a **spec-to-implementation gap** means the process worked but execution didn't — fix validation. An **intent-to-spec gap** means the process itself has a blind spot — fix the harness/constitution/template, or the same bug class recurs across every future feature.

## 13.7 The more radical vision: "architecture becomes executable"

A second, more theoretical school of thought (Griffin & Carroll, InfoQ) proposes something more extreme: what if the **spec**, not the code, became the actual protected source of truth — with code treated as fully disposable, regenerated on demand, the way a `bin/` folder is disposable today?

**Five-layer model:** Specification (declares what must be true, zero implementation detail) → Generation (a "multi-target compiler" turning spec into code/tests/docs) → Artifact (the generated code itself — disposable) → Validation (continuously checks running behavior against the spec) → Runtime (the live system, whose behavior is guaranteed by Layer 1, not decided independently here).

**"Architectural inversion":** today, if code and docs disagree, code wins — docs get updated to match. This vision inverts it: if code and spec disagree, the **spec** wins — code gets regenerated to match it.

**Humans are "relocated, not removed":** the AI owns *mechanical* enforcement (does the code match the spec). Humans retain authority over *meaning* — approving breaking changes, authorizing policy shifts, judging whether a trade-off is acceptable. Machines own "did we do what we said." Humans own "do we actually want to do this."

**Honest reality check — this is a future destination, not today's road:**

| Vision | Spec Kit's actual reality today |
|---|---|
| Regenerating from the same spec always gives identical code | Not true — AI generation varies (exactly why the QUICKSTART Part 4.3 constitution-gap example happened) |
| Continuous, automatic drift detection | Manual — you run `/analyze`/`/converge` yourself |
| Code fully disposable, zero information lost if deleted | Not true yet — real decisions still get made *during* `/implement` that aren't fully captured back in the spec |
| Spec always wins when code and spec disagree | Currently the reverse is usually easier — audit code, update the spec to match (Part 3.5) |

**Why this section is worth reading anyway:** it gives you sharp vocabulary — *SpecOps, architectural inversion, bounded autonomy* — for sounding sophisticated in strategic conversations about where this category is heading, without mistaking it for what you should actually do on tomorrow's pilot.

---

# Part 14: FAQ
*Reading time: ~5 minutes*

**Should every change use the full workflow?** No — scale to risk/complexity/impact. Any change that alters accepted behavior should still leave a durable record of the corrected intent and its tests.

**Is `/constitution` run for every feature?** No — once per project, updated only when a project-wide principle genuinely changes.

**Can generated Markdown be edited manually?** Yes, expected. After editing an upstream artifact, reconcile the downstream ones.

**Does Spec Kit automatically know about my other feature specs?** Not reliably. The agent *can* inspect the repo when explicitly instructed, but commands operate primarily on the *active* feature — no automatic cumulative reconciliation across every historical spec. Use `ProjectContext.md` and explicit impact prompts (Part 4).

**Can 1–2 developers use this without multiple "AI roles"?** Yes — the essential controls are artifact quality, review, and validation, not the number of personas involved.

**Difference between `/analyze` and `/converge`?** `/analyze` checks spec/plan/tasks agree *before* code exists (pre-flight). `/converge` checks the *real code* against all three *after* implementation (landing check).

**Does a constitution guarantee compliant code?** No — use precise wording, tests, analyzers, CI, and `/converge`, not trust alone.

**How are rejected changes rolled back?** Git branches, PRs, commits, reverts, feature flags, DB recovery practices — Spec Kit doesn't replace source-control rollback.

**How should a new requirement be handled after implementation?** Decide: new capability (Flow-Forward), change to existing behavior (Living Spec), or mid-work discovery (Flow-Back) — see Part 5.

**How do I control Copilot credit usage?** Focused prompts · reference artifacts with `#filename` instead of pasting · implement in small increments · new chat sessions for major context switches · review before regenerating · never ask for the whole app repeatedly.

---

# Part 15: Glossary
*Reading time: ~3 minutes*

**SDD (Specification-Driven Development)** — an approach where structured artifacts (spec, plan, tasks) are the durable source of intent, actively guiding implementation, not just documentation written after the fact.

**Constitution** — the project-wide, non-negotiable engineering rules file, read by every downstream command.

**Spec (`spec.md`)** — the feature's requirements in plain language, with zero implementation detail.

**Plan (`plan.md`)** — the technical design: architecture, stack, layering, derived from the spec.

**Drift** — when code diverges from what the spec/plan/constitution describes, usually because a change happened outside the Spec Kit process.

**Flow-Forward** — a persistence model where every new requirement gets its own new spec folder; old ones stay untouched.

**Living Spec** — a persistence model where an existing `spec.md` is edited in place when planned requirements change.

**Flow-Back** — a persistence model where a mid-implementation discovery flows backward into earlier artifacts (spec/plan), with discipline to reconcile everything afterward.

**`/converge`** — the command that checks finished, real code against spec/plan/tasks — the "landing check."

**`/analyze`** — the command that checks spec/plan/tasks agree with each other, before code is written — the "pre-flight check."

**`ProjectContext.md`** — a recommended (not built-in) file summarizing an existing codebase's stack, architecture, and constraints, so brownfield specs are grounded in reality.

**Brownfield** — an existing codebase you're applying Spec Kit to, as opposed to a brand-new project.

**Extension** — adds new Spec Kit commands/capabilities.

**Preset** — overrides existing Spec Kit templates to enforce org-specific standards.

**SpecFall** — the risk of adopting SDD's commands mechanically without changing real cross-role collaboration.

**Intent-to-spec gap** — a bug caused because the spec itself missed something during requirements gathering.

**Spec-to-implementation gap** — a bug caused because the spec was correct but the generated code diverged from it anyway.

---

# Part 16: Decision Trees
*Reading time: ~2 minutes*

## Which persistence model should I use? (Part 5)

```
Is this a completely new capability?
        │
        ├── YES ──► Flow-Forward (new specs/00X/ folder)
        │
        └── NO ──► Am I changing existing, already-specified
                    behavior, and did I know this before starting?
                        │
                        ├── YES ──► Living Spec (edit spec.md in place)
                        │
                        └── NO, I only realized it WHILE working
                                ──► Flow-Back (capture where discovered,
                                     reconcile backward, then /analyze)
```

## Do I need `ProjectContext.md`?

```
Does code already exist in this repo, written before Spec Kit?
        │
        ├── YES ──► Create/update ProjectContext.md BEFORE /specify (Part 3)
        │
        └── NO, brand new project ──► Skip it, go straight to /constitution
```

## Should I run the full workflow, or just make the change directly?

```
Is this a one-line typo or purely cosmetic change?
        │
        ├── YES ──► Just fix it directly, no need for the full pipeline
        │
        └── NO ──► Does it touch shared architecture, span multiple
                    files, or matter for future traceability?
                        │
                        ├── YES ──► Full workflow (QUICKSTART Parts 3-4)
                        │
                        └── NO, small and isolated ──► Lighter touch is fine,
                             but still update spec.md if behavior changed
                             (Part 13.5 — even small fixes ideally leave
                              a durable record)
```

---

# Part 17: What Good Looks Like
*Reading time: ~2 minutes*

A healthy Spec Kit feature has: a clearly stated business problem · a bounded, testable spec · resolved ambiguity · a reviewed plan with real rationale · complete, ordered tasks · small, reviewable implementation increments · automated and manual test evidence · implementation consistent with the constitution · aligned artifacts and code · a normal PR/release process.

**The real test:** a new team member should be able to open the repo, read `ProjectContext.md` and the active feature's artifacts, run `quickstart.md`, understand *why* the feature was built that way, and continue the work — without ever needing the original chat conversation.

---

# One-Page Quick Reference

```text
NEW PROJECT
specify init → branch → /constitution → /specify → /clarify
→ review → /plan → review → /tasks → /analyze
→ /implement incrementally → test → /converge → PR

EXISTING PROJECT
specify init --here → repository discovery → ProjectContext.md
→ constitution based on reality → bounded /specify
→ explicit impact-aware /plan → /tasks → /analyze
→ incremental /implement → test → /converge → PR

REQUIREMENT CHANGE
classify: Flow-Forward / Living Spec / Flow-Back
→ update the correct upstream artifact
→ reconcile downstream artifacts
→ /analyze → implement → /converge

NEVER SKIP
human review, Git controls, tests, artifact reconciliation,
security/architecture review for high-risk changes
```

---

*This is a living handbook. Improve it using evidence from real pilots, recurring questions, missed requirements, architecture violations, and production feedback. Maintained by the EA Spec Kit SPOC.*
