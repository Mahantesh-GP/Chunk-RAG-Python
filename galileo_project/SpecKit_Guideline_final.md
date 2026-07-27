# Specification-Driven Development with GitHub Spec Kit
## The Complete Practical Handbook — From First Install to Enterprise Adoption

**Maintained by:** Mahantesh, EA GenAI Team — Spec Kit SPOC
**Audience:** New Joiners, Developers, Tech Leads, Architects, Business Analysts, QA Engineers
**What this is:** Not a command reference. A guide for *using* Spec Kit successfully — what to review at every stage, how to complete a feature end to end, how to use it safely on an existing enterprise application, and where the industry is heading next.

---

## How to Use This Guide

| You are... | Start here |
|---|---|
| Completely new to SDD and Spec Kit | Part 1 → Part 4 |
| Starting a new project | Part 2 → Part 6 |
| Adding a feature to an existing project | Part 1, Part 2, Part 7 |
| Handling a requirement change | Part 9 |
| Working in a team | Part 10 |
| Something broke | Part 13 |
| Rolling out Spec Kit across an organization | Part 16, Part 17 |
| Just need a quick answer | Part 18 (FAQ), Part 19 (Glossary), or the Quick Reference at the very end |

## What You Will Learn (Parts 1–4)

By the end of Part 4, you should be able to: explain SDD and Spec Kit in your own words → initialize a project → write and review a constitution → generate a feature spec → clarify ambiguity → create a plan and task list → implement in controlled, reviewable increments → verify the final code against the original intent.

---

## Table of Contents
*(click any line to jump there)*

- [Part 1: Foundations](#part-1-foundations) — *8 min*
- [Part 2: Getting Started](#part-2-getting-started) — *10 min*
- [Part 3: Ten-Minute Command Orientation](#part-3-ten-minute-command-orientation) — *4 min*
- [Part 4: Complete Todo Application Walkthrough](#part-4-complete-todo-application-walkthrough) — *20 min*
- [Part 5: Understanding the Generated Artifacts](#part-5-understanding-the-generated-artifacts) — *8 min*
- [Part 6: Daily Development Lifecycle](#part-6-daily-development-lifecycle) — *4 min*
- [Part 7: Existing Enterprise Applications](#part-7-existing-enterprise-applications) — *10 min*
- [Part 8: Cross-Cutting and Repository-Wide Changes](#part-8-cross-cutting-and-repository-wide-changes) — *4 min*
- [Part 9: Changing Requirements — Three Persistence Models](#part-9-changing-requirements--three-persistence-models) — *8 min*
- [Part 10: Working in Teams](#part-10-working-in-teams) — *3 min*
- [Part 11: Best Practices](#part-11-best-practices) — *3 min*
- [Part 12: Common Mistakes](#part-12-common-mistakes) — *3 min*
- [Part 13: Troubleshooting](#part-13-troubleshooting) — *5 min*
- [Part 14: Prompt Cookbook](#part-14-prompt-cookbook) — *4 min*
- [Part 15: Templates and Checklists](#part-15-templates-and-checklists) — *3 min*
- [Part 16: Enterprise Adoption and Governance](#part-16-enterprise-adoption-and-governance) — *10 min*
- [Part 17: The Long-Term Vision (Strategic, Not Tactical)](#part-17-the-long-term-vision-strategic-not-tactical) — *12 min*
- [Part 18: FAQ](#part-18-faq) — *5 min*
- [Part 19: Glossary](#part-19-glossary) — *3 min*
- [Part 20: Decision Trees](#part-20-decision-trees) — *2 min*
- [Part 21: What Good Looks Like](#part-21-what-good-looks-like) — *2 min*
- [One-Page Quick Reference](#one-page-quick-reference)

---

# Part 1: Foundations
*Reading time: ~8 minutes*

**What you will learn:** why prompt-driven AI coding breaks down on real projects, what SDD actually is, what Spec Kit is, when to use it (and when not to), and its honest benefits and limitations.

## 1.1 The problem with prompt-driven development

A prompt like:
```text
Build a Todo application using .NET and React.
```
generates code quickly, but leaves real questions unanswered: what exact problem are we solving, which requirements are mandatory, which architectural rules must be followed, why a decision was made, what happens when the requirement changes, how another developer continues the work in a new chat, and how a reviewer verifies the code matches original intent.

In ordinary AI-assisted coding, this knowledge lives in chat history — which is hard to review, version, approve, or maintain.

```
┌─────────────────────────────────────────────────────┐
│  "Build a Todo app" ──► code appears                 │
│                                                        │
│  Six weeks later: "why did we build it this way?"    │
│  Answer: buried in a chat log nobody re-reads.        │
└─────────────────────────────────────────────────────┘
```

> **Why does this matter enough to justify a whole new process?**
> Because the AI has no memory between sessions. Every new chat starts from zero unless intent is written down somewhere durable. SDD's entire premise is: write intent into files, not conversations.

## 1.2 What is Specification-Driven Development?

SDD treats structured engineering artifacts as the durable source of intent — not documentation written *after* coding, but something that actively guides planning, implementation, and validation.

```
Business problem
      ↓
Specification  ── what the system must do
      ↓
Technical plan ── how it will be built
      ↓
Tasks          ── ordered implementation work
      ↓
Implementation ── source code and tests
      ↓
Validation     ── proof that implementation matches intent
```

## 1.3 What is GitHub Spec Kit?

An open-source toolkit for applying SDD with AI coding assistants like GitHub Copilot. It provides reusable commands and templates for producing consistent specification artifacts.

```
/constitution → /specify → /clarify → /plan → /tasks → /analyze → /implement → /converge
```

## 1.4 When Spec Kit is useful

Especially valuable when: requirements have multiple acceptance criteria, implementation spans several layers, architectural rules must stay consistent, multiple developers/teams are involved, requirements are likely to evolve, auditability matters, or AI will do a significant share of implementation.

**For a one-line typo or isolated formatting change, the full workflow is overkill.** Scale the process to the risk and impact of the change — use judgment, not ritual.

## 1.5 Benefits

Durable requirements and decision history · clear separation between intent and implementation · consistent artifacts across developers · better onboarding and handover · earlier discovery of ambiguity · traceability from requirement to task to code · controlled AI use instead of unconstrained generation.

## 1.6 Limitations — be honest about these

- Doesn't automatically understand an existing enterprise codebase (Part 7 fixes this).
- Generated specs and plans still need human review.
- Constitution rules guide the model — they are **not compiler-enforced**.
- Cross-feature or repo-wide impact must often be explicitly requested (Part 8).
- Artifacts can go stale if code changes outside the process (Part 9 covers recovery).
- Git and pull-request controls are still required — Spec Kit doesn't replace them.

### Common Questions — Part 1

**"Isn't this just more process for the sake of process?"**
Not if scaled to risk — Section 1.4 explicitly says skip the full workflow for trivial changes. It's meant for anything with real complexity or team coordination cost.

**"Doesn't writing all this take longer than just coding?"**
Upfront, yes. The bet is that catching a wrong assumption in a 2-minute spec review is cheaper than catching it after a day of implementation — see the `/clarify` "why" box in Part 4.

---

# Part 2: Getting Started
*Reading time: ~10 minutes*

**What you will learn:** the two separate prerequisite lists people conflate, how to install Spec Kit, how to start a new vs. existing project, and safe Git setup before generating any code.

## 2.1 Prerequisites — Two Different Things, Don't Confuse Them

> **Why split this into two lists?**
> Because Spec Kit itself is stack-agnostic. Confusing "what Spec Kit needs" with "what my .NET/React example needs" is the single most common source of newcomer confusion.

**1. What Spec Kit itself needs, regardless of your project's stack:**
- Python 3.11+ and `uv` (only to install/run the `specify` CLI — you're not writing Python)
- Git
- An AI coding agent — this handbook assumes GitHub Copilot Business
- VS Code (for the Copilot Chat slash-command integration)

```bash
pip install uv
uv tool install specify-cli --from git+https://github.com/github/spec-kit.git
specify check
```
**Expected output:** `specify check` reports which required tools are found/missing on your machine — fix anything flagged before continuing.

**2. What you need to *run* the example project used throughout this handbook (.NET + React + SQL Server Todo app):**
- .NET 8 SDK
- Node.js 20+ and npm
- SQL Server or SQL Server LocalDB
- GitHub Copilot Business or equivalent org access

**If your real project uses a different stack**, swap list 2 for whatever that stack needs — list 1 never changes.

## 2.2 Initialize a new project

```bash
specify init todo-sdd-demo --integration copilot
cd todo-sdd-demo
code .
```
**Expected output:** two new folders appear — `.github/prompts/` (your slash commands, auto-discovered by Copilot Chat) and `.specify/` (templates + scripts). No Copilot credits used yet — this is local scaffolding only.

## 2.3 Initialize an existing repository

```bash
specify init --here --integration copilot
```
**Do not immediately run `/specify` on a large existing system** — complete the brownfield preparation in Part 7 first.

## 2.4 Verify Copilot commands

Open Copilot Chat in VS Code and confirm Spec Kit's slash commands appear. Command naming can vary slightly by version — the files under `.github/prompts/` are always the source of truth for what's actually installed in *your* project.

## 2.5 Safe Git setup — before generating anything

```bash
git init
git add .
git commit -m "Initialize Spec Kit project"
git checkout -b feature/todo-management
```
**Never perform experimental AI-generated changes directly on the protected main branch.**

## 2.6 The Artifact Dependency Map — What Actually Reads What

```
constitution.md ───────────────────────────────────────┐
  (read by every downstream command — the one artifact  │
   that isn't "owned" by a single stage)                │
                                                          ▼
spec.md ──► requirements.md (quality checklist on spec.md itself)
   │
   ▼
/plan ──► research.md (decision rationale)
      └─► data-model.md (entity design)
      └─► plan.md (technical design; references the two above)
      └─► quickstart.md (validation runbook — used AFTER /implement)
   │
   ▼
/tasks ──► tasks.md
   (derived from plan.md — does NOT re-read spec.md directly;
    it trusts plan.md as the source of truth)
   │
   ▼
/implement ──► actual code
   │
   ▼
/analyze  — checks spec.md ↔ plan.md ↔ tasks.md agree (pre-flight, before/after tasks)
/converge — checks the REAL CODE against spec/plan/tasks (landing check, after implement)
   │
   ▼
quickstart.md — the one artifact that deliberately reaches back to spec.md's
                original acceptance criteria, closing the loop
```

**The one rule that matters most:** `constitution.md` is read by everything downstream. Everything else is a strict pipeline — spec → plan → tasks → implement — where each stage trusts the *previous* stage's output rather than re-deriving from the original spec every time. This is exactly why drift is dangerous (Part 9): if `plan.md` goes stale, `tasks.md` inherits that staleness, and `/implement` never independently double-checks against the original spec.

## 2.7 Every Artifact File — What It Is, Why It Exists, What's In It, What It Must Not Become

| File | Created by | Why it exists | Contains | Must not become |
|---|---|---|---|---|
| `constitution.md` | `/constitution` | Prevents every feature from using a different architecture style | Non-negotiable, project-wide rules | A list of one feature's requirements |
| `spec.md` | `/specify` | Captures *what*, free of tech detail, readable by non-engineers | Numbered functional requirements, scenarios, edge cases, acceptance criteria | A technical design document |
| `requirements.md` (checklist) | `/specify` | Quality-gates `spec.md` itself before planning is allowed — GitHub calls this "unit tests for English" | Checklist: no leaked implementation detail, no unresolved `[NEEDS CLARIFICATION]`, testable requirements | A code test report |
| `research.md` | `/plan` | Your architecture decision record — auditable *why*, not just *what* | Decision + rationale + alternatives rejected | A duplicate of the whole plan |
| `data-model.md` | `/plan` | Source of truth for entity design | Entities, fields, types, relationships | An uncontrolled replacement for real migrations |
| `plan.md` | `/plan` | The "how" that complements spec.md's "what" | Architecture, stack, layering; references research.md + data-model.md | A restatement of every business requirement |
| `tasks.md` | `/tasks` | Ordered, numbered steps `/implement` actually executes | Numbered tasks (T001...), setup tasks first | A vague, unordered backlog |
| `quickstart.md` | `/plan` (used after `/implement`) | Validates the *finished* feature against spec.md's original acceptance criteria | Step-by-step validation instructions | Generic setup text that no longer matches the code |

### Common Questions — Part 2

**"Why does `tasks.md` not just re-read `spec.md` directly?"**
Because the pipeline trusts each stage's output. This is efficient, but it's also *why* drift is dangerous — see Part 9.

**"Do I need Python forever, or just once?"**
Just to install/run the `specify` CLI tool itself. It has nothing to do with your project's actual language.

---

# Part 3: Ten-Minute Command Orientation
*Reading time: ~4 minutes*

**What you will learn:** the one-line job of every command, and the review gate that must happen before you move to the next one.

| Command | Main question it answers | Review gate before moving on |
|---|---|---|
| `/constitution` | Which project-wide rules are non-negotiable? | Are the rules explicit and testable — not vague words like "clean" or "proper"? |
| `/specify` | What must the feature do? | Is the business behavior correct, and could a non-engineer understand it? |
| `/clarify` | What is still ambiguous? | Are important assumptions actually resolved? |
| `/plan` | How will it be built? | Is the architecture appropriate, with real rationale? |
| `/tasks` | In what order will the work happen? | Is the list complete, ordered, and executable? |
| `/analyze` | Do spec, plan, and tasks agree? | Are inconsistencies resolved *before* any code is written? |
| `/implement` | Can approved tasks be executed? | Is each increment reviewed and tested, not accepted blindly? |
| `/converge` | Does the final code match the artifacts? | Can the feature actually be merged? |

```
┌──────────┐   ┌────────┐   ┌─────────┐   ┌─────────┐   ┌──────────┐
│ Generate │──►│ Review │──►│ Correct │──►│ Approve │──►│ Continue │
└──────────┘   └────────┘   └─────────┘   └─────────┘   └──────────┘
```

> **Why treat every stage as a gate instead of one automatic pipeline?**
> Because letting the pipeline run end-to-end unreviewed just moves all your bugs one layer up — from "code bugs" to "spec bugs that got faithfully implemented." A gate at every stage is what actually catches problems while they're still cheap to fix.

---

# Part 4: Complete Todo Application Walkthrough
*Reading time: ~20 minutes — the most important part for a first-time user*

**What you will learn:** every command run for real, in order, on one continuous example, with the exact expected output and review checkpoint after each step.

## 4.1 Business problem

Users currently write daily tasks on paper or in temporary notes. They need a simple app to create, complete, delete, and filter Todo items.

## 4.2 Target behavior

Users must be able to: add a Todo with a required title and optional due date · mark complete/incomplete · delete a Todo · view all/active/completed · retain data after restart.

## 4.3 Step 1 — Create the constitution

```text
/constitution

Mandatory principles:
- Use Clean Architecture with Domain, Application, Infrastructure and API layers.
- Controllers must not directly access DbContext or any data-access API,
  including simple CRUD operations.
- All data access must go through an application service or repository abstraction.
- Expose REST-conventional resource endpoints using plural nouns and standard HTTP verbs.
- Use EF Core code-first migrations.
- Business logic must be covered by automated tests.
- React components must be functional components using hooks.
- No feature is complete until implementation is checked against its spec and plan.
```

**Expected output:** `constitution.md` created under `.specify/memory/`.

**✅ Review gate:** the rules apply to the *whole project*, not just Todo CRUD · vague words like "proper" or "best practice" are replaced by explicit, verifiable rules · no feature-specific field like `DueDate` appears here.

> **Why does "no business logic in controllers" need to be so specific?**
> Because a model can read that rule literally and decide `DbContext.Add()` counts as plumbing, not "business logic" — and place it directly in the controller anyway. This actually happened during this project's own pilot. State the prohibited behavior explicitly: *"including simple CRUD operations, with zero exceptions."*

## 4.4 Step 2 — Create the feature specification

```text
/specify

Users can:
- Add a Todo with a required title and optional due date.
- Mark a Todo complete or incomplete.
- Delete a Todo.
- Filter Todos by All, Active and Completed status.
- See validation feedback when the title is empty.
- Retain Todos after the application restarts.

Out of scope:
- Authentication, sharing, notifications, recurring Todos.
```

**Expected output:** `spec.md` with numbered functional requirements, user scenarios, edge cases, measurable success criteria, and scope boundaries — plus `requirements.md`, the quality checklist confirming the spec is ready to plan against.

**✅ Review gate:** can a business stakeholder understand it? · no mention of React/EF Core/SQL Server · every requirement testable · empty-title, missing-Todo, and invalid-ID cases addressed · out-of-scope stated clearly · **no `[NEEDS CLARIFICATION]` markers remain.**

## 4.5 Step 3 — Clarify ambiguity

```text
/clarify
```
Typical questions this surfaces: permanent vs. soft delete? title length limit? update-on-missing-Todo behavior? default filter?

**Recommended answers for this tutorial:** permanent delete · title 1–200 characters · missing Todo → HTTP 404 · default filter is All.

**✅ Review gate:** confirm the accepted answers are actually reflected back in `spec.md` before planning.

> **Why run `/clarify` at all — why not just start planning?**
> Because fixing ambiguity now takes about 5 minutes. Fixing the same ambiguity *after* implementation — once code, tests, and a database migration all assume the wrong answer — can take days.

## 4.6 Step 4 — Create the technical plan

```text
/plan

Backend: ASP.NET Core Web API on .NET 8, Clean Architecture layers,
EF Core code-first with SQL Server, repository/service abstraction
between controllers and persistence, REST endpoints under /api/todos,
xUnit tests.

Frontend: React + TypeScript, functional components and hooks,
a small API client module, components for add/list/item/filter.

Include: architecture rationale, data model, API contract,
validation approach, testing strategy, quickstart steps.
```

**Expected output:** `plan.md`, `research.md`, `data-model.md`, `quickstart.md` (exact file set can vary slightly by Spec Kit version — the installed templates are the real source of truth).

**✅ Review gate:** the plan actually satisfies the spec · controllers are *not* planned to touch `DbContext` directly · entity fields match what was approved · API responses/status codes documented · tests included, not deferred · every technical choice has stated rationale, not just a name.

## 4.7 Step 5 — Generate tasks

```text
/tasks
```
**Expected output:** an ordered list like:
```
T001 Create backend solution and projects.
T002 Configure project references and dependency boundaries.
T003 Create React TypeScript application.
T004 Add Todo domain entity and validation rules.
T005 Add application use cases / service interfaces.
T006 Implement EF Core DbContext and repository.
T007 Add initial migration.
T008 Implement REST API endpoints.
T009 Add backend tests.
T010 Add frontend API client.
T011 Add Todo form, list, item, filter components.
T012 Add frontend tests.
T013 Run integration validation and quickstart steps.
```

**✅ Review gate:** setup tasks come *before* feature tasks (T001/T002 above) · backend, frontend, DB, and testing all covered · dependencies correctly ordered · tasks small enough to actually review · no vague task like "finish remaining code."

## 4.8 Step 6 — Analyze consistency

```text
/analyze
```
Checks whether the active feature's `spec.md`, `plan.md`, and `tasks.md` agree. Real examples of what it catches: spec requires persistence but tasks only set up in-memory state · plan defines a repository but a task still calls `DbContext` from a controller · a field exists in `data-model.md` but never appears in the API contract.

**Resolve every high-severity finding before `/implement`.**

## 4.9 Step 7 — Implement in controlled increments

**Never request the entire full-stack app in one uncontrolled generation.**

```
Increment A — Setup:      backend/frontend project shells only
Increment B — Domain:     entity + validation + interfaces + unit tests
Increment C — Persistence: DbContext + repository + migration
Increment D — API:        REST endpoints, must call abstractions not DbContext
Increment E — Frontend:   (new chat session) components + real API client
Increment F — Integration: URLs, CORS, persistence, error handling
```
Example prompt for Increment A:
```text
/implement
Execute only the project setup tasks from tasks.md.
Create the backend solution and projects, configure references,
and create the React TypeScript project.
Do not implement Todo behaviour yet.
```
**✅ Review gate after every increment:** build it, review the diff, run tests — before moving to the next increment.

> **Why increments instead of one big `/implement` call?**
> Because a bad output in a 20-line increment costs you 2 minutes to fix. A bad output across the entire app costs you an afternoon of untangling which part went wrong.

## 4.10 Step 8 — Run the application

```bash
# Backend
dotnet restore
dotnet build
dotnet test
dotnet ef database update --project <InfrastructureProject> --startup-project <ApiProject>
dotnet run --project <ApiProject>

# Frontend
npm install
npm run dev
npm test
```
Follow the generated `quickstart.md` for actual project names/ports — they vary per project.

## 4.11 Step 9 — Validate the feature manually

Minimum acceptance checks: create with title only · create with title + due date · reject empty title · mark complete → incomplete · filter All/Active/Completed · delete · restart both servers, confirm data persists · unknown ID → 404 · **confirm controllers never touch the database directly.**

## 4.12 Step 10 — Converge

```text
/converge
Validate the completed implementation against constitution.md and the
active feature's spec.md, plan.md and tasks.md.
Report: missing requirements, plan contradictions, incomplete tasks,
constitution violations, missing tests, stale quickstart steps.
```
Correct any valid findings, then rerun validation.

## 4.13 Step 11 — Commit and raise a pull request

```bash
git add .
git commit -m "Implement Todo management feature using Spec Kit"
git push -u origin feature/todo-management
```
PR should include: business summary · link to the spec · architecture summary · tests run · any deviations and their approved artifact updates.

### Common Questions — Part 4

**"Why isn't `/implement` the first command?"** Because without a spec and plan first, the AI is guessing at requirements and architecture simultaneously — that's exactly the "prompt-driven" failure mode from Part 1.1.

**"Why can't I skip `/clarify`?"** You can, for trivial features. For anything real, skipped ambiguity doesn't disappear — it just resurfaces later as a bug, at a much more expensive point to fix.

---

# Part 5: Understanding the Generated Artifacts
*Reading time: ~8 minutes*

**What you will learn:** what to check *before* moving from one artifact to the next — a review checklist per transition, distinct from Part 2.7's "what each file is."

## 5.1 What to review before moving forward

**Before `/plan`:** spec is correct · scope bounded · no unresolved ambiguity · requirements measurable and testable.

**Before `/tasks`:** architecture acceptable · interfaces/boundaries clear · security/performance/data concerns addressed · technical decisions justified.

**Before `/implement`:** tasks cover the entire approved scope · setup ordered correctly · testing work included · `/analyze` reports no unresolved critical inconsistency.

**Before merge:** implementation passes tests · quickstart steps actually work · `/converge` findings resolved · artifacts and code describe the *same* system.

```
spec.md ──[gate]──► plan.md ──[gate]──► tasks.md ──[gate]──► code ──[gate]──► merge
```

---

# Part 6: Daily Development Lifecycle
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

This is simply Part 4's full walkthrough, compressed into the loop you'll repeat for every real feature going forward.

---

# Part 7: Existing Enterprise Applications
*Reading time: ~10 minutes — the most important part if you're not starting from a blank folder*

**What you will learn:** why Spec Kit can't see your existing codebase by default, how to fix that with `ProjectContext.md`, and the full brownfield workflow.

## 7.1 Why existing projects are different

Spec Kit has **no knowledge of your existing codebase.** Running `/specify: Introduce Result<T> across all APIs` on day one of a brownfield project means the AI doesn't know your architecture, controllers, schema, standards, or constraints — it will guess, and guesses conflict with what already exists.

## 7.2 The fix: `ProjectContext.md`

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

## 7.3 The brownfield workflow, full picture

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

## 7.4 Good vs. bad prompts on existing projects

**Bad:** `/specify: Add caching to the API.` — no context on what already exists.
**Good:**
```
/specify
Use docs/ProjectContext.md as context.
Add response caching to the /api/todos GET endpoint only.
Constraint: must not change the existing response schema.
```

## 7.5 Handling architectural drift

Someone changes the architecture outside the process under deadline pressure — now `constitution.md`/`plan.md` no longer match reality.

**Recovery:**
1. Audit: *"Compare the current codebase structure against plan.md and data-model.md. List every place they disagree."*
2. Update `constitution.md` if the change is now permanent.
3. Regenerate `plan.md`/`data-model.md` — don't hand-edit from memory.
4. Update `tasks.md` honestly — mark divergent tasks "implemented differently."
5. Log *why* the drift happened.

### Common Questions — Part 7

**"Why can't I just run `/specify` cold on our existing system?"** You can — but the AI will confidently guess at architecture it can't see, and those guesses will often actively conflict with your real code. `ProjectContext.md` is the fix.

**"Do I need `ProjectContext.md` for a brand new project too?"** No — Part 2's `specify init` (new project) doesn't need it. It's specifically for when code already exists before Spec Kit shows up.

---

# Part 8: Cross-Cutting and Repository-Wide Changes
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

# Part 9: Changing Requirements — Three Persistence Models
*Reading time: ~8 minutes*

**What you will learn:** the three official ways Spec Kit lets a spec evolve, and a decision tree for picking the right one every time.

Per Spec Kit's own docs ([evolving-specs.md](https://github.com/github/spec-kit/blob/main/docs/guides/evolving-specs.md)):

## 9.1 Flow-Forward Spec
New requirement → new `specs/00X-.../` folder. Old ones stay untouched, permanent history.
*Analogy: a diary — you don't rewrite yesterday's entry, you write a new one.*

## 9.2 Living Spec
A requirement *changes existing behavior* → edit the existing `spec.md` in place, regenerate `plan.md`/`tasks.md`, re-validate with `/analyze`.
*Analogy: a Wikipedia page — continuously edited to reflect current truth.*

## 9.3 Flow-Back Spec
An *insight discovered mid-work* (during `/plan`, `/tasks`, or `/implement`) should reshape an earlier artifact. Capture it wherever it happened → decide what it actually changes → update whichever artifacts now disagree → run `/analyze` → only continue once the artifact set is trustworthy again.
*Analogy: editing a novel — a Chapter 10 realization sends you back to fix Chapter 3.*

**Official caveat, worth repeating verbatim:** *"Flow-back is flexible, but it requires discipline. Do not leave a lower-level change in tasks.md or code if spec.md still says something different."* — this is the formal name for the drift failure mode in Part 7.5; Flow-Back is the *disciplined, intentional* version of that scenario.

## 9.4 All Three, Side by Side

| | Flow-Forward | Living Spec | Flow-Back |
|---|---|---|---|
| Trigger | Genuinely new capability | A planned requirement change | A mid-work discovery |
| Starting point | New `specs/00X/` folder | Always `spec.md` first | Wherever the insight landed |
| Direction | Forward only | Forward only, re-run per change | Can move **backward** |
| Validation | `/analyze` less critical | `/analyze` before resuming | `/analyze` is essential — highest drift risk |

> ⚠️ **Separate risk:** these three models govern your *feature artifacts*. `specify init --here --force` (refreshing Spec Kit's own project files) is different — it can **overwrite your customized `constitution.md`** unless backed up first.

### Common Questions — Part 9

**"Why can't I just always use Flow-Forward — isn't it the simplest?"** You can, but old specs pile up disconnected from each other, and a feature that's genuinely still evolving ends up scattered across several folders instead of one coherent document.

**"Isn't Flow-Back just... fixing bugs?"** Not quite — Flow-Back specifically means the fix reshapes an *earlier artifact* (the spec or plan), not just the code. If you only touch code and never reconcile the spec, that's uncontrolled drift, not disciplined Flow-Back.

---

# Part 10: Working in Teams
*Reading time: ~3 minutes*

**Solo developer:** self-review the spec before `/plan` — read it as a stranger would.
**2–5 developers:** one prepares the spec, one teammate reviews before `/plan` runs.
**Larger/cross-team:** add an architecture review step before `/plan` for shared infrastructure; consider a shared `constitution.md` preset (Part 16.4) so every team enforces the same standards.

---

# Part 11: Best Practices
*Reading time: ~3 minutes*

✅ One feature per specification · reference files with `#filename`, don't paste content · review every generated spec/plan before proceeding · run `/clarify` and `/analyze` — cheap, catches problems early · create `ProjectContext.md` once, maintain after major changes · implement layer by layer · run `/converge` — "it compiled" ≠ "it matches the spec" · periodically test constitution compliance directly on a small feature.

---

# Part 12: Common Mistakes
*Reading time: ~3 minutes*

❌ Running `/specify` on an existing project without `ProjectContext.md` · jumping to `/implement` before setup tasks are done · combining unrelated features into one spec · pasting whole files into chat instead of referencing them · treating AI output as production-ready without review · letting `plan.md`/`tasks.md` go stale after out-of-process changes · assuming a "MUST" constitution rule guarantees compliant code.

---

# Part 13: Troubleshooting
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
Not a bug — constitution rules aren't compiler-enforced (see the Part 4.3 "why" box). Use `/converge` to check compliance directly, then request a targeted fix.

**Direct DB calls inside a controller, or non-REST endpoints**
Same root cause as above — tighten the specific constitution wording (e.g., "no exceptions, including simple CRUD") and regenerate just that layer.

---

# Part 14: Prompt Cookbook
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

# Part 15: Templates and Checklists
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

# Part 16: Enterprise Adoption and Governance
*Reading time: ~10 minutes*

## 16.1 Start with a pilot

Use a bounded but realistic feature: more than one layer/component, meaningful acceptance criteria, a real DB/API change, automated tests, at least one requirement change *during* the pilot. **Don't judge enterprise fit from a blank Todo project alone** — the Todo tutorial validates mechanics; a brownfield pilot validates real organizational fit.

## 16.2 Pilot success measures

Clarification issues found before coding · rework avoided/introduced · spec/plan review time · implementation cycle time · defect/regression rate · % of acceptance criteria with test evidence · artifact drift found by `/analyze`/`/converge` · stakeholder usability feedback · AI token/credit consumption.

## 16.3 Organization-level standards worth considering

Constitution baseline · required spec sections · security/compliance sections · ADR format · task traceability expectations · PR evidence requirements · artifact ownership · retention/versioning approach · permitted tools and data-handling rules.

## 16.4 Extensions and Presets

**Extensions** add entirely new commands/capabilities (Jira sync, security review command, project discovery, test traceability). `specify extension search` / `specify extension add <name>`.

**Presets** override existing templates to enforce org standards (mandatory NFR sections, compliance-oriented acceptance criteria, standard task ordering). `specify preset search` / `specify preset add <name>`.

| | Extensions | Presets |
|---|---|---|
| Adds new commands? | Yes | No |
| Changes existing templates? | No | Yes, overrides in place |
| Analogy | Installing a new app | Changing settings on an existing app |

Validate available commands against your installed Spec Kit version before org-wide rollout.

## 16.5 Multi-repository pattern

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

## 16.6 Preventing specification sprawl

Keep historical Flow-Forward specs as an untouched audit trail · maintain a separate, periodically-regenerated "current system overview" (not hand-maintained like a BMAD-style PRD) · link capabilities to relevant specs/code · never manually merge dozens of old specs into one unreviewable document.

## 16.7 Classify failures correctly

**Intent-to-spec gap** — the requirement conversation or spec missed something. Fix: discovery, clarification, spec templates, stakeholder review, constitution wording (this is exactly what the Part 4.3 REST/DbContext example was).

**Spec-to-implementation gap** — the spec was right, code diverged anyway. Fix: task precision, smaller increments, tests, `/converge` validation, PR review.

## 16.8 Recommended rollout stages

```
Stage 1: Learn        → complete the Todo tutorial + orientation
Stage 2: Pilot         → one bounded brownfield feature
Stage 3: Standardize   → shared templates, constitution baseline, checklists
Stage 4: Integrate     → connect to backlog, CI, testing, PR governance
Stage 5: Scale         → cross-team adoption, metrics, ongoing template improvement
```

---

# Part 17: The Long-Term Vision (Strategic, Not Tactical)
*Reading time: ~12 minutes*

**What you will learn:** where the industry believes this category is heading — useful vocabulary for strategic conversations, clearly separated from what Spec Kit actually does today.

## 17.1 "SpecFall" — the core organizational risk

Named after "Scrumerfall" (adopting Agile ceremonies without changing real collaboration). **SpecFall** is the same failure applied to SDD: running `/specify → /plan → /tasks` mechanically, without changing how product/architecture/engineering/QA actually work together, produces a **"markdown monster"** — technically valid files nobody actually uses as a real collaboration surface.

A solo pilot (like the Todo app) proves the *mechanics* work. It doesn't yet prove the *collaboration* value — which is the bigger prize at enterprise scale.

## 17.2 Seven real tooling gaps at enterprise scale

1. **Developer-centric tooling** — Git/CLI-based, real friction for PMs/BAs who should own the "what."
2. **Mono-repo assumption** — most tools, Spec Kit included, don't have a clean answer for specs spanning multiple repos.
3. **No separation by audience/lifecycle** — strategic architecture decisions and tactical task lists share one folder.
4. **Unclear starting point** — most orgs already have refined Jira/Azure DevOps backlogs; integration isn't built in.
5. **Undefined collaboration patterns** — no built-in clarity on who approves what, when.
6. **No standard spec style across tools** — Spec Kit's format, Amazon Kiro's EARS format, etc. all diverge.
7. **Unclear brownfield path** — spec the whole legacy system, or build incrementally? **Incremental is right** — comprehensively specing a large legacy system risks exceeding context limits, and even if it succeeds, the result is too large for a human to meaningfully review, defeating the purpose. This directly validates Part 7's `ProjectContext.md`-first, incremental approach.

## 17.3 Multi-repo orchestration, the fuller version

```
Product Owner  → articulates business "what" (stays stable long-term)
        ↓
Architect      → breaks it into repo-specific sub-issues, scoped boundaries
        ↓
Engineer (per repo) → implementation tasks scoped to just that repo
```
Architects don't manually decompose every story by hand — document repo boundaries and constraints *once*, as a reusable harness, and let the agent apply it to new incoming stories.

## 17.4 Role-specific "harnesses" beyond the architect's constitution

Infrastructure specialists → deployment constraints · Performance specialists → optimization requirements · Security specialists → compliance requirements — each becomes an additional automatically-applied "constitution," catching concerns *before* implementation instead of in a review gate afterward.

## 17.5 The provocative claim: should even tiny fixes go through the spec?

The stronger argument: yes, even small bug fixes — because a direct code patch can be **silently overwritten the next time that code is regenerated by AI**, since the spec never captured the fix and AI generation is non-deterministic. Same logic as why teams stopped allowing direct production server changes: not a rule for its own sake, but because direct changes get silently overwritten by the next deployment.

**[VALIDATE ON PILOT]** — stronger discipline than anything in Parts 1–16. Worth testing deliberately: does skipping the spec for a "trivial" fix actually cause recurrence, or does this matter more at higher AI-autonomy scale than one pilot will reveal?

## 17.6 Harness governance — classify where bugs actually come from

Same as Part 16.7, applied at the philosophical level: a **spec-to-implementation gap** means the process worked but execution didn't — fix validation. An **intent-to-spec gap** means the process itself has a blind spot — fix the harness/constitution/template, or the same bug class recurs across every future feature.

## 17.7 The more radical vision: "architecture becomes executable"

A second, more theoretical school of thought (Griffin & Carroll, InfoQ) proposes something more extreme: what if the **spec**, not the code, became the actual protected source of truth — with code treated as fully disposable, regenerated on demand, the way a `bin/` folder is disposable today?

**Five-layer model:** Specification (declares what must be true, zero implementation detail) → Generation (a "multi-target compiler" turning spec into code/tests/docs) → Artifact (the generated code itself — disposable) → Validation (continuously checks running behavior against the spec) → Runtime (the live system, whose behavior is guaranteed by Layer 1, not decided independently here).

**"Architectural inversion":** today, if code and docs disagree, code wins — docs get updated to match. This vision inverts it: if code and spec disagree, the **spec** wins — code gets regenerated to match it.

**Humans are "relocated, not removed":** the AI owns *mechanical* enforcement (does the code match the spec). Humans retain authority over *meaning* — approving breaking changes, authorizing policy shifts, judging whether a trade-off is acceptable. Machines own "did we do what we said." Humans own "do we actually want to do this."

**Honest reality check — this is a future destination, not today's road:**

| Vision | Spec Kit's actual reality today |
|---|---|
| Regenerating from the same spec always gives identical code | Not true — AI generation varies (exactly why the Part 4.3 constitution-gap example happened) |
| Continuous, automatic drift detection | Manual — you run `/analyze`/`/converge` yourself |
| Code fully disposable, zero information lost if deleted | Not true yet — real decisions still get made *during* `/implement` that aren't fully captured back in the spec |
| Spec always wins when code and spec disagree | Currently the reverse is usually easier — audit code, update the spec to match (Part 7.5) |

**Why this section is worth reading anyway:** it gives you sharp vocabulary — *SpecOps, architectural inversion, bounded autonomy* — for sounding sophisticated in strategic conversations about where this category is heading, without mistaking it for what you should actually do on tomorrow's pilot.

---

# Part 18: FAQ
*Reading time: ~5 minutes*

**Should every change use the full workflow?** No — scale to risk/complexity/impact. Any change that alters accepted behavior should still leave a durable record of the corrected intent and its tests.

**Is `/constitution` run for every feature?** No — once per project, updated only when a project-wide principle genuinely changes.

**Can generated Markdown be edited manually?** Yes, expected. After editing an upstream artifact, reconcile the downstream ones.

**Does Spec Kit automatically know about my other feature specs?** Not reliably. The agent *can* inspect the repo when explicitly instructed, but commands operate primarily on the *active* feature — no automatic cumulative reconciliation across every historical spec. Use `ProjectContext.md` and explicit impact prompts (Part 8).

**Can 1–2 developers use this without multiple "AI roles"?** Yes — the essential controls are artifact quality, review, and validation, not the number of personas involved.

**Difference between `/analyze` and `/converge`?** `/analyze` checks spec/plan/tasks agree *before* code exists (pre-flight). `/converge` checks the *real code* against all three *after* implementation (landing check).

**Does a constitution guarantee compliant code?** No — use precise wording, tests, analyzers, CI, and `/converge`, not trust alone.

**How are rejected changes rolled back?** Git branches, PRs, commits, reverts, feature flags, DB recovery practices — Spec Kit doesn't replace source-control rollback.

**How should a new requirement be handled after implementation?** Decide: new capability (Flow-Forward), change to existing behavior (Living Spec), or mid-work discovery (Flow-Back) — see Part 9.

**How do I control Copilot credit usage?** Focused prompts · reference artifacts with `#filename` instead of pasting · implement in small increments · new chat sessions for major context switches · review before regenerating · never ask for the whole app repeatedly.

---

# Part 19: Glossary
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

# Part 20: Decision Trees
*Reading time: ~2 minutes*

## Which persistence model should I use? (Part 9)

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
        ├── YES ──► Create/update ProjectContext.md BEFORE /specify (Part 7)
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
                        ├── YES ──► Full workflow (Parts 3-4)
                        │
                        └── NO, small and isolated ──► Lighter touch is fine,
                             but still update spec.md if behavior changed
                             (Part 17.5 — even small fixes ideally leave
                              a durable record)
```

---

# Part 21: What Good Looks Like
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
