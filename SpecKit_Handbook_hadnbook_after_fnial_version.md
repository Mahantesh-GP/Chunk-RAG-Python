# Specification-Driven Development with Spec Kit
### A Practical Handbook — From Zero to Your First Enterprise Feature

**Maintained by:** Mahantesh, EA GenAI Team (SPOC for Spec Kit)
**Audience:** Developers, Tech Leads, Architects, New Joiners
**Style note:** This isn't a rewrite of GitHub's docs. GitHub's docs teach you *how to run commands*. This handbook teaches you *how to use Spec Kit successfully on a real project* — including the parts nobody documents, like what to do with an existing enterprise codebase, what each artifact actually depends on, and what to do when something breaks.

---

# Part 1 — Foundations

## 1.1 What is Specification-Driven Development?

You're new to Spec Kit? Let's start with the idea before the tool.

Normally, when you ask an AI assistant to "build a login feature," it jumps straight to code. That works for small things. It falls apart on real projects, because:

- The AI has no memory of *why* you built things a certain way last week.
- Requirements and implementation details get tangled into one prompt — so when requirements change, you re-explain everything from scratch.
- Nobody can point to *why* a decision was made six weeks later — it's buried in a chat log nobody re-reads.

**SDD's fix:** write down intent — requirements, architecture decisions, constraints — as durable files the AI re-reads every time, instead of relying on conversation memory.

```
Requirement
    ↓
Specification   (what — no tech detail)
    ↓
Plan            (how — architecture, stack)
    ↓
Tasks           (ordered steps)
    ↓
Implementation  (actual code)
```

## 1.2 Traditional Development vs. SDD

```
Traditional AI-assisted coding          Specification-Driven Development
─────────────────────────────          ──────────────────────────────────
"Build me a login page"                "/specify: users need to log in..."
        ↓                                       ↓
AI guesses architecture                spec.md → reviewed → approved
        ↓                                       ↓
Code appears, maybe right               /plan → architecture decided,
        ↓                                        documented with rationale
Next session: AI has forgotten                  ↓
why anything was built this way        /tasks → ordered, trackable steps
                                                 ↓
                                        /implement → code, traceable back
                                                     to an approved spec
```

## 1.3 Where Spec Kit Fits

Spec Kit is GitHub's open-source implementation of SDD, wired into GitHub Copilot Chat as slash commands:

| Command | Purpose |
|---|---|
| `/constitution` | Project-wide, non-negotiable engineering principles |
| `/specify` | Define the feature specification (what, not how) |
| `/clarify` | Surface ambiguity before planning begins |
| `/plan` | Create the technical design and architecture rationale |
| `/tasks` | Break the plan into ordered, numbered implementation steps |
| `/analyze` | Cross-check spec/plan/tasks for consistency before coding |
| `/implement` | Generate the actual code |
| `/converge` | Validate the actual implementation against the active feature's spec/plan/tasks (post-implementation check) |

## 1.4 Benefits and Limitations — Be Honest About Both

**Benefits:**
- Requirements and rationale become durable, reviewable artifacts, not chat history.
- Consistent structure across features and across engineers.
- Easier onboarding — a new teammate can read `spec.md`/`plan.md` instead of asking you to explain verbally.

**Limitations (learned the hard way, not from a brochure):**
- Spec Kit doesn't automatically know your existing codebase — more on this in Part 4.
- Constitution rules are *guidance* the AI is supposed to follow, not a hard compiler-enforced constraint — generated code can still partially miss a rule.
- If someone changes the architecture outside the process, the docs go stale and nothing detects this automatically — you have to catch it.
- Cost isn't eliminated, it's redistributed — more upfront time in spec/plan, ideally less wasted time later.

---

# Part 2 — Getting Started

## 2.1 Prerequisites

- .NET SDK 8.x, Node.js 20.x + npm, SQL Server (adjust to your actual stack — Spec Kit is stack-agnostic)
- GitHub Copilot Business, signed in
- Spec Kit CLI:
  ```bash
  pip install uv
  uv tool install specify-cli --from git+https://github.com/github/spec-kit.git
  specify check
  ```

## 2.2 Starting a New Project

```bash
specify init <project-name> --integration copilot
cd <project-name>
code .
```
This scaffolds two folders:
- `.github/prompts/` — your slash commands, discovered automatically by Copilot Chat
- `.specify/` — templates and helper scripts used behind the scenes

No Copilot credits are used yet. This is local scaffolding only.

## 2.3 Starting on an Existing Repository

```bash
specify init --here --integration copilot
```
The `--here` flag installs into your current repo instead of a new folder. **Do not run `/specify` yet** if this is your situation — see Part 4 first, it changes everything about how you should proceed.

## 2.4 CLI Reference Beyond `init`

| Command | Purpose |
|---|---|
| `specify check` | Verifies your environment has the required tools/versions installed |
| `specify init <name>` | Scaffolds a new project |
| `specify init --here` | Scaffolds into the current directory (brownfield) |
| `specify init --here --force` | **Overwrites** existing Spec Kit project files (commands, scripts, templates) with the latest version. See the warning in Section 4.8 — this can destroy your customized `constitution.md` unless backed up first |
| `specify upgrade` | Updates Spec Kit itself to a newer release, distinct from evolving your feature specs (see Section 4.8) |
| `specify extension search` / `specify extension add <name>` | Adds new commands/capabilities — see Part 8 |
| `specify preset search` / `specify preset add <name>` | Overrides existing templates to enforce org standards — see Part 8 |

## 2.5 The Artifact Dependency Map — What Actually Reads What

This is the part most newcomers get wrong: assuming every artifact independently talks to every other artifact. It doesn't work that way. Here's the real dependency chain:

```
constitution.md  ─────────────────────────────────────────┐
   (governs everything below — read by every command)     │
                                                             ▼
spec.md  ──────────►  requirements.md (checklist)
   │                     (quality-gates spec.md itself,
   │                      before planning is allowed)
   ▼
/plan  ──────────►  research.md   (decision rationale)
   │            └─►  data-model.md (entity design)
   │            └─►  plan.md       (the technical design itself,
   │                                references research.md +
   │                                data-model.md)
   ▼
/tasks  ──────────►  tasks.md
   (derived from plan.md, which is transitively
    derived from spec.md — tasks.md does NOT re-read
    spec.md directly, it trusts plan.md as the source)
   ▼
/implement  ──────────►  actual code
   (executes tasks.md; also has access to constitution.md,
    plan.md, and data-model.md for architectural context
    while writing code)
   ▼
quickstart.md  (generated during /plan, used AFTER /implement)
   (validates the finished feature against spec.md's original
    acceptance criteria — this is the one artifact that reaches
    back to the very start of the chain, on purpose)
   ▼
/analyze  (can run any time after /tasks, before/after /implement)
   (cross-checks spec.md ↔ plan.md ↔ tasks.md agree with each other
    — does NOT check other, unrelated feature folders)
   ▼
/converge  (runs after /implement)
   (checks the actual written code against spec.md/plan.md/tasks.md
    for this specific feature — the final proof the loop closed)
```

**The one rule that matters most:** `constitution.md` is read by every downstream command (`/plan`, `/tasks`, `/implement`, `/analyze`, `/converge`) — it's the only artifact that isn't "owned" by one specific stage. Everything else follows a strict pipeline: **spec → plan → tasks → implement**, each stage trusting the *previous* stage's output rather than re-deriving from the original spec every time. This is exactly why drift is dangerous (Section 4.6) — if `plan.md` goes stale, `tasks.md` inherits that staleness, and `/implement` never independently double-checks against the original `spec.md`.

## 2.6 Every Artifact File, Explained — What It Is, Why It Exists, What It Contains

| File | Created by | Why it exists | What it actually contains |
|---|---|---|---|
| `constitution.md` | `/constitution` | Every other command reads this — without it, every feature could use a different architecture style | Non-negotiable, project-wide rules: architecture style, testing policy, layering conventions |
| `spec.md` | `/specify` | Captures *what* the feature does, deliberately free of implementation detail, so a non-technical stakeholder could read it | Numbered functional requirements (FR-001, FR-002...), user scenarios, edge cases, acceptance criteria |
| `requirements.md` (checklist) | `/specify` | A **quality gate** — GitHub itself describes this as "unit tests for English." It checks `spec.md`'s own clarity and completeness *before* you're allowed to move to planning — it does not check code or later artifacts | A checklist: no implementation details leaked in, no unresolved `[NEEDS CLARIFICATION]` markers, requirements are testable, success criteria are measurable, edge cases identified, scope bounded |
| `research.md` | `/plan` | Your architecture decision record (ADR) — captures *why* a technical choice was made, not just what was chosen, so decisions are auditable later without re-reading code | For each major decision: the decision itself, the rationale, and the alternatives considered and rejected (e.g., "ASP.NET Core Web API chosen over minimal APIs because...") |
| `data-model.md` | `/plan` | The source of truth for entity design — what your actual EF Core entities/migrations should match | Entities, fields, types, relationships (e.g., Todo entity: GUID id, title, isCompleted, optional dueDate) |
| `plan.md` | `/plan` | The overall technical design — the "how" that complements spec.md's "what" | Architecture decisions, stack choices, layering approach; references `research.md` and `data-model.md` as supporting detail rather than repeating them |
| `tasks.md` | `/tasks` | Breaks the plan into ordered, numbered, trackable steps that `/implement` actually executes against | Numbered tasks (T001, T002...) — almost always starting with **project setup** tasks (create backend/frontend project shells) before feature-level tasks |
| `quickstart.md` | `/plan` (used after `/implement`) | A **validation runbook** — proves the *finished* feature actually satisfies the original `spec.md`, closing the loop back to where the process started | Step-by-step manual or scripted validation steps, referencing spec.md's acceptance criteria directly |

**Why this layered structure matters:** each artifact answers one specific question, and no artifact tries to do another artifact's job. `spec.md` never mentions tech stack. `plan.md` never restates requirements verbatim. `tasks.md` never re-justifies architecture. This separation is what makes the whole system reviewable — a reviewer can check "does this plan make sense" without wading through requirements language, and check "was this requirement met" via `quickstart.md` without re-reading the entire implementation.

---

# Part 3 — Every Command, Explained

Each section below answers: why does this exist, what do you feed it, what comes out, and what mistakes to avoid.

## 3.1 `/constitution`

**Why it exists:** Every other command reads this file. Without it, every feature might use a different architecture style, because nothing is enforcing consistency across features.

**What to provide:** Standing rules that apply to *every* feature in this project — never feature-specific detail.

```
/constitution
Principles: tests written before implementation (Playwright E2E per requirement);
Clean Architecture layering (Domain/Application/Infrastructure/API);
EF Core code-first with migrations; no business logic in controllers;
React functional components with hooks only.
```

**Common mistake:** putting feature-specific details here (e.g., "the Todo app needs a due date field"). Test: if the rule would still apply to a totally different feature in the same project, it belongs here. If not, it belongs in `spec.md` or `plan.md`.

**Run this once per project**, not once per feature.

### A Real Example: Why Vague Rules Get Interpreted Narrowly

This actually happened while piloting the Todo app, and it's a genuinely important lesson: **the constitution only prevents what it explicitly names — anything unstated gets filled in with whatever's convenient, not what you assumed was implied.**

The constitution said "no business logic in controllers" and "Clean Architecture layering" — but never explicitly said "expose REST-conventional endpoints" or "controllers must never directly call the database." The result:

- Controller methods came out as plain CRUD actions rather than clearly REST-conventional resource endpoints, because "REST" was never named as a requirement — only "Clean Architecture" was, which doesn't by itself dictate URL/verb conventions.
- Data access calls like `dbContext.Add(...)` / `dbContext.SaveChangesAsync()` appeared **directly inside the controller**, bypassing any repository or service layer — even though "no business logic in controllers" was in the constitution.

**Why this happened:** the model apparently treated plain data access (`Add`, `SaveChanges`) as plumbing, not "business logic" — so the rule didn't trigger for it. The rule was true to its literal wording; it just didn't cover the case the author actually meant.

**The fix — be maximally explicit, don't rely on implied scope:**
```
/constitution
...
Controllers MUST NOT directly reference DbContext or any data access API —
all data access MUST go through a repository or service interface, with zero
exceptions, including simple CRUD operations.
Expose REST-conventional, resource-based endpoints (GET/POST/PUT/DELETE on
plural nouns like /api/todos), not RPC-style action methods.
```

**The lesson to carry forward:** every constitution rule should be read by imagining the *laziest possible technically-true interpretation* of it, and then made explicit enough to close that gap. "No business logic in controllers" sounds complete — it wasn't, until "including simple CRUD operations, with zero exceptions" was added. This is worth doing for every rule before you trust a constitution on real project work, not just this one.

## 3.2 `/specify`

**Why it exists:** Turns your plain-English description into a structured spec with numbered functional requirements — deliberately with zero implementation detail, so a non-technical stakeholder could read and understand it.

**Todo App example:**
```
/specify
Todo app. Users can:
- Add a task with title and optional due date
- Mark a task complete/incomplete
- Delete a task
- Filter tasks by status (all/active/completed)
```

**Output:** `spec.md` with FR-001–FR-004, plus a `requirements.md` quality checklist confirming the spec is ready for planning.

**Common mistakes:**
- Mixing multiple unrelated features into one spec — keep one feature per specification.
- Mentioning implementation details ("use React state") — this belongs in `/plan`, not here.

## 3.3 `/clarify`

**Why it exists:** Surfaces ambiguity *before* you commit to a plan, offering you a few candidate answers instead of forcing you to write clarifications from scratch.

**When to run it:** Immediately after `/specify`, before `/plan`. Don't skip this — it's cheap (small credit cost) and catches misunderstandings early, when they're cheap to fix.

## 3.4 `/plan`

**Why it exists:** Converts the "what" into a concrete "how" — architecture, stack, entity design — with documented reasoning (`research.md`) so decisions are auditable later.

**Todo App example:**
```
/plan
Backend: ASP.NET Core Web API, EF Core, SQL Server, Clean Architecture layers.
Frontend: React, useState/useReducer, no external state library.
Endpoints: GET/POST/PUT/DELETE /api/todos.
```

**Output:** `plan.md`, `research.md` (decision + rationale + alternatives considered), `data-model.md` (entity design), `quickstart.md` (validation runbook for later).

**Common mistake:** accepting the plan without reading `research.md` — this is where you catch a bad architectural choice before any code gets written, which is far cheaper than catching it after `/implement`.

## 3.5 `/tasks`

**Why it exists:** Breaks the plan into ordered, numbered, trackable steps (T001, T002...) that `/implement` executes against.

**Critical thing to check every time:** the first 1–2 tasks are almost always **project setup** (create the backend project shell, create the frontend project shell). Skipping straight to feature-level implementation without these produces code fragments with nowhere to run — this is the single most common failure newcomers hit.

## 3.6 `/analyze`

**Why it exists:** Runs after `/tasks`, before `/implement`. Cross-checks that spec, plan, and tasks actually agree with each other — catching disagreement before code gets written rather than after.

**When to run it:** Every time, right before your first `/implement` call on a feature. It's cheap and it's your first line of defense against drift. It only checks the *active feature's* own artifacts — it does not reconcile against other, unrelated feature folders (see Section 4.8's FAQ note on cross-feature awareness).

## 3.7 `/implement`

**Why it exists:** Executes `tasks.md`, actually writing code.

**This is the expensive step** — do it layer by layer, not as one giant prompt:
```
/implement
Generate only: Todo entity (Models/TodoItem.cs) and AppDbContext (Data/AppDbContext.cs).
Reference #tasks.md for field list. Nothing else yet.
```
Then, same chat session, follow-up messages don't need `/implement` repeated — just plain instructions, since context is already loaded:
```
Now generate the EF Core migration and repository/service layer.
```
```
Now generate TodosController with the 4 REST endpoints.
```
Switch to frontend in a **new chat session** (old sessions resend full history as tokens every turn):
```
/implement
Generate React components AddTodo, TodoList, TodoItem, FilterBar per #spec.md.
```

**Common mistake:** one giant prompt asking for the entire feature at once. If something's wrong, you're now debugging a huge diff instead of a small one.

## 3.8 `/converge`

**Why it exists:** After `/implement` has written code, `/converge` checks the *actual, finished code* against `spec.md`, `plan.md`, and `tasks.md` for that specific feature — it's the final proof the loop actually closed, not just that code was generated.

**How it differs from `/analyze`:** `/analyze` checks whether spec/plan/tasks agree with *each other*, before code exists. `/converge` checks whether the *real code* agrees with all three, after implementation. Think of `/analyze` as a pre-flight check and `/converge` as a landing check.

**When to run it:** After `/implement` completes for a feature, before you consider it done. This is also your best tool for the constitution-compliance testing described in Part 6 — ask it directly whether the generated code actually follows every constitution rule, not just whether it compiles.

```
/converge
Verify the implemented TodosController.cs and related files against
spec.md, plan.md, and tasks.md for this feature. Report any place the
actual code disagrees with what was specified or planned.
```

**Common mistake:** treating `/implement` finishing without errors as proof the feature is done. Compiling successfully and matching the spec are two different things — `/converge` checks the second one.

---

# Part 4 — Existing Enterprise Applications (The Most Important Part)

This is the chapter that doesn't exist in the official docs — and it's the one that matters most if you're not starting from a blank folder.

## 4.1 Why Existing Projects Are Different

Spec Kit has **no knowledge of your existing codebase.** If you run:
```
/specify
Introduce Result<T> across all APIs.
```
...on day one of a brownfield project, the AI doesn't know your current architecture, your existing controllers, your database schema, your coding standards, or your constraints. It will guess — and its guesses will often conflict with what already exists.

## 4.2 The Fix: `ProjectContext.md`

This is a **recommended team practice, not a built-in Spec Kit feature.** Create `docs/ProjectContext.md` once per repository, before your first `/specify` call.

**Template:**
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

Keep it concise — 2 to 5 pages, not a full architecture document. Pull the content from your README, solution structure, existing controllers, database schema, and any architecture docs you already have.

> 💡 **Optional shortcut:** If your org has access to it, you can use the **`project-discovery`** plugin (published on EA-Marketplace) to auto-generate a first-pass discovery report instead of writing this by hand. It scans the repo and produces an evidence-based report — citing actual files (controllers, services, DB context, etc.) rather than guessing — covering architecture, project hierarchy, and existing capabilities in much more depth than the manual template below. Review its output and distill it into `docs/ProjectContext.md` before your first `/specify` call. This is a recommended time-saver, not a required step — the manual template still works fine if the plugin isn't available to you.
>
> **Next step after running the plugin:** the discovery report is intentionally thorough (it can run 15+ sections) — don't hand it straight to `/specify` as-is. Ask Copilot Chat to condense it into the concise `ProjectContext.md` template:
> ```
> Using todo-app_project_discovery_report.md as the source, generate a concise
> docs/ProjectContext.md following this template: Overview, Technology Stack,
> Architecture, Existing Features, APIs, Database, External Integrations,
> Coding Standards, Constraints, Non-functional Requirements, Known Technical Debt.
> Keep it to 2-5 pages — summarize, don't copy sections verbatim.
> ```
> Review the result before your first `/specify` call — the plugin gives you complete evidence, but `ProjectContext.md` should stay short enough that every later command can cheaply re-read it.

**Example (Todo App, brownfield):**
```markdown
# Project Context

Backend: ASP.NET Core Web API
Frontend: React
Database: SQL Server

Architecture: Clean Architecture, Repository Pattern
Authentication: JWT

Existing Features: Login, Todo CRUD, Search

Constraints:
- No breaking API changes
- Maintain Swagger compatibility
```

## 4.3 Using `ProjectContext.md` with `/specify`

```
/specify

Use docs/ProjectContext.md as the current system context.

New Requirement: Introduce Result<T> API response pattern.

Generate:
- Functional requirements
- Non-functional requirements
- Migration strategy
- Impact analysis
- Acceptance criteria
```

Now the spec is grounded in what actually exists, not a generic assumption.

## 4.4 The Brownfield Workflow, Full Picture

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
Review                         ← architect/tech lead sign-off
        ↓
/plan → /tasks → /implement → /converge
```

## 4.5 Good vs. Bad Prompts on Existing Projects

**Bad:**
```
/specify
Add caching to the API.
```
No context about what "the API" currently looks like, what's already cached, or what caching library the project standardizes on.

**Good:**
```
/specify
Use docs/ProjectContext.md as context.
Add response caching to the /api/todos GET endpoint only.
Constraint: must not change the existing response schema.
Generate functional requirements, cache invalidation strategy, and acceptance criteria.
```

## 4.6 Handling Architectural Drift

This will happen — someone changes the architecture outside the Spec Kit process under deadline pressure, and now `constitution.md`/`plan.md` no longer match reality.

**Recovery:**
1. Audit first: *"Compare the current codebase structure against plan.md and data-model.md. List every place they disagree."*
2. Update `constitution.md` if the change is now the permanent standard.
3. Regenerate `plan.md`/`data-model.md` — don't hand-edit from memory.
4. Update `tasks.md` honestly — mark divergent tasks "implemented differently," don't blanket-check them.
5. Log *why* the drift happened, so the next person understands it wasn't an oversight.

## 4.7 Update `ProjectContext.md` After Significant Changes

Treat it like a living document — after any major architecture shift, update it. A stale `ProjectContext.md` is worse than none, because it gives false confidence.

## 4.8 Three Specification Persistence Models

Per Spec Kit's own documentation ([evolving-specs.md](https://github.com/github/spec-kit/blob/main/docs/guides/evolving-specs.md)), there are **three** models — pick deliberately, per feature, rather than drifting into one by accident.

**1. Flow-Forward Spec**
Use when each feature directory should remain a historical record. Every new requirement gets its own new `specs/00X-feature-name/` directory; the previous one stays untouched for audit, comparison, or explaining how the project reached its current state.
*Analogy: a diary. Each entry is dated and permanent — you don't rewrite yesterday's entry, you write a new one.*

**2. Living Spec**
Use when `spec.md` is the contract and `plan.md`/`tasks.md` are derived from it. When intended behavior changes, revise the existing `spec.md` first, then regenerate the downstream artifacts to match — run `/analyze` before implementation resumes to catch gaps.
*Analogy: a Wikipedia page. It's continuously edited to reflect current truth — you don't create "version 2" as a separate page.*

**3. Flow-Back Spec**
Use when implementation discoveries are allowed to reshape the artifact set — i.e., insight doesn't have to start at the spec. The first useful edit can happen wherever it naturally lands: `spec.md`, `plan.md`, `tasks.md`, or the code itself. After the change, you bring everything else back into alignment:
1. Capture the discovery in the artifact closest to the work.
2. Decide whether it changes intended behavior, implementation strategy, task breakdown, or only code.
3. Update any other artifacts that now disagree with the accepted direction.
4. Run `/analyze` to check for gaps across spec/plan/tasks.
5. Continue implementation only once the artifact set describes the behavior and approach you want future contributors to trust.

*Analogy: editing a novel — realizing something in Chapter 10 makes you go back and fix a plot detail in Chapter 3. The insight flows backward through the document, not just forward.*

**Official caveat, worth repeating verbatim:** *"Flow-back is flexible, but it requires discipline. Do not leave a lower-level change in tasks.md or code if spec.md still says something different and the spec is meant to remain trustworthy."* — this is the formal name for the exact architectural-drift failure mode in Section 4.6. Flow-Back is the *intentional, disciplined version* of that scenario; uncontrolled drift is what happens when nobody does steps 3–4.

### All Three Models, Side by Side

| | Flow-Forward | Living Spec | Flow-Back |
|---|---|---|---|
| **Trigger** | Genuinely new capability/feature | A new *requirement* changes existing behavior | An *insight discovered mid-work* (planning, tasking, or coding) |
| **Starting point** | New `specs/00X-.../` directory | Always `spec.md` first | Wherever the insight landed — spec, plan, tasks, or code |
| **What happens to old artifacts** | Left untouched, permanent history | The existing `spec.md` is edited in place | Whichever artifacts now disagree get updated to match |
| **Direction of flow** | Forward only, new folder each time | Forward only, but re-run each time behavior changes | Can move **backward** — code/tasks → plan → spec |
| **Who initiates it** | Anyone adding new scope | A planned business/product decision | The engineer, discovering something nobody planned for |
| **Analogy** | A diary — each entry dated and permanent | A Wikipedia page — continuously edited to reflect current truth | Editing a novel — a Chapter 10 realization sends you back to fix Chapter 3 |
| **Validation step** | `/analyze` optional, less critical since nothing old changed | `/analyze` before implementation resumes | `/analyze` is essential — this is where drift risk is highest |

**The one-line test, in order:**
1. "Is this brand new capability?" → **Flow-Forward**
2. "Am I changing something that already has a spec, and I knew this before I started?" → **Living Spec**
3. "Did I only realize this *while* doing the work, and it should reshape something earlier?" → **Flow-Back**

> ⚠️ **Separate but related risk:** these three models govern your *feature artifacts* (`specs/`). They're distinct from updating Spec Kit's own *project files* (commands, scripts, templates) via `specify init --here --force`. That forced refresh can **overwrite your customized `constitution.md`** and anything under `.specify/templates/` or `.specify/scripts/` unless you back them up first — `specs/` itself isn't touched, but shared project files are. Never run the forced refresh without protecting your constitution first.

## 4.9 Cross-Cutting Architectural Changes

For a change that spans many existing files (e.g., introducing a `Result<T>` response pattern across every controller), don't assume the agent will infer the full scope on its own. Be explicit in your `/plan` prompt:
```
/plan
Scan all controllers, middleware, frontend API consumers, and tests in the
current codebase. Identify every location that needs to change to adopt the
Result<T> response pattern. Generate repository-wide tasks covering all of them.
```
This gets you real repository-wide impact analysis — but it's something you have to explicitly ask for, not something that happens automatically just because the codebase exists.

## 4.10 Preventing Specification Sprawl at Scale

As a project accumulates specs over months or years, individual Flow-Forward specs become harder to navigate as a *current* picture of the system — each one is accurate for its moment in time, but nobody wants to read 40 spec folders to understand "what does this system do today."

**Recommended approach:** maintain a periodically-regenerated **"Current Product" (or "System Overview") document**, separate in purpose from individual feature specs:

- **Individual specs stay untouched** as your audit trail — never edit old Flow-Forward specs to "keep them current." That defeats their purpose.
- **The Current Product doc is a navigation layer**, not a source of truth — regenerate it periodically (e.g., after every N merged features, or quarterly) using the same repo-scanning approach as the `project-discovery` plugin (see Section 4.2), rather than hand-maintaining it incrementally like a BMAD-style PRD.
- **Treat it as disposable and regeneratable.** If it drifts from reality, throw it away and regenerate from the current codebase + specs, rather than trying to patch it piecemeal.
- This avoids two failure modes at once: spec sprawl (nobody manually reconciling dozens of historical specs) and staleness (a hand-maintained doc nobody keeps updated).

**[VALIDATE ON PILOT]** — this recommendation is reasoned from Spec Kit's documented persistence model, not yet tested at real multi-year project scale. Revisit after the first pilot accumulates enough specs to test it for real.

---

# Part 5 — Working in Teams

No unnecessary process — scale governance to team size.

**Solo developer:**
- Self-review the spec before running `/plan`. Read it as if you were a stranger to the project.

**2–5 developers:**
- One developer prepares the spec.
- One teammate reviews it before `/plan` runs.

**Larger teams / cross-team changes:**
- Add an architecture review step before `/plan` for anything touching shared infrastructure or crossing team boundaries.
- Consider a shared `constitution.md` preset so all teams enforce the same standards — see Part 8.

---

# Part 6 — Best Practices and Common Mistakes

**✅ Best Practices**
- One feature per specification.
- Keep prompts focused and small — reference files with `#filename` instead of pasting content.
- Always review generated specs and plans before moving forward — don't treat AI output as final.
- Run `/clarify` and `/analyze` — they're cheap and catch problems early.
- Create `ProjectContext.md` once, maintain it after major changes.
- Do `/implement` layer by layer.
- Run `/converge` after implementation — don't assume "it compiled" means "it matches the spec."
- Periodically test constitution compliance directly: pick a rule, generate a small feature, and manually verify the generated code actually follows it (see Part 7 if it doesn't).

**❌ Common Mistakes**
- Running `/specify` directly on an existing project without `ProjectContext.md`.
- Jumping to `/implement` without checking that setup tasks (T001/T002) are addressed first.
- Combining multiple unrelated features into one specification.
- Copy-pasting entire files into chat instead of referencing them.
- Treating AI-generated specs or code as production-ready without review.
- Letting `plan.md`/`tasks.md` go stale after an out-of-process architectural change.
- Assuming a constitution rule stated as "MUST" guarantees compliant generated code — it doesn't; verify with `/converge`.

---

# Part 7 — Troubleshooting

Real errors hit while piloting this, with the actual fix — not hypothetical scenarios.

**"Could not execute because the specified command or file was not found" on `dotnet ef database update`**
The `dotnet-ef` CLI tool isn't installed globally.
```powershell
dotnet tool install --global dotnet-ef
```
If it says a version is already installed but not found, try `dotnet tool update --global dotnet-ef` instead, then close and reopen the terminal (PATH needs a refresh).

**"No migrations were applied. The database is already up to date." — but you never created a migration**
This message is misleading — it means there are *zero migrations to apply*, not that your schema is set up. Create the first one:
```powershell
dotnet ef migrations add InitialCreate
dotnet ef database update
```

**After `/implement`, there's no `.csproj` or `package.json` anywhere**
You skipped the setup tasks (T001/T002 — see Section 3.5). Check `tasks.md` for pending setup tasks and ask Copilot to execute those specifically before any feature-level work:
```
Execute only the setup tasks (T001, T002) from tasks.md — create the
backend project structure and frontend project structure first.
```

**Frontend runs but data doesn't persist / frontend seems disconnected from backend**
The frontend may still be using local in-memory state instead of calling the real API. Ask explicitly:
```
Wire the frontend to call the backend API (GET/POST/PUT/DELETE /api/{resource})
instead of local in-memory state. Update the API base URL to match the
backend's launch port, and add CORS support in Program.cs for the frontend origin.
```
Then restart both the backend (to pick up new CORS/controller changes) and the frontend.

**Generated code doesn't match a constitution rule you set**
This isn't a bug — see Part 6's last "common mistake." Constitution rules aren't automatically compiler-enforced, and vague wording gets interpreted narrowly (see the worked example in Section 3.1). Ask `/converge` (Section 3.8) to check compliance directly, and if it's wrong, ask for a targeted fix rather than regenerating the whole feature.

**Direct database calls showed up inside a controller, or endpoints aren't REST-conventional**
See the Section 3.1 worked example — this is almost always a constitution wording gap, not a bug. Tighten the specific rule (e.g., "no exceptions, including simple CRUD") and regenerate that layer only.

**`plan.md`/`tasks.md` describe an architecture that doesn't match the real code**
This is architectural drift — see the full recovery procedure in Section 4.6.

---

# Part 8 — Extensions and Presets

Spec Kit supports two customization mechanisms beyond the default commands — different problems, don't confuse them.

## 8.1 Extensions — Add New Capabilities

Extensions introduce entirely new commands/templates beyond what Spec Kit ships by default — they expand *what* Spec Kit can do.

**Examples:** Jira integration (specs/tasks sync with tickets), post-implementation code review as a pipeline step, V-Model test traceability, project health diagnostics.

```bash
specify extension search
specify extension add <extension-name>
```

**Think of it as:** installing a new plugin that gives you commands you didn't have before.

## 8.2 Presets — Customize Existing Workflows

Presets override the templates/commands that already ship with Spec Kit — they change *how* the existing process behaves, without adding new capabilities.

**Examples:** enforcing a compliance-oriented spec format, domain-specific terminology, organizational standards on plans/tasks, mandatory security review gates, test-first task ordering, localizing the workflow to a different language.

```bash
specify preset search
specify preset add <preset-name>
```

**Think of it as:** reskinning/rewiring the existing `/specify`, `/plan`, `/tasks` commands to match your org's specific rules.

## 8.3 The Core Distinction

| | Extensions | Presets |
|---|---|---|
| Adds new commands? | Yes | No |
| Changes existing template behavior? | No (adds alongside) | Yes (overrides in place) |
| Analogy | Installing a new app | Changing settings on an existing app |

**Resolution order:** Spec Kit resolves templates by walking the stack top-down, using the first match — a project-local override wins over a generic preset for the same template.

## 8.4 Why This Matters for Org-Wide Rollout

Rather than every team hand-writing their own `constitution.md` conventions from scratch, an EA-wide **preset** could enforce consistent spec/plan format standards (e.g., mandatory security review gates, standard compliance sections) across every pilot team — turning this handbook's guidance into something actually enforced, not just documented and hoped-for. Worth considering as a "phase 2" once your first real pilot validates the base workflow.

---

# Part 9 — Prompt Library

Ready-to-adapt prompts for common situations:

**Creating a new specification:**
```
/specify
[Feature description in plain language — what users can do, no tech detail]
```

**Extending an existing feature (brownfield):**
```
/specify
Use docs/ProjectContext.md as context.
Extend the existing [feature name] to support [new capability].
Constraint: [any backward-compatibility or non-breaking requirement].
```

**Architectural change with impact analysis:**
```
/specify
Use docs/ProjectContext.md as context.
Introduce [pattern/change] across [scope — all controllers / specific module].
Generate: functional requirements, migration strategy, impact analysis, acceptance criteria.
```

**Layer-by-layer implementation:**
```
/implement
Generate only: [specific file/layer]. Reference #tasks.md. Nothing else yet.
```

**Drift audit:**
```
Compare the current codebase structure against plan.md and data-model.md.
List every place they disagree.
```

**Constitution compliance check:**
```
/converge
Verify [specific file] against constitution.md's [specific principle].
Report every place they disagree.
```

---

# Part 10 — FAQ

**When should I use `/constitution`?**
Once per project, at the very start. Update it only when a standing rule genuinely changes project-wide — not per feature.

**Can I modify generated specs?**
Yes — and you should review every one before moving to `/plan`. Treat AI output as a draft, not a final answer.

**What if the specification is wrong?**
Edit `spec.md` directly, or re-run `/specify` with corrected input. Don't proceed to `/plan` on a spec you haven't reviewed.

**How do I use Spec Kit on an existing project?**
See Part 4. Create `docs/ProjectContext.md` first — don't run `/specify` cold on a brownfield codebase.

**How do I reduce GitHub Copilot token/credit usage?**
`specify init`, `/constitution`, `/specify`, `/clarify`, `/plan`, `/tasks`, and `/analyze` are all relatively cheap. `/implement` is the expensive one — do it layer by layer, reference files instead of pasting, and start new chat sessions when switching context (e.g., backend to frontend).

**Does Spec Kit automatically know about my other existing feature specs?**
More nuanced than a simple yes/no. The Copilot agent *can* inspect your existing codebase and discover prior implementation — for a cross-cutting change (e.g., "introduce `Result<T>` across all APIs"), you can explicitly tell `/plan` to scan all controllers, middleware, frontend consumers, and tests, and it will generate repository-wide tasks accordingly.
What it does **not** do by default is maintain a BMAD-style cumulative PRD that automatically loads and reconciles every historical feature spec. Commands operate primarily on the **active feature** (tied to the current git branch). `/analyze` validates consistency within that active feature's spec/plan/tasks; it doesn't reconcile against every other spec in the repo automatically.
**Practical rule:** for anything cross-cutting, explicitly instruct the agent to scan the relevant scope (controllers, tests, frontend consumers) in your `/plan` prompt — don't assume it will infer that scope on its own. See Section 4.8 for the three specification persistence models this affects.

**Is there a rollback mechanism if a generated architectural change is rejected?**
Not built into Spec Kit itself. Use standard git discipline — feature branches, no direct merge to main — as the practical safeguard.

**What's the difference between `/analyze` and `/converge`?**
`/analyze` checks whether spec/plan/tasks agree with each other, *before* code is written. `/converge` checks whether the *actual finished code* agrees with spec/plan/tasks, *after* implementation. Pre-flight vs. landing check.

**Does a constitution rule marked "MUST" guarantee the generated code follows it?**
No. Constitution rules are strong guidance the AI is supposed to follow, not a compiler-enforced constraint. Verify important rules directly with `/converge` rather than assuming compliance.

---

# Part 11 — Templates

## `docs/ProjectContext.md` template
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
- [ ] Dependencies and assumptions identified
- [ ] No [NEEDS CLARIFICATION] markers remain
```

## Feature Request Template
```markdown
## Feature Name

## Business Requirement (plain language, no tech detail)

## Existing Context (link to relevant part of ProjectContext.md)

## Constraints

## Success Criteria
```

---

# Part 12 — What Good Looks Like

By the end of this handbook, you should be able to:

✅ Explain Specification-Driven Development without naming a specific tool
✅ Run every Spec Kit command confidently, in the right order — including `/converge`, not just the five everyone remembers
✅ Explain what each artifact file actually contains and which other artifacts it depends on
✅ Know when to stop and create `ProjectContext.md` before touching an existing project
✅ Spot the setup-tasks mistake before it costs you debugging time
✅ Recognize architectural drift and know the recovery steps
✅ Diagnose and fix the common real errors in Part 7 without external help
✅ Know the difference between an extension and a preset, and when your org might want either
✅ Keep Copilot credit usage under control on a real project

---

*This handbook is a living document. If you hit something it doesn't cover, that's useful — bring it back so it can be added for the next person. Maintained by the EA Spec Kit SPOC.*
