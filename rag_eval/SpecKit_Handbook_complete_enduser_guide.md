# Specification-Driven Development with Spec Kit
### A Practical Handbook — From Zero to Your First Enterprise Feature

**Maintained by:** Mahantesh, EA GenAI Team (SPOC for Spec Kit)
**Audience:** Developers, Tech Leads, Architects, New Joiners
**Style note:** This isn't a rewrite of GitHub's docs. GitHub's docs teach you *how to run commands*. This handbook teaches you *how to use Spec Kit successfully on a real project* — including the parts nobody documents, like what to do with an existing enterprise codebase.

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

## 2.4 Understanding the Generated Files

Once you run through a feature, you'll see:

| File | What it actually is |
|---|---|
| `constitution.md` | Your project's standing rules — architecture style, testing policy, layering |
| `spec.md` | The feature's requirements, in plain language, no tech detail |
| `requirements.md` (checklist) | A quality gate — checks the spec is complete and unambiguous *before* planning starts |
| `research.md` | Your architecture decision record — what was chosen, why, and what alternatives were rejected |
| `data-model.md` | Entities, fields, relationships |
| `plan.md` | The overall technical design |
| `tasks.md` | Ordered, numbered implementation steps |
| `quickstart.md` | A validation runbook — proves the finished feature matches the original spec |

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

**Output:** `plan.md`, `research.md` (decision + rationale + alternatives considered), `data-model.md` (entity design).

**Common mistake:** accepting the plan without reading `research.md` — this is where you catch a bad architectural choice before any code gets written, which is far cheaper than catching it after `/implement`.

## 3.5 `/tasks`

**Why it exists:** Breaks the plan into ordered, numbered, trackable steps (T001, T002...) that `/implement` executes against.

**Critical thing to check every time:** the first 1–2 tasks are almost always **project setup** (create the backend project shell, create the frontend project shell). Skipping straight to feature-level implementation without these produces code fragments with nowhere to run — this is the single most common failure newcomers hit.

## 3.6 `/analyze`

**Why it exists:** Runs after `/tasks`, before `/implement`. Cross-checks that spec, plan, and tasks actually agree with each other — catching disagreement before code gets written rather than after.

**When to run it:** Every time, right before your first `/implement` call on a feature. It's cheap and it's your first line of defense against drift.

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
/plan → /tasks → /implement
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

**❌ Common Mistakes**
- Running `/specify` directly on an existing project without `ProjectContext.md`.
- Jumping to `/implement` without checking that setup tasks (T001/T002) are addressed first.
- Combining multiple unrelated features into one specification.
- Copy-pasting entire files into chat instead of referencing them.
- Treating AI-generated specs or code as production-ready without review.
- Letting `plan.md`/`tasks.md` go stale after an out-of-process architectural change.

---

# Part 7 — Prompt Library

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

---

# Part 8 — FAQ

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
Not reliably — cross-feature awareness generally requires you to explicitly reference the earlier spec folder in your prompt. Don't assume it's automatic.

**Is there a rollback mechanism if a generated architectural change is rejected?**
Not built into Spec Kit itself. Use standard git discipline — feature branches, no direct merge to main — as the practical safeguard.

---

# Part 9 — Templates

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

# Part 10 — What Good Looks Like

By the end of this handbook, you should be able to:

✅ Explain Specification-Driven Development without naming a specific tool
✅ Run every Spec Kit command confidently, in the right order
✅ Know when to stop and create `ProjectContext.md` before touching an existing project
✅ Spot the setup-tasks mistake before it costs you debugging time
✅ Recognize architectural drift and know the recovery steps
✅ Keep Copilot credit usage under control on a real project

---

*This handbook is a living document. If you hit something it doesn't cover, that's useful — bring it back so it can be added for the next person. Maintained by the EA Spec Kit SPOC.*
