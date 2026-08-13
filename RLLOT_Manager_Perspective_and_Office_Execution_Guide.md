# RLLOT Rollout Dashboard — Manager Perspective, SDD Mindset, and Office Execution Guide

**Date:** 2026-08-13  
**Purpose:** Consolidate the manager's expectations, the SDD mindset for the RLLOT rollout use case, and the exact next steps to execute on the office laptop using GitHub Copilot Chat (GHCP) and GitHub Spec Kit.

---

# 1. Executive Summary

The RLLOT exercise is **not simply a dashboard modernization task** and it is **not simply a Spec Kit command demo**.

The manager's expectation is broader:

1. Use the existing Rollout Dashboard as a real enterprise case to understand how **Spec-Driven Development (SDD)** should be applied.
2. Do not blindly reproduce the current dashboard.
3. Use the current application as **one source of business knowledge**, not as the target design.
4. Before committing to a solution, expand the idea from multiple perspectives: business/product, end user/manager, UX/information architecture, data/domain, security/access, operations, architecture capabilities, and future changeability.
5. Look for a **better way of representing and solving the rollout problem**, not merely a better implementation of the current screen.
6. Capture the **journey and time spent**, not only the final artifacts.
7. After the first implementation, deliberately introduce requirement changes and observe how SDD artifacts and code evolve.
8. Use the exercise to produce a **practical greenfield SDD guideline** for future teams.

The immediate next step is therefore **not Feature 001 implementation**.

> **Next step: Greenfield Idea Exploration / Multi-Perspective Discovery before the first Spec Kit feature specification.**

---

# 2. What the Manager Is Really Asking

The manager is not primarily asking:

```text
Can we rewrite Python + static HTML using React and .NET?
```

The deeper question is:

```text
Given a high-level application idea,
how should we think before we lock onto a solution,
and how can SDD / AI help us make that thinking explicit,
structured, testable, and adaptable to later change?
```

The RLLOT application is useful because it already works and therefore gives us evidence about the domain. But the exercise must also simulate how we would think when a future greenfield project starts with much less information — sometimes only a one-line requirement or one screenshot.

---

# 3. Key Manager Feedback

## 3.1 Capture the journey and time

The manager wants the **journey**, not only the final artifacts.

Capture:

```text
Initial high-level ask
        ↓
Understanding / discovery
        ↓
Questions raised
        ↓
Alternative ideas considered
        ↓
Decisions made
        ↓
Specification
        ↓
Clarification
        ↓
Planning
        ↓
Implementation
        ↓
Requirement changes
        ↓
Impact analysis / rework
        ↓
Final lessons
```

For each stage record:

- approximate time spent,
- input available at the beginning,
- what was learned,
- questions discovered,
- decisions made,
- what AI helped with,
- what required human/stakeholder judgment,
- what changed compared with the previous stage.

The purpose is not to claim all future applications take the same time. The purpose is to show a reusable **SDD journey pattern**.

## 3.2 Do not assume the existing solution is the best product

Avoid:

```text
Existing dashboard
        ↓
Understand current screens
        ↓
Build the same screens using new technology
```

Use:

```text
High-level rollout problem
        ↓
Understand business intent
        ↓
Use existing system as one reference
        ↓
Challenge assumptions
        ↓
Explore alternative representations
        ↓
Ask if there is a better product experience
        ↓
Select target direction
        ↓
Specify and implement
```

## 3.3 Avoid solution bias

Different specialists naturally think from their own expertise. That is useful, but it can also cause premature solution design.

The experiment should deliberately broaden the idea before choosing the solution.

## 3.4 Explore multiple viewpoints

Analyze from at least:

1. Business / Product Analyst
2. End User / Manager
3. UX / Information Architecture
4. Data / Domain
5. Security / Access
6. Operations
7. Architecture Capabilities
8. Future Evolution / Changeability

The goal is to expose hidden requirements, assumptions, conflicts, alternate product structures, edge cases, and future change risks.

## 3.5 Do not limit creativity to today's UI

The existing screenshots are **UX evidence**, not mandatory target UI.

The new product may keep much of the current information hierarchy, but only after we intentionally conclude it is still the best fit.

## 3.6 Requirement changes will be introduced later

The manager intentionally has additional requirements that are not being revealed now. That gives us a real SDD change-impact experiment.

Observe:

```text
Requirement change
        ↓
Which spec changes?
        ↓
Which plan decisions change?
        ↓
Which tasks change?
        ↓
Which code changes?
        ↓
What remains stable?
        ↓
How much effort/time?
        ↓
What did /analyze and /converge detect?
```

---

# 4. How to Describe Our Current RLLOT Work

The most accurate description is:

> **Brownfield-informed discovery leading to a greenfield reimplementation.**

We studied the existing application because it contains business knowledge, but the target implementation is not intended to evolve the old static HTML codebase in place.

```text
Existing RLLOT
    ↓
Brownfield discovery / evidence gathering
    ↓
Business intent and invariants
    ↓
Greenfield product exploration
    ↓
New specifications
    ↓
Greenfield implementation
```

---

# 5. What We Already Learned from the Current System

The current RLLOT references already reveal:

- six rollout initiatives,
- Phase 1 / Phase 2 / Phase 3 concepts,
- Customer Approval,
- overdue behavior,
- Productivity / Quality / Coverage metrics,
- Pre/Post GenAI values,
- Team / Project / Org / PM / DR context,
- cross-DR consolidation,
- current filters/search,
- team-level details,
- calculation formulas,
- Excel interpretation,
- import edge cases,
- operational pain points.

This is useful evidence. But before finalizing the new product we must ask:

> **If we were solving rollout-management from scratch, would we still organize the product the same way?**

---

# 6. Earlier Feature Roadmap — Now a Baseline Hypothesis

The previous roadmap remains useful but is **not yet committed**:

```text
001 Controlled Excel Import
002 Authentication / Authorization
003 Backend Unit Testing vertical slice
004 Generate Test Cases
005 Generate PostMan Scripts
006 Frontend Unit Testing
007 Integration Testing
008 Vulnerability Discovery
009 Overview / Global Filters
010 Production Readiness
011+ Deferred automation/features
```

Do not discard it. Do not execute it yet.

Greenfield exploration may confirm it, refine it, or reorganize it.

---

# 7. New Stage Before Spec Kit Feature Execution

## Stage 0 — Greenfield Idea Exploration

Purpose:

> Explore the problem broadly enough that the specification is not just a transcription of the current application.

Updated flow:

```text
High-level business ask
        ↓
Greenfield idea exploration
        ↓
Multiple perspectives
        ↓
Alternative product concepts
        ↓
Assumptions and open questions
        ↓
Compare with current RLLOT
        ↓
Stakeholder clarification
        ↓
Select target product direction
        ↓
Project constitution
        ↓
Feature specifications
        ↓
Plan / tasks / implementation
```

---

# 8. Office Laptop — Exact Working Structure

Create a clean greenfield repository, not inside the legacy project.

```text
RLLOT-Modernized/
│
├── docs/
│   ├── reference/
│   └── sdd-journey/
│
└── [Spec Kit project files later]
```

Recommended reference files:

```text
docs/reference/
    RLLOT_MASTER_CONTEXT.md
    RLLOT_Dashboard_SDD.md
    SPECKIT_extract_data.md
    SPECKIT_build_rllot_dashboard.md
    RLLOT_Modernization_Discovery_Corrected.md
    RLLOT_Architecture_Decision_Record.md
    RLLOT_SDD_Mindset_Manager_Articulation.md
```

These are **reference/evidence**, not the new specification.

---

# 9. Start the SDD Journey Log First

Create:

```text
docs/sdd-journey/
    00-initial-ask.md
    01-greenfield-exploration.md
    02-alternative-product-concepts.md
    03-current-system-comparison.md
    04-stakeholder-clarifications.md
    05-selected-product-direction.md
    journey-log.md
```

Use this template in `journey-log.md`:

```markdown
# RLLOT SDD Journey Log

## Entry

**Date:**
**Stage:**
**Start time:**
**End time:**
**Approximate effort:**

### Input available
What information did we have before starting?

### Activity
What did we do?

### AI/tool used
Example: GHCP normal chat, Spec Kit command, manual analysis.

### Questions discovered
What became unclear?

### Assumptions found
What were we assuming without evidence?

### Decisions made
What was decided and by whom?

### Artifact/output
What was created?

### What AI helped with
What did AI accelerate or expose?

### What required human judgment
What could AI not legitimately decide?

### What changed from the previous understanding
Did our product/architecture direction change?

### Lesson for future SDD guideline
What should another project/team learn from this?
```

Start recording time immediately.

---

# 10. First GHCP Activity — Normal Chat, NOT `/speckit.specify`

Use this prompt:

```text
I want to explore a greenfield product idea before writing a specification.

Problem:
We need an internal Production Rollout application for monitoring GenAI SDLC adoption/progress across teams and initiatives.

Important constraints for this exercise:

- Treat this first as a product/problem-discovery exercise.
- Do not choose a technology stack.
- Do not create code.
- Do not create a technical implementation plan.
- Do not assume the current dashboard layout is the best representation.
- Do not simply recreate a dashboard from existing screenshots.
- Challenge assumptions and identify missing information.
- Treat any existing application only as later reference evidence, not as the target design.

Analyze the problem independently from the following perspectives.

1. Business / Product Analyst
   - What business problem are we solving?
   - What decisions should users be able to make using the application?
   - What business outcomes should improve?
   - What would make the product valuable rather than just a reporting screen?

2. End User / Manager
   - What information would leadership, PMs, rollout owners, and delivery managers actually need?
   - What situations require action?
   - What information should be immediately visible?
   - What drill-downs would users need?

3. UX / Information Architecture
   - What are meaningfully different ways of structuring the product?
   - Could it be overview-first, exception-first, initiative-first, team-first, trend-first, or another model?
   - What would reduce cognitive load and improve decision making?
   - Do not limit the UX to the current application's layout.

4. Data / Domain
   - Identify the major business/domain concepts.
   - Identify important relationships.
   - What historical/trend information might be useful?
   - What information would be needed to calculate rollout health?
   - Which concepts should be configurable rather than hardcoded?

5. Security / Access
   - What kinds of access boundaries might exist?
   - Could visibility vary by organization, DR, manager, project, team, or role?
   - What must be clarified before designing authorization?

6. Operations
   - How might data enter the system?
   - Who owns source data?
   - How fresh must it be?
   - How should incomplete or invalid data be handled?
   - What operational visibility is needed?

7. Architecture Capabilities
   - Do not choose technologies.
   - Identify capabilities the solution may require, such as dynamic data retrieval, authorization, auditability, history, configurability, data validation, extensibility, or integration boundaries.

8. Future Evolution
   - What requirements are likely to change?
   - What should not be hardcoded too early?
   - What future capabilities may reasonably appear?
   - Where could today's assumptions cause future rework?

After the eight perspectives, produce:

A. Problem statement.
B. User/actor map.
C. Business outcomes.
D. User decision/use-case map.
E. Business invariants that appear likely.
F. Assumptions.
G. Stakeholder questions.
H. Potential edge cases.
I. Candidate capabilities.
J. Current ideas that may be implementation/UX assumptions rather than real requirements.
K. Three meaningfully different product concepts for solving the problem.

The three product concepts must differ in information architecture and user workflow, not merely colors, card layouts, or technology.

Examples of genuinely different conceptual directions could include:
- initiative-centric tracking,
- exception/action-first management,
- rollout-health / risk / trend intelligence.

Do not select the winner yet.
Do not produce a Spec Kit specification yet.
```

Save output to:

```text
docs/sdd-journey/01-greenfield-exploration.md
```

Record time and lessons.

---

# 11. Second GHCP Activity — Critically Review the Product Concepts

```text
Review the three product concepts you proposed.

Act as a critical product review panel.

For each concept evaluate:

1. Primary user value.
2. Which management decisions it supports best.
3. What information it makes easy to understand.
4. What information becomes harder to understand.
5. Risks and assumptions.
6. Complexity introduced.
7. Dependence on historical/trend data.
8. Dependence on access-control sophistication.
9. How well it supports future requirement change.
10. Whether it solves the actual rollout-management problem or merely creates attractive reporting.

Then identify:

- capabilities common to all concepts,
- capabilities unique to each concept,
- requirements that must be clarified before choosing,
- things that should remain configurable,
- things that would be dangerous to hardcode.

Do not choose technology.
Do not create code.
Do not create the final specification.
```

Save to:

```text
docs/sdd-journey/02-alternative-product-concepts.md
```

Record time.

---

# 12. Third GHCP Activity — Bring in Existing RLLOT Reference

Now use the old application/reference docs.

```text
Now compare the independent greenfield product concepts with the current RLLOT reference material in docs/reference.

Important:

- The existing RLLOT implementation is evidence of current behavior, not automatically the target design.
- Preserve genuine business rules and important user outcomes.
- Do not preserve legacy technical constraints merely because they exist.
- Do not assume the current UX is the best possible UX.

Classify each important current RLLOT capability or behavior into:

1. Business invariant / approved rule — must be preserved unless explicitly changed.
2. Valuable current capability — likely worth retaining.
3. Current UX / information-architecture choice — may be redesigned.
4. Legacy technical constraint — should not drive the new solution.
5. Existing behavior that appears ambiguous and requires stakeholder confirmation.
6. Potential business capability discovered in greenfield exploration that the current system does not provide.

Then:

- identify conflicts between the greenfield concepts and current behavior,
- identify any current capabilities missing from the greenfield analysis,
- identify new opportunities the existing system may have hidden,
- propose an updated shortlist of target product directions,
- list the stakeholder decisions required before selecting the target direction.

Do not write code.
Do not create the implementation plan.
Do not silently treat legacy behavior as a requirement.
```

Save to:

```text
docs/sdd-journey/03-current-system-comparison.md
```

Record time.

---

# 13. Reduce Questions to Product-Shaping Clarifications

Do not take 30–50 AI-generated questions to the manager.

Reduce to about **5–10 decisions that materially change the product**.

Suggested RLLOT questions:

1. Is the primary purpose reporting progress, identifying actions/exceptions, management risk/health visibility, or a combination?
2. Should the landing experience be overview-first, exception-first, or rollout-health/risk-first?
3. Is historical trend visibility important for V1?
4. Are the six initiatives fixed, or should initiatives become configurable?
5. Do all internal viewers see the same information, or will visibility vary by DR/manager/org/project/team?
6. Is Excel expected to remain the long-term business input or only an initial source?
7. Which current calculations are approved business rules versus current conventions?
8. Should the app only show rollout information, or also surface recommended attention areas/actions?
9. What does success look like 6–12 months after launch?
10. Which current dashboard capabilities could be removed if a better user workflow replaces them?

Capture answers in:

```text
docs/sdd-journey/04-stakeholder-clarifications.md
```

---

# 14. Select and Document the Target Product Direction

Create:

```text
docs/sdd-journey/05-selected-product-direction.md
```

It should capture:

```text
Problem we are solving
Users
Decisions the application supports
V1 outcomes
Chosen product concept
Why it was selected
Capabilities required
Capabilities deferred
Business invariants
UX principles
Important open questions
Constraints
What from the current app will be preserved
What will deliberately change
```

Only after this should we finalize the Spec Kit feature roadmap.

---

# 15. Then Verify / Initialize Spec Kit

Verify installation:

```powershell
specify version
```

Optional read-only update check:

```powershell
specify self check
```

Initialize in the clean repository if not already initialized:

```powershell
specify init --here --integration copilot --script ps
```

Do not reinitialize for every feature.

---

# 16. Current Official Spec Kit Flow

Use:

```text
/speckit.constitution
        ↓
/speckit.specify
        ↓
/speckit.clarify
        ↓
/speckit.plan
        ↓
/speckit.checklist
        ↓
/speckit.tasks
        ↓
/speckit.analyze
        ↓
/speckit.implement
        ↓
/speckit.converge
```

Interpretation:

```text
constitution  → project-wide governing principles
specify       → WHAT and WHY
clarify       → resolve underspecified requirements
plan          → HOW / technical implementation
checklist     → requirement-quality gate
tasks         → dependency-ordered work
analyze       → spec/plan/tasks consistency
implement     → execute tasks
converge      → check code against spec/plan/tasks and append missing work
```

For large features, implement in stages rather than forcing everything into one agent context.

---

# 17. Spec Kit Does Not Replace Product Exploration

For RLLOT:

```text
PRE-SPEC-KIT
Greenfield exploration
Multiple perspectives
Alternative product concepts
Current-system comparison
Stakeholder clarification
Selected product direction

THEN SPEC KIT
Constitution
Specify
Clarify
Plan
Checklist
Tasks
Analyze
Implement
Converge
```

This is the most important process correction after the manager's latest feedback.

---

# 18. Constitution — Updated Interpretation

Likely principles still include:

- approved business behavior preservation,
- specification before implementation,
- dynamic data access,
- server-side security,
- ingestion compatibility,
- single calculation authority,
- separation of concerns,
- database portability,
- testable business rules,
- operational safety,
- untrusted source-data handling,
- zero incremental cost,
- incremental vertical delivery,
- explicit ambiguity.

Important correction:

> Do not make **"preserve the current dashboard information hierarchy"** a strong constitutional rule before greenfield UX exploration is complete.

The current UX is evidence, not law.

---

# 19. What Happens to the Earlier Feature Roadmap

Possible outcomes after exploration:

## A. Initiative-centric model is confirmed

Earlier roadmap remains mostly valid.

## B. Exception-first model is selected

Feature decomposition may instead prioritize:

```text
Data foundation
Access boundary
Rollout health / exception model
Attention dashboard
Team / initiative drilldown
Metrics / trends
```

## C. Rollout-health / trend intelligence is selected

Feature decomposition may prioritize:

```text
Data/history model
Rollout health rules
Risk/trend summaries
Management overview
Exception analysis
Initiative/team drilldowns
```

Therefore, do not run the old Feature 001–010 sequence yet.

---

# 20. Requirement Change Experiment

After the first accepted implementation baseline, introduce deliberate requirement changes.

For every change capture:

```text
Original intent/spec
        ↓
New requirement
        ↓
Spec impact
        ↓
Plan impact
        ↓
Task impact
        ↓
Code impact
        ↓
Tests changed
        ↓
Analyze findings
        ↓
Implementation
        ↓
Converge findings
        ↓
Time/effort
        ↓
Lesson learned
```

---

# 21. How to Handle Evolving Specs

Two useful patterns:

## Flow-forward

Use a new feature specification for a substantial follow-on capability. Keep the older feature directory as project history.

## Living spec

Use when the intended behavior of an existing feature changes.

```text
Update spec first
        ↓
Update/regenerate plan
        ↓
Update/regenerate tasks
        ↓
/speckit.analyze
        ↓
/speckit.implement
        ↓
/speckit.converge
```

Record which pattern was chosen and why.

---

# 22. What to Measure

## Time

- idea exploration,
- stakeholder clarification,
- constitution,
- specify,
- clarify,
- plan,
- tasks,
- implementation,
- convergence,
- requirement-change cycle.

## Quality

- assumptions exposed,
- ambiguities clarified,
- business-rule defects caught before coding,
- issues caught by `/speckit.analyze`,
- gaps caught by `/speckit.converge`,
- parity defects,
- requirement-change rework.

## Human vs AI contribution

AI can accelerate:

- possibility scanning,
- multi-perspective analysis,
- structured requirement drafting,
- inconsistency detection,
- alternative proposals,
- artifact generation.

Human/stakeholder judgment remains necessary for:

- business intent,
- formula approval,
- product direction,
- access scope,
- organizational constraints,
- trade-offs,
- final acceptance.

---

# 23. Short Update to Give the Manager

> I understood the additional point from our discussion. I should not take the existing Rollout Dashboard and simply rebuild the same product using a new stack. I will use the existing application only as one source of business evidence. Before the first Spec Kit feature, I am adding a greenfield exploration stage where I will start from the high-level rollout problem, analyze it from business, user, UX, data, security, operations, architecture and future-change perspectives, and produce genuinely different product concepts. Then I will compare those concepts with the existing dashboard, bring only the product-shaping ambiguities back for clarification, select the target direction, and only then enter the Spec Kit constitution/specification flow. I will also log the time, decisions, AI contribution, human clarifications, and later requirement-change impact throughout the journey.

---

# 24. What NOT to Do Tomorrow

Do not:

```text
open old repo
→ ask Copilot to generate new spec
→ accept it
→ start coding
```

Do not:

```text
copy current screenshot
→ ask Copilot to recreate it in React
```

Do not run all Spec Kit features at once.

Do not let Copilot decide unresolved business rules.

Do not treat every current behavior as mandatory.

Do not choose architecture only because it matches personal technical expertise.

Do not start SharePoint automation before product direction is settled.

---

# 25. Exact Office-Laptop Checklist

```text
[ ] Create/open clean RLLOT-Modernized repo
[ ] Copy reference documents into docs/reference
[ ] Create docs/sdd-journey
[ ] Create journey-log.md
[ ] Record start time
[ ] Open normal GHCP Chat
[ ] Run Greenfield Multi-Perspective Exploration prompt
[ ] Save result
[ ] Record time + lessons
[ ] Run Product Concepts Critical Review prompt
[ ] Save result
[ ] Record time + lessons
[ ] Bring in current RLLOT reference
[ ] Run Current-System Comparison prompt
[ ] Save result
[ ] Record time + lessons
[ ] Reduce questions to 5–10 product-shaping stakeholder decisions
[ ] Review with manager/product owner
[ ] Capture answers
[ ] Select/document target product direction
[ ] Only then verify/init Spec Kit
[ ] Run constitution
[ ] Start one bounded feature
```

---

# 26. Final SDD Mindset Statement

> **SDD is not "write a prompt and generate a spec." It is making intent explicit before implementation, surfacing ambiguity instead of hiding it, exploring alternative interpretations before committing to a solution, using specifications as the source of truth for implementation, and maintaining traceability when requirements evolve.**

For RLLOT:

> **We use the old application to learn from the past, but we do not allow the past implementation to define the future product.**

---

# 27. Web-Verified GitHub Spec Kit Notes — 2026-08-13

Current official GitHub Spec Kit documentation confirms:

- `specify init --here --integration copilot` is supported for current-directory initialization.
- `--script ps` is supported for PowerShell scripts on Windows.
- `specify version` verifies the installed CLI.
- `specify self check` can check for a newer release without modifying the installation.
- `/speckit.constitution` establishes project-wide governing principles.
- `/speckit.specify` focuses on **what to build and why**, not the technology stack.
- `/speckit.clarify` is the structured workflow for underspecified requirements and is recommended before `/speckit.plan`.
- `/speckit.plan` introduces the technical implementation approach.
- `/speckit.checklist`, `/speckit.tasks`, `/speckit.analyze`, `/speckit.implement`, and `/speckit.converge` form the quality/execution stages.
- `/speckit.converge` compares implementation with `spec.md`, `plan.md`, and `tasks.md` and appends missing work when gaps remain.
- For large features, staged `/speckit.implement` runs are supported.
- Official evolving-spec guidance supports both flow-forward feature specs and living-spec updates.

Official references checked:

- GitHub Spec Kit README
- GitHub Spec Kit Quickstart
- GitHub Spec Kit Core CLI Reference
- GitHub Spec Kit Agentic SDD Reference
- GitHub Spec Kit Evolving Specs Guide
- GitHub Spec Kit Installation Guide

---

**End of consolidated report**
