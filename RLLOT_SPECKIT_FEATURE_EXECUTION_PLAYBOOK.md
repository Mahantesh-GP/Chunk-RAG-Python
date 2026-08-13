# RLLOT Modernization — SDD / GitHub Spec Kit Feature Execution Playbook

**Status:** Execution roadmap for greenfield SDD implementation  
**Project:** RLLOT Production Rollout Dashboard Modernization  
**Purpose:** Use this document on the office laptop while executing GitHub Spec Kit with GitHub Copilot Chat (GHCP).  
**Execution rule:** Work **one feature at a time**. Do not run all feature prompts together.

---

# 1. Why this roadmap exists

The goal is not to ask Copilot to read the old project and generate a replacement application.

The SDD approach for RLLOT is:

```text
Understand the problem
        ↓
Capture business invariants
        ↓
Separate legacy behavior from legacy implementation
        ↓
Make constraints and unknowns explicit
        ↓
Validate risky architecture assumptions
        ↓
Establish project constitution
        ↓
Specify one bounded feature
        ↓
Clarify ambiguity
        ↓
Plan the implementation
        ↓
Generate dependency-ordered tasks
        ↓
Analyze artifact consistency
        ↓
Implement
        ↓
Converge
        ↓
Verify / accept feature
        ↓
Move to the next feature
```

The current RLLOT codebase is **evidence/reference**, not the new specification.

---

# 2. Reference material to keep in the greenfield repository

Recommended location:

```text
docs/
└── reference/
    ├── RLLOT_MASTER_CONTEXT.md
    ├── RLLOT_Dashboard_SDD.md
    ├── SPECKIT_extract_data.md
    ├── SPECKIT_build_rllot_dashboard.md
    ├── RLLOT_Modernization_Discovery_Corrected.md
    ├── RLLOT_Architecture_Decision_Record.md
    └── RLLOT_SDD_Mindset_Manager_Articulation.md
```

These files have different purposes:

```text
RLLOT_MASTER_CONTEXT.md
    → consolidated project decisions and working context

RLLOT_Dashboard_SDD.md
    → legacy/current system behavior

SPECKIT_extract_data.md
    → exact Excel extraction and normalization reference

SPECKIT_build_rllot_dashboard.md
    → exact dashboard formulas, aggregation, filters and detail behavior

RLLOT_Modernization_Discovery_Corrected.md
    → problem, constraints, scope, unknowns and feasibility checks

RLLOT_Architecture_Decision_Record.md
    → proposed architecture and decision gates

Current dashboard screenshots
    → UX / navigation / information hierarchy reference
```

Do **not** tell Copilot to treat these files as the new spec. They are reference evidence used to prepare and verify new specs.

---

# 3. Office laptop pre-flight

Use a **new greenfield repository/folder**, not the old dashboard repository.

First inspect the installed Spec Kit version/integration:

```powershell
specify version
specify check
specify integration list
```

Initialize Spec Kit once in the project folder if it has not already been initialized:

```powershell
specify init --here --integration copilot --script ps
```

Equivalent current-directory form:

```powershell
specify init . --integration copilot --script ps
```

Do not repeatedly run `specify init` for each feature.

After initialization, confirm the `/speckit.*` commands are visible in GitHub Copilot Chat.

---

# 4. One-time project constitution

Run this once before feature specifications.

## Ready-to-paste constitution prompt

```text
/speckit.constitution

Establish the governing principles for the RLLOT Production Rollout Dashboard modernization.

Context:
RLLOT is an internal Production Rollout Dashboard currently generated from multiple DR-maintained Excel workbooks through Python extraction, JSON consolidation, and one self-contained static HTML dashboard. The modernization is a greenfield technical rebuild. The existing application is a reference for approved business behavior, source-data interpretation, calculations, edge cases, and useful user workflows, but its static HTML architecture must not constrain the new design.

Create a constitution containing these non-negotiable principles:

1. Business Behavior Preservation
Approved rollout calculations, status normalization, customer approval behavior, overdue rules, metric interpretation, filter semantics, and core user workflows must remain behaviorally equivalent unless an intentional business change is explicitly documented and approved.

2. Specification Before Implementation
Requirements must describe users, business outcomes, behavior, constraints, edge cases, and acceptance conditions before implementation details are introduced. Existing source code is evidence/reference and must never automatically become the new specification.

3. Dynamic Data Access
The modernized dashboard must not require the browser to receive the entire historical/current dataset when only a subset is needed. Data access must support server-side filtering, search, sorting, pagination, aggregation, and detail retrieval.

4. Server-Side Security
Client-side filtering, hidden rows, or UI state must never be treated as authorization. Any required access restrictions must be enforced before restricted data is returned to the client.

5. Ingestion Compatibility
The new ingestion capability must preserve approved Excel interpretation and normalization behavior through automated compatibility tests while allowing fragile legacy implementation details, such as fixed positional column indexes, to be corrected.

6. Single Calculation Authority
Completion, overdue, metric classification, metric aggregation, reporting-count, filtered-summary, and related business calculations must have one authoritative server-side/domain implementation. The frontend must not maintain an independent copy of business formulas.

7. Separation of Concerns
Source ingestion, schema validation, persistence, domain/business rules, API/application behavior, authentication/authorization, and UI concerns must remain independently testable and loosely coupled.

8. Database Portability
The business/domain layer must not depend on a specific relational database provider. The V1 provider may be selected during technical planning, but unnecessary provider-specific SQL/functions or data-model assumptions must be avoided.

9. Testable Business Rules
Critical calculations, normalization logic, ingestion behavior, filter behavior, and known edge cases must have automated tests using approved representative/reference datasets and expected results.

10. Operational Safety
Invalid, incomplete, or failed imports must never silently replace the last-known-good active dataset. Import failures and validation problems must be visible and diagnosable.

11. Untrusted Source Data
All Excel-originated text/data must be treated as untrusted input. Validate input, encode output safely, and do not introduce unsafe raw HTML rendering from imported values.

12. Zero Incremental Cost
The first production baseline must reuse approved existing organizational infrastructure wherever feasible and must not introduce new paid platform/service dependencies unless the zero-cost approach is explicitly proven insufficient and the deviation is approved.

13. Incremental Vertical Delivery
Implementation must proceed through bounded, independently verifiable features. A feature is not complete merely because code was generated; it must satisfy its specification, plan, tests, and convergence checks before the next dependent feature begins.

14. UX Continuity Without Pixel Lock-In
The current dashboard's information hierarchy and primary user workflows are the functional UX baseline. Visual/accessibility/responsive improvements are allowed; exact HTML, CSS, colors, SVG mechanics, and pixel-level layout are not architectural requirements.

15. Explicit Ambiguity and Intentional Change
Unknowns, legacy inconsistencies, and unresolved business rules must be surfaced through clarification rather than guessed. Any intentional deviation from current approved behavior must be documented with rationale and acceptance impact.

Governance must explain:
- how constitution violations are identified and resolved,
- how an intentional exception is documented,
- how architecture decisions trace to requirements/constraints,
- how changes to governing principles are reviewed,
- and that later specs/plans/tasks must be checked against this constitution.
```

## Constitution review gate

Before proceeding, manually verify that the generated constitution did **not** accidentally:

- turn React/.NET/SQLite into business requirements,
- require the old static HTML architecture,
- weaken server-side security,
- omit business-rule parity,
- omit atomic/last-known-good import safety,
- allow business formulas to be duplicated in the frontend,
- interpret “zero cost” as “use anything free on the internet” rather than zero **incremental** organizational cost.

Correct the constitution before Feature 001 if needed.

---

# 5. Architecture / feasibility gates

The specification phase can start while some technical questions are still open, but do **not** lock the implementation plan around unverified assumptions.

The following must be known before the relevant `/speckit.plan` is accepted:

| Gate | Question | Baseline | Fallback |
|---|---|---|---|
| IIS | Can target/representative IIS host supported ASP.NET Core? | Existing IIS | Other already-approved zero-incremental-cost internal host |
| Runtime | Which supported .NET runtime/hosting components are approved? | Validated environment | Approved available runtime |
| Authentication | Is seamless corporate/Windows Authentication available if identity is needed? | Windows/IIS integrated identity | Approved corporate alternative / same-data-for-all model |
| Persistence | Is application-owned SQLite permitted with controlled backup/permissions? | SQLite V1 | Existing zero-cost SQL Server if available/required |
| Ingestion | Can representative Excel be parsed with behavioral parity? | Prefer single-stack implementation if proven | Retain/refactor Python ingestion if lower-risk |
| Query | Can filtered/paged API return only requested data? | Dynamic server-side query | Adjust schema/query plan |
| Business rule | Are known rule ambiguities resolved? | Approved behavior | Explicitly defer affected behavior |

Do not ask Spec Kit to “decide” organizational policy questions.

---

# 6. Mandatory clarification register — do not silently guess these

These items were identified from the current implementation/reference material. Resolve them in the relevant `/speckit.clarify` cycle.

## Data/import ambiguities

1. **Missing target sheet**
   - Legacy: warn and skip.
   - Modernization question: reject whole import, reject workbook, or accept partial import with warning?

2. **Unreadable one workbook in a multi-file batch**
   - Legacy: warn and continue.
   - Modernization question: should any unreadable DR workbook prevent activation of the new dataset?

3. **Header row not found**
   - Legacy: sheet silently produces no rows.
   - Modernization direction: must not fail silently.
   - Decide whether this is blocking validation failure.

4. **Unknown status strings**
   - Legacy: preserve as-is.
   - Unknown statuses are not part of the standard completion denominator.
   - Decide whether new imports may preserve them with warning or must reject them.

5. **Invalid/non-date date cells**
   - Legacy effectively treats non-datetime values as missing.
   - Decide warning vs blocking validation.

6. **Formula-backed Excel cells**
   - Legacy reads cached calculated values.
   - New ingestion must define equivalent behavior if cached formula values are absent/stale.

7. **DR identity**
   - Legacy derives DR from filename using the existing naming convention/fallback.
   - Confirm this remains V1 behavior or whether DR metadata becomes explicit.

8. **Duplicate records**
   - Legacy concatenates rows across DR files and does not deduplicate.
   - Define whether duplicates are valid, warnings, or errors.

9. **Row identity across daily imports**
   - Current static output has no durable application identity.
   - New relational model needs a stable way to identify detail records while avoiding accidental identity collisions.

10. **Historical retention**
    - Define whether the app retains only active + last-known-good dataset, multiple import snapshots, or a specific retention period.

11. **Atomic activation**
    - Confirm that partial/bad imports never replace the current active dataset.

## Calculation/UI ambiguities

12. **Pre GenAI = 0**
    - Current builder returns no improvement.
    - Separate business reference describes a sprint-to-sprint proxy-baseline rule.
    - Must be explicitly approved before implementing the final metric calculation.

13. **Zero active rows: `0%` vs `N/A`**
    - Current calculation helper returns 0 when active denominator is zero.
    - Current rendering/test references also contain an N/A state.
    - Decide the required user-visible outcome. Do not guess.

14. **Customer Approval filter**
    - Current data/table includes Customer Approval.
    - Legacy JavaScript references customer-approval cascading behavior while captured UI mainly shows phase/overdue/search controls.
    - Clarify whether a dedicated Customer Approval filter is required in the new UX.

15. **Overview “Teams with Due Phase(s)”**
    - Row-level overdue logic also considers Customer Approval prerequisite date.
    - Clarify whether the Overview “Due Phase(s)” KPI should count rows overdue only in P1/P2/P3 or also Customer Approval prerequisite overdue.

16. **Exact access scope**
    - All internal viewers see all data?
    - DR-specific?
    - manager-specific?
    - organization/project/team-specific?
    - This must be decided before identity-aware authorization rules are finalized.

17. **No-login wording**
    - Requirement should mean “no additional interactive login” if seamless corporate identity is used.
    - Do not silently interpret this as anonymous access when differentiated authorization is needed.

18. **Refresh expectation**
    - Define acceptable delay from successful import to visible dashboard data.

These ambiguities are deliberately carried forward rather than hidden.

---

# 7. Feature dependency roadmap

This is the recommended Spec Kit execution order.

```text
ONE-TIME
Constitution
    |
    v
FEATURE 001
Controlled Excel Import, Validation & Active Dataset
    |
    v
FEATURE 002
Seamless Internal Access & Authorization Boundary
    |
    v
FEATURE 003
Backend Unit Testing — First Dynamic Vertical Slice
    |
    +------------------------------+
    | shared initiative pattern    |
    v                              |
FEATURE 004                        |
Generate Test Cases Initiative     |
    |                              |
    v                              |
FEATURE 005                        |
Generate PostMan Scripts           |
    |                              |
    v                              |
FEATURE 006                        |
Frontend Unit Testing              |
    |                              |
    v                              |
FEATURE 007                        |
Integration Testing                |
    |                              |
    v                              |
FEATURE 008                        |
Vulnerability Discovery            |
    |                              |
    +------------------------------+
    |
    v
FEATURE 009
Cross-Initiative Overview & Global Filters
    |
    v
FEATURE 010
Production Readiness, Parity, Cutover & Rollback

DEFERRED AFTER CORE MODERNIZATION
    |
    +--> FEATURE 011 SharePoint File Retrieval Automation
    |       |
    |       v
    |    FEATURE 012 Scheduled Ingestion
    |
    +--> FEATURE 013 Saved Views / Filter Profiles
    |
    +--> FEATURE 014 ADO / Work-Item Drill-Through
```

### Why this order

- Feature 001 establishes trustworthy normalized data before dashboard functionality.
- Feature 002 establishes the access/security boundary before broad user-facing expansion.
- Feature 003 proves the complete architecture and user workflow using one initiative.
- Features 004–008 deliberately reuse the Feature 003 pattern rather than inventing six implementations.
- Feature 009 is built only after all six initiatives exist, because Overview aggregates and global filters span all initiatives.
- Feature 010 validates operational/parity/cutover concerns only after the full V1 behavior exists.
- SharePoint automation is deliberately postponed so tenant/identity approval cannot block the primary modernization.

---

# 8. Standard execution cycle for every feature

For each feature, use this sequence and stop between stages to review artifacts.

```text
/speckit.specify
        ↓
MANUAL REVIEW OF spec.md
        ↓
/speckit.clarify
        ↓
ANSWER / RESOLVE QUESTIONS
        ↓
REVIEW spec.md AGAIN
        ↓
/speckit.plan
        ↓
REVIEW plan.md
        ↓
/speckit.checklist
        ↓
REVIEW / FIX REQUIREMENT QUALITY ISSUES
        ↓
/speckit.tasks
        ↓
REVIEW dependency order
        ↓
/speckit.analyze
        ↓
FIX spec/plan/tasks AT SOURCE IF NEEDED
        ↓
/speckit.implement
        ↓
RUN TESTS / MANUAL VALIDATION
        ↓
/speckit.converge
        ↓
if tasks appended:
    /speckit.implement
    /speckit.converge
repeat until converged
        ↓
FEATURE ACCEPTANCE
        ↓
NEXT FEATURE
```

`/speckit.taskstoissues` is optional. Use it only if the team wants generated tasks represented as GitHub issues.

Do not start the next dependent feature while the current feature has unresolved convergence gaps.

---

# 9. FEATURE 001 — Controlled Excel Import, Validation & Active Dataset

## Why this is first

Every user-facing feature depends on trustworthy source interpretation and a queryable active dataset.

This feature converts the current implicit extraction behavior into an explicit **data ingestion contract**.

## User/business outcome

An authorized/operator-controlled import process can take the current DR Excel workbooks, validate them, normalize them consistently, and make a complete successful dataset available to the application without allowing a bad/partial import to replace the last-known-good dataset.

## Dependencies

- Constitution complete.
- Persistence feasibility known before final plan.
- Representative Excel available.
- `SPECKIT_extract_data.md` used as compatibility reference.

## Must cover

### Workbook/file behavior

- Multiple DR `.xlsx` workbooks.
- Deterministic processing order.
- Six initiative sheets:
  - `BackEnd_Unit_Testing`
  - `Gen_Test_Case`
  - `Gen_Scripts_4_PostMan`
  - `Integration_Testing`
  - `Vuln_Discovery`
  - `FrontEnd_Unit_Testing`
- Current DR filename pattern/fallback unless intentionally changed.
- Header row detection from the early workbook rows.
- Category row immediately above detected header.
- Replace fragile positional field mapping with required-header/schema mapping wherever practical.
- Explicit validation diagnostics rather than silent structural failure.

### Row semantics

Preserve approved behavior for:

- numeric row identifiers,
- `1.0` accepted as 1,
- non-numeric label rows such as `Total` ignored,
- termination after the equivalent of five consecutive empty data rows unless intentionally changed,
- missing cell defaults,
- blank strings,
- Phase 3 datetime legacy behavior where relevant to parity.

### Status normalization

Phase status compatibility:

```text
blank / null      → Not Filled
in progress       → In Progress
in-progress       → In Progress
completed         → Completed
open              → Open
tbd               → TBD
na / n/a          → NA
hold / on hold    → On Hold
datetime          → Open (legacy compatibility case)
unknown text      → clarification required
```

Customer Approval compatibility includes:

```text
approved          → Approved
rejected          → Rejected
not yet provided  → Not Filled
yet to request    → Not Filled
hold / on hold    → On Hold
completed         → Completed
open              → Open
tbd               → TBD
in progress       → In Progress
```

### Metrics extraction

- Discover `<Metric Name> (Pre GenAI)` / `<Metric Name> (Post GenAI)` pairs dynamically.
- Pre column drives metric creation.
- Pre with no Post → Post missing/null.
- Post with no Pre → legacy ignores metric.
- Category is inherited left-to-right from category row.
- Default current extraction category behavior retained unless clarified.
- Support both:
  - `Metrics Explanation`
  - `Metrics Explaination`
- One explanation can apply to all metrics in the row.
- Numeric values stored without percentage scaling.
- Missing/non-numeric metric values handled according to approved compatibility behavior.
- Recognize real-world category typo `covergae` as Coverage at calculation/normalization boundary.

### Data activation/safety

- Successful import produces a complete normalized active dataset.
- Failed validation/import must not replace last-known-good active data.
- Import outcome/metadata must be recorded.
- Partial activation is not allowed unless explicitly approved.
- Data made available to later API features must be structured/queryable; identical legacy JSON output is not a requirement.

## Out of scope

- Dashboard UI.
- SharePoint/Graph automatic download.
- Scheduled ingestion.
- Saved filters.
- ADO links.
- Business metric summary calculations except what is required for data normalization.

## Mandatory clarification topics

Resolve items 1–11 from the clarification register, especially:

- missing sheet policy,
- unreadable workbook/batch policy,
- unknown status policy,
- invalid date policy,
- duplicate row policy,
- stable identity,
- retention/snapshot policy,
- formula/cached value semantics.

## Ready-to-paste `/speckit.specify`

```text
/speckit.specify

Create the first RLLOT modernization feature: Controlled Excel Import, Validation and Active Dataset.

Business outcome:
An operator-controlled process must be able to take the current DR-maintained rollout Excel workbooks, validate their structure and content, normalize the approved rollout fields/metrics, and make a complete successful dataset available for later dashboard features. A failed, malformed, incomplete, or rejected import must not silently replace the last-known-good active dataset.

The current extractor is reference evidence for approved source interpretation, but the new feature is greenfield and must not preserve fragile implementation details merely because they exist in the legacy code.

Required behavior:

- Accept multiple DR Excel workbooks in a controlled/manual V1 process.
- Support the six current rollout initiatives: Backend Unit Testing, Generate Test Cases, Generate PostMan Scripts, Frontend Unit Testing, Integration Testing, and Vulnerability Discovery.
- Validate required workbook/sheet/header structure before activation.
- Prefer semantic/header-based field identification rather than relying on fixed numeric column positions.
- Preserve approved status normalization and Customer Approval normalization behavior.
- Preserve approved DR/file interpretation unless clarification intentionally changes it.
- Preserve current row interpretation needed for compatibility, including valid numeric row IDs, ignored non-data label rows, missing-value defaults, and current end-of-data behavior.
- Discover Pre GenAI and Post GenAI metric columns dynamically by metric name.
- Preserve metric pairing semantics, category inheritance, explanation-column typo compatibility, numeric-value handling, and missing-value behavior.
- Produce semantically equivalent normalized application data; generating the old rllot_data.json file is not a requirement.
- Treat imported Excel content as untrusted input and surface validation problems clearly.
- Record import outcome/metadata sufficient to diagnose what was accepted or rejected.
- Activate a new dataset only after the complete import succeeds according to the approved validation policy.
- Keep the previous successful dataset available when a new import fails.

Do not include automatic SharePoint retrieval or scheduling in this feature.
Do not prescribe frontend framework, backend framework, database product, or implementation language in this specification.
Do not silently decide unresolved behaviors such as missing-sheet policy, unreadable-workbook policy, unknown statuses, duplicates, retention, formula/cached values, or invalid date severity. Surface them for clarification.
```

## Feature 001 exit criteria

Do not move on until:

- spec/clarify decisions are explicit,
- representative source data imports successfully,
- invalid sample import cannot replace active data,
- normalization/parsing parity tests pass,
- import diagnostics are visible,
- `/speckit.converge` reports converged.

---

# 10. FEATURE 002 — Seamless Internal Access & Authorization Boundary

## Why this comes before the main UI

The manager's future direction includes broader organizational visibility with controlled access. The legacy browser-only filters cannot provide authorization.

This feature establishes the identity/access boundary before the application expands.

## User/business outcome

Internal users can access RLLOT through the approved intranet experience without unnecessary additional login interaction, while the server can enforce whatever access scope the business approves.

## Dependencies

- Constitution.
- Authentication feasibility decision.
- Feature 001 for operator/data roles if import is exposed through application access.
- Exact access model clarified.

## Must cover

- Internal-only access.
- “No additional interactive login” interpretation.
- Caller identity when differentiated access is required.
- Viewer vs operator/admin responsibilities where applicable.
- Server-side authorization before data return.
- Explicit unauthorized/forbidden behavior.
- No client-side hiding as security.
- Preserve ability for same-data-for-all model if that is the approved business decision.
- Keep future row/group/data-scope authorization possible without rebuilding the UI/data model.

## Out of scope

- Saved views.
- ADO access.
- SharePoint app identity.
- External users.
- Complex role administration UI unless explicitly required.

## Mandatory clarification topics

- all viewers same data vs scoped data,
- approved corporate identity mechanism,
- viewer/operator roles,
- group membership mapping,
- whether anonymous intranet viewing is still permitted,
- expected forbidden/unauthorized UX.

## Ready-to-paste `/speckit.specify`

```text
/speckit.specify

Create the RLLOT Internal Access and Authorization Boundary feature.

Business outcome:
Internal RLLOT users must be able to access the modernized dashboard using the approved corporate/intranet experience without an unnecessary additional interactive login, while the application must be capable of enforcing approved access restrictions before data is returned.

Required behavior:

- RLLOT remains an internal application with no external-user requirement.
- If differentiated access is required, the server must know the caller identity through an approved seamless corporate authentication mechanism.
- Client-side filtering, hidden rows, route visibility, or UI state must never be treated as authorization.
- Any data-access restriction must be enforced before restricted records or aggregates are returned.
- The design must support the approved viewer/operator responsibilities needed by the application.
- Unauthorized or forbidden access must produce a safe, understandable result and must not expose restricted data.
- If the business confirms that every internal viewer may see the same data, preserve that simpler access model without inventing artificial row-level rules.
- Keep the authorization boundary extensible for future DR/manager/organization/project/team scoping if those rules are approved later.

Do not prescribe a specific authentication technology in this specification.
Do not invent the access model. Clarify whether access is same-data-for-all, DR-based, manager-based, organization/project/team-based, or another approved scope before the feature is planned.
```

## Feature 002 exit criteria

- approved access model documented,
- identity behavior validated if needed,
- server-side authorization tests exist,
- unauthorized data cannot be returned,
- no extra login behavior is understood/accepted,
- converged.

---

# 11. FEATURE 003 — Backend Unit Testing: First Dynamic Vertical Slice

## Why Backend is the first dashboard feature

This is the first full user-facing proof of the modernization:

```text
Excel
  ↓
validated active dataset
  ↓
server-side query/calculation
  ↓
dynamic UI
  ↓
filtered/paged rows
  ↓
on-demand details
```

The generic initiative pattern created here will be reused by Features 004–008.

## User/business outcome

Users can explore the Backend Unit Testing initiative dynamically without loading all RLLOT data into the browser.

## Dependencies

- Feature 001.
- Feature 002/access boundary as required.
- Query feasibility.
- Dashboard calculation reference available.
- Pre=0 decision can be resolved here or explicitly treated as known/accepted no-data behavior until business decision.

## Must cover

### Initiative header

- Backend Unit Testing label.
- Team count for current filter scope.
- Initiative overall completion.
- last refreshed / active-data timestamp where appropriate.

### Phase Progress

P1/P2/P3:

- completion percentage,
- Completed/In Progress/Open/etc status breakdown,
- canonical status ordering,
- overdue indicators,
- filtered-state recalculation.

### Completion formulas

Active statuses:

```text
Completed
In Progress
Open
On Hold
Not Filled
```

Excluded:

```text
NA
TBD
Rejected
```

Phase:

```text
Completed / Active × 100
```

Per-initiative overall:

```text
(P1% + P2% + P3%) / 3
```

Do not replace this with a flat weighted total.

### Overdue

Overdue only when:

```text
status = Open or In Progress
AND
date exists
AND
today > date
```

Row overdue if any approved overdue source is overdue:

- P1 ETA,
- P2 ETA,
- P3 ETA,
- Customer Approval prerequisite date.

Completed/NA/TBD are not overdue solely due to past date.

### Filters/search

- P1 status.
- P2 status.
- P3 status.
- Overdue yes/no.
- Free-text search over Team, Project, Organization, PM, DR.
- Customer Approval filter only if clarified/approved.
- Reset.
- Combined-filter behavior.
- Every displayed summary must reflect current filter scope.

### Table

Columns/user-visible concepts:

- #.
- Org.
- Project.
- Team.
- PM.
- DR.
- Approval + prerequisite date.
- P1 status + ETA + overdue.
- P2 status + ETA + overdue.
- P3 status + ETA + overdue.
- Productivity.
- Quality.
- Coverage.
- Details.

Also:

- server-side paging,
- useful sorting, preferably server-side with paging,
- empty result behavior,
- no full active dataset in browser by default.

### Backend-specific Productivity metric

```text
Unit test creation effort Hrs
```

### Metric summary/business rules

Lower-is-better tokens:

```text
time
effort
hrs
hour
escape
maintenance
flaky
```

Lower-is-better:

```text
(pre - post) / pre × 100
```

Higher-is-better:

```text
(post - pre) / pre × 100
```

Post-average classification:

- all Coverage,
- Acceptance Rate %,
- Critical/High Found Pre-Check-In,
- Critical/High Found Pre Check-In,
- name containing `cov` / `coverage`.

Metric aggregation must use aggregated total Pre/Post across valid pairs, **not average of individual team improvements**.

Reported count requires both Pre and Post.

Exclude from summary:

- Number of test cases generated.
- Number of test scripts generated.
- Number of unit tests generated.
- Number of integration scenarios generated.
- Number of APIs covered.

Recognize category typo `covergae`.

Critical/High metric is a plain number, not percentage.

### Details

On-demand team details:

- row/team title,
- project/org context,
- metric explanations,
- duplicate explanations shown once preserving meaningful order,
- metrics grouped Productivity / Quality / Coverage / Other,
- Before,
- After,
- change where calculable,
- remarks only when present,
- clear no-data state.

### Presentation semantics

- preserve semantic status distinctions and consistent status styling,
- exact legacy colors/SVG mechanics are not required,
- no unsafe raw HTML from Excel-originated content.

## Mandatory clarification topics

- Pre=0 rule.
- zero-active-row display: 0% vs N/A.
- Customer Approval filter.
- current date/time-zone definition for overdue.
- page size/default/max behavior.
- sort columns/order.
- whether Customer Approval prereq overdue contributes to row-overdue filter.

## Ready-to-paste `/speckit.specify`

```text
/speckit.specify

Create the first dynamic RLLOT dashboard vertical slice for the Backend Unit Testing initiative.

Business outcome:
An internal user must be able to explore Backend Unit Testing rollout progress dynamically, using server-side data retrieval rather than loading the complete RLLOT dataset into the browser.

Required user behavior:

- View Backend Unit Testing overall completion and team count.
- View Phase 1, Phase 2 and Phase 3 progress with status breakdowns and overdue indications.
- View Productivity, Quality and Coverage summary metrics using the approved RLLOT calculation rules.
- Filter records by Phase 1, Phase 2 and Phase 3 status.
- Filter overdue/non-overdue records.
- Search using Team, Project, Organization, PM and DR values.
- Combine filters and see all displayed phase/overall/metric summaries recalculate for the active filter scope.
- Browse team records using pagination rather than receiving the entire dataset.
- Sort useful table columns.
- See Customer Approval, prerequisite date, three phase statuses/ETAs, Productivity, Quality, Coverage and a Details action.
- Open one team record on demand to see metric explanations, Before/After values, calculated change where available, and remarks.
- Show safe and understandable empty/no-data states.

Business calculation compatibility:

- Active completion statuses are Completed, In Progress, Open, On Hold and Not Filled.
- NA, TBD and Rejected are excluded from the completion denominator.
- Phase completion = Completed / Active.
- Initiative overall completion = average of P1%, P2% and P3%.
- Overdue applies only to Open/In Progress with a past applicable date; Completed/NA/TBD must not be overdue merely because a date is past.
- Backend Unit Testing Productivity uses `Unit test creation effort Hrs`.
- Preserve approved lower-is-better, higher-is-better, Post-average, reported-count, excluded-summary-metric, plain-number-metric and category-normalization behavior from the captured RLLOT reference.
- Initiative metric aggregation must use aggregate Pre/Post totals across valid pairs rather than averaging each team's individual improvement percentages.
- Business formulas must have one authoritative server-side/domain implementation and must not be independently reimplemented in the frontend.

UX:
Preserve the current initiative information hierarchy as the baseline, but do not require pixel-perfect legacy HTML/CSS/SVG reproduction.

Security/data:
Imported text is untrusted and must be rendered safely.
Only data allowed by the approved access boundary may be returned.

Do not include the cross-initiative Overview/global filters in this feature.
Do not include SharePoint automation.
Do not silently decide the known Pre=0 rule, zero-active 0% vs N/A behavior, Customer Approval filter requirement, or any other unresolved ambiguity; surface them for clarification.
Do not prescribe implementation technology in this specification.
```

## Feature 003 exit criteria

- one complete initiative works end-to-end,
- phase/overall formulas pass parity tests,
- overdue tests pass,
- metric parity matrix passes,
- combined filters recalculate summaries correctly,
- pagination prevents whole-dataset loading,
- details are on-demand,
- no duplicated client business formulas,
- converged.

---

# 12. FEATURE 004 — Generate Test Cases Initiative

## Dependency

Feature 003 must have established the generic initiative/query/calculation/detail pattern.

## Initiative-specific Productivity metric

```text
Total Effort Hrs required to create test cases
```

## Important rule

Do not fork a second set of calculation logic. Reuse the single calculation authority and generic initiative behavior.

## Ready-to-paste `/speckit.specify`

```text
/speckit.specify

Add the Generate Test Cases initiative to the modernized RLLOT dashboard.

Business outcome:
Users must be able to explore Generate Test Cases rollout progress with the same approved dynamic initiative experience established for Backend Unit Testing.

Required behavior:

- Provide initiative completion, P1/P2/P3 progress, status breakdowns and overdue indicators.
- Provide server-side phase filters, overdue filtering, search, pagination and sorting.
- Recalculate all displayed summaries for the active filter scope.
- Provide the standard team detail table and on-demand team details with explanations, Before/After metric values and remarks.
- Preserve the shared RLLOT completion, overdue, status, metric aggregation, reported-count, Post-average, lower-is-better, excluded-summary and no-data rules.
- The designated Productivity metric for this initiative is `Total Effort Hrs required to create test cases`.
- Preserve initiative-specific Quality/Coverage metrics present in the normalized source data according to the shared metric-classification rules.
- Reuse the established single server-side calculation authority and generic initiative behavior rather than creating initiative-specific duplicate formulas.

Do not redesign the shared initiative experience.
Do not include Overview/global filters yet.
Do not include SharePoint automation.
Do not prescribe the technology stack in the specification.
```

---

# 13. FEATURE 005 — Generate PostMan Scripts Initiative

## Initiative-specific Productivity metric

```text
Time required to Generate API Collections
```

## Ready-to-paste `/speckit.specify`

```text
/speckit.specify

Add the Generate PostMan Scripts initiative to the modernized RLLOT dashboard.

Business outcome:
Users must be able to explore Generate PostMan Scripts rollout progress using the established dynamic initiative workflow.

Required behavior:

- Provide initiative completion, P1/P2/P3 progress, status breakdowns and overdue indicators.
- Provide server-side phase filters, overdue filtering, search, pagination and sorting.
- Recalculate summaries using the active filter scope.
- Provide the standard team table and on-demand details.
- Preserve the shared RLLOT completion, overdue, metric aggregation, reported-count, lower-is-better, Post-average, excluded-summary, category-normalization and no-data rules.
- The designated Productivity metric is `Time required to Generate API Collections`.
- Reuse the existing generic initiative behavior and the single calculation authority. Do not create a separate formula implementation for this initiative.

Do not include Overview/global filters yet.
Do not include SharePoint automation.
Do not prescribe implementation technology in this specification.
```

---

# 14. FEATURE 006 — Frontend Unit Testing Initiative

## Initiative-specific Productivity metric

```text
Total Effort Hrs required to create UI unit tests
```

## Ready-to-paste `/speckit.specify`

```text
/speckit.specify

Add the Frontend Unit Testing initiative to the modernized RLLOT dashboard.

Business outcome:
Users must be able to explore Frontend Unit Testing rollout progress using the established dynamic initiative workflow.

Required behavior:

- Initiative overall and phase progress.
- P1/P2/P3 status breakdowns and overdue indicators.
- Server-side filtering, search, pagination and sorting.
- Filter-scoped summary recalculation.
- Team detail table and on-demand metric/remarks detail.
- Shared approved RLLOT calculation and no-data behavior.
- The designated Productivity metric is `Total Effort Hrs required to create UI unit tests`.
- Reuse shared configuration/components/calculation authority instead of duplicating business rules.

Do not include Overview/global filters yet.
Do not include SharePoint automation.
Do not prescribe implementation technology in this specification.
```

---

# 15. FEATURE 007 — Integration Testing Initiative

## Initiative-specific Productivity metric

```text
Total time required to create integration tests
```

## Ready-to-paste `/speckit.specify`

```text
/speckit.specify

Add the Integration Testing initiative to the modernized RLLOT dashboard.

Business outcome:
Users must be able to explore Integration Testing rollout progress through the same approved dynamic initiative experience.

Required behavior:

- Initiative overall and P1/P2/P3 progress.
- Status breakdowns and overdue indicators.
- Server-side filters, search, pagination and sorting.
- Filter-scoped recalculation.
- Standard team table and on-demand details.
- Preserve shared completion, overdue, metric classification, aggregation, reported-count, excluded-summary and no-data behavior.
- The designated Productivity metric is `Total time required to create integration tests`.
- Reuse the shared initiative implementation and single calculation authority.

Do not include Overview/global filters yet.
Do not include SharePoint automation.
Do not prescribe implementation technology in this specification.
```

---

# 16. FEATURE 008 — Vulnerability Discovery Initiative

## Initiative-specific Productivity metric

```text
Time taken to run vulnerability
```

## Ready-to-paste `/speckit.specify`

```text
/speckit.specify

Add the Vulnerability Discovery initiative to the modernized RLLOT dashboard.

Business outcome:
Users must be able to explore Vulnerability Discovery rollout progress using the shared dynamic initiative experience.

Required behavior:

- Initiative overall and P1/P2/P3 progress.
- Status breakdowns and overdue indicators.
- Server-side filters, search, pagination and sorting.
- Filter-scoped summary recalculation.
- Team table and on-demand details.
- Preserve the shared RLLOT completion, overdue, metric, reported-count, excluded-summary and no-data semantics.
- The designated Productivity metric is `Time taken to run vulnerability`.
- Reuse the shared initiative configuration/components and authoritative server-side business calculations.

Do not include Overview/global filters yet.
Do not include SharePoint automation.
Do not prescribe implementation technology in this specification.
```

---

# 17. Important completeness check after Feature 008

At this point all six initiative detail experiences must exist:

```text
Backend Unit Testing
Generate Test Cases
Generate PostMan Scripts
Frontend Unit Testing
Integration Testing
Vulnerability Discovery
```

Before Feature 009, verify there was **no accidental divergence** between initiatives in:

- phase formulas,
- status order/meaning,
- overdue logic,
- metric classification,
- category handling,
- reporting count,
- filters/search,
- pagination/sorting,
- detail model,
- data authorization.

Differences should come from initiative data/configuration, not duplicated formulas.

---

# 18. FEATURE 009 — Cross-Initiative Overview & Global Filters

## Why this is after all six initiative features

Overview is not independent. It combines all six initiatives and its global filters must affect every activity card/KPI consistently.

## User/business outcome

Users can understand rollout health across the entire program and narrow the whole dashboard by Organization, DR and PM/SM while all KPIs/activity summaries remain consistent with that active filter scope.

## Dependencies

Features 003–008.

## Must cover

### Global shell/filter behavior

- Overview + six initiative navigation.
- Global Organization filter.
- Global DR filter.
- Global PM / SM filter.
- Reset Global.
- Values derived from active authorized dataset.
- Cascading/relevant filter options where approved.
- Global filter state preserved while navigating initiatives.
- Global filters affect initiative-level data/aggregates and Overview.
- Server-side filter enforcement/calculation, not DOM hiding.

### Seven Overview KPIs

1. Overall Completion.
2. Total Team-Initiatives.
3. Fully Completed — all three phases completed.
4. Phase 1 Completed count / active.
5. Phase 2 Completed count / active.
6. Phase 3 Completed count / active.
7. Teams with Due Phase(s) — exact Customer Approval inclusion must be clarified.

### Overview Overall formula

This is intentionally **different** from initiative overall:

```text
overview completed =
    all P1 completed
  + all P2 completed
  + all P3 completed

overview active =
    all P1 active
  + all P2 active
  + all P3 active

Overview Overall % =
    overview completed / overview active × 100
```

Do not calculate Overview Overall as the average of six initiative overall percentages.

### Activity summary cards

Six cards, in current business/UX order:

- Backend Unit Testing.
- Generate Test Cases.
- Generate PostMan Scripts.
- Frontend Unit Testing.
- Integration Testing.
- Vulnerability Discovery.

Each card includes:

- initiative overall,
- team count,
- P1/P2/P3 completed counts,
- status/progress visualization,
- Productivity,
- concise Quality,
- Coverage,
- navigation to initiative.

Current Overview concise Quality behavior uses `Acceptance Rate %`.

### Descriptive activity information

Preserve the current Overview activity-description section unless business chooses to remove it.

### Filtered state

After global filters change:

- Overview Overall recalculates.
- phase counts recalculate.
- fully completed recalculates.
- overdue/due-team count recalculates.
- each initiative activity card recalculates.
- navigating to an initiative uses the same global scope.
- no unauthorized rows are used in aggregates.

## Mandatory clarification topics

- Customer Approval prerequisite contribution to “Teams with Due Phase(s)”.
- global filter cascade behavior.
- empty/zero-active Overview display.
- whether filter selections should persist only for session or via URL/state.
- exact Total Team-Initiatives semantics if duplicate/source rows exist.

## Ready-to-paste `/speckit.specify`

```text
/speckit.specify

Create the cross-initiative RLLOT Overview and Global Filters feature after all six initiative detail features are available.

Business outcome:
Users must be able to understand overall GenAI SDLC rollout progress across all six initiatives and narrow the entire dashboard using global Organization, DR and PM/SM filters while all displayed KPIs and initiative summaries remain consistent with the active global scope.

Required behavior:

Global navigation and filters:
- Provide Overview plus navigation to all six initiatives.
- Provide global Organization, DR and PM/SM filters and a Reset Global action.
- Global filter state must consistently affect Overview and initiative views.
- Filter values must come from data the caller is authorized to access.
- Do not implement global filtering by hiding rows already delivered to the browser.

Overview KPIs:
1. Overall Completion.
2. Total Team-Initiatives.
3. Fully Completed where all three phases are Completed.
4. Phase 1 Completed count relative to active Phase 1 population.
5. Phase 2 Completed count relative to active Phase 2 population.
6. Phase 3 Completed count relative to active Phase 3 population.
7. Teams with Due Phase(s), with the precise Customer Approval prerequisite-date treatment clarified before planning.

Important calculation rule:
The Overview Overall KPI is a flat weighted completed/active fraction across all phase records from all initiatives. It is not the average of initiative overall percentages.

Activity cards:
- Show one activity card for each of the six initiatives in the established order.
- Show initiative overall %, team count, P1/P2/P3 completion information, progress/status visualization and concise Productivity/Quality/Coverage summaries.
- Use Acceptance Rate % as the concise Overview Quality metric where that is the approved current behavior.
- Activity cards navigate to the matching initiative while preserving global scope.

Filtered-state requirement:
Every Overview KPI and activity card must recalculate from the active authorized global filter scope. The frontend must consume authoritative calculated results rather than implement a separate copy of RLLOT business formulas.

Preserve the current Overview information hierarchy and activity-description content as a UX baseline without requiring pixel-perfect legacy design.

Do not include SharePoint automation, saved views, or ADO drill-through in this feature.
Do not prescribe technology in the specification.
Do not silently guess the due-phase/Customer Approval ambiguity, zero-active display, global-filter cascade behavior, or duplicate/team-initiative counting rules; surface them for clarification.
```

## Feature 009 exit criteria

- all seven KPIs verified,
- weighted Overview formula parity verified,
- all six cards verified,
- global filters affect every relevant result,
- filter scope persists correctly through navigation,
- no unauthorized data contributes to aggregates,
- converged.

---

# 19. FEATURE 010 — Production Readiness, Parity, Cutover & Rollback

## Why this is a separate feature

A generated application is not production-ready just because screens work.

This feature validates operations, correctness, security, recoverability, and migration from the old dashboard.

## User/business outcome

RLLOT can be safely deployed and operated on the approved internal environment, compared against the legacy dashboard, rolled back if required, and cut over only after accepted parity.

## Dependencies

Features 001–009 complete.

## Must cover

### Hosting/configuration

- approved internal IIS/Windows deployment,
- supported runtime/hosting components,
- environment-specific configuration,
- protected data/log paths outside public web root,
- no secrets committed to source,
- safe static frontend/API hosting arrangement.

### Operational behavior

- health/readiness indication,
- structured application/import logging,
- understandable import failure diagnostics,
- active dataset/import timestamp visibility,
- backup/recovery for persistence,
- rollback procedure,
- service/application restart procedure,
- deployment/runbook.

### Security

- authorization enforced server-side,
- Excel-originated strings handled safely,
- no unsafe raw HTML injection path for imported data,
- appropriate security headers/configuration,
- least privilege for application/data folder identities,
- no public exposure unless explicitly approved.

### Performance

Validate representative and projected workload:

```text
current teams      ~100–150
12-month estimate  ~200–300
current viewers    <20
projection         <50 concurrent viewers
```

The goal is reasonable internal responsiveness, not artificial big-data engineering.

Verify:

- filtered/paged endpoints do not ship full dataset by default,
- overview aggregate response is bounded,
- detail response is on-demand.

### Parallel parity validation

Run old and new dashboard against equivalent source data.

Compare at minimum:

- row/team counts,
- normalized statuses,
- customer approvals,
- P1/P2/P3 counts,
- P1/P2/P3 percentages,
- initiative overall,
- Overview Overall,
- fully completed,
- overdue counts,
- metric summaries,
- reported counts,
- designated productivity metrics,
- excluded metrics,
- filters/search,
- sorting/pagination result correctness,
- team detail values,
- explanation deduplication,
- remarks.

### Calculation parity matrix

Completion:

- Completed + Open + In Progress.
- NA/TBD/Rejected exclusions.
- zero active.
- initiative overall equal-weight phase average.
- Overview weighted formula.

Overdue:

- Open past ETA.
- In Progress past ETA.
- future ETA.
- Completed past ETA.
- NA/TBD past ETA.
- Customer Approval prerequisite behavior.
- any applicable overdue source makes row overdue.

Metrics:

- higher-is-better.
- lower-is-better.
- Pre missing.
- Post missing.
- Pre=0 approved behavior.
- Coverage post-average.
- Acceptance Rate post-average.
- Critical/High plain-number.
- aggregate totals vs average-of-row-improvements.
- reported count.
- excluded count metric.
- initiative-specific productivity.
- `covergae` compatibility.

Filters:

- Org.
- DR.
- PM.
- P1/P2/P3.
- overdue.
- search.
- combined filters.
- filtered aggregates equal direct calculation from the same filtered source.

Details:

- explanation.
- duplicate explanation.
- missing explanation.
- metrics present/missing.
- remarks present/absent.

### Cutover

- legacy dashboard remains available through acceptance window,
- define acceptance owner,
- define rollback trigger,
- cut over only after parity + operational approval,
- retain legacy briefly for rollback,
- then explicitly retire old static generation pipeline.

## Ready-to-paste `/speckit.specify`

```text
/speckit.specify

Create the RLLOT Production Readiness, Parity, Cutover and Rollback feature.

Business outcome:
The fully modernized RLLOT application must be safe to deploy and operate in the approved internal environment, must demonstrate accepted behavioral parity with the legacy dashboard, and must support recovery/rollback before the old static pipeline is retired.

Required behavior:

- Provide an approved internal deployment and operational process.
- Provide application/import diagnostics sufficient for operators to understand failures.
- Provide health/readiness visibility appropriate for the internal application.
- Protect application data and logs from public web access.
- Keep configuration environment-specific and keep secrets out of source control.
- Provide persistence backup/recovery and a documented rollback path.
- Validate that server-side paging/filtering/detail retrieval prevents unnecessary full-dataset delivery.
- Validate the expected internal workload without introducing unnecessary big-data infrastructure.
- Run the modern and legacy solutions in parallel against equivalent source data during acceptance.
- Compare and validate row counts, normalized statuses, approvals, all phase counts/percentages, initiative overall, Overview Overall, overdue behavior, metrics, reported counts, filters/search, sorting/pagination and team details.
- Execute automated parity tests covering completion exclusions, zero-active behavior, overdue edge cases, higher/lower-is-better metrics, Post-average metrics, Pre/Post missing values, approved Pre=0 behavior, excluded metrics, category typo compatibility, filtered aggregates and detail no-data cases.
- Ensure imported/untrusted text is rendered safely and access control is enforced server-side.
- Define cutover acceptance, rollback trigger and the controlled retirement of the legacy static dashboard only after acceptance.

Do not include automatic SharePoint retrieval in this feature.
Do not treat production cutover as complete merely because deployment succeeded; parity and operational acceptance are required.
```

---

# 20. DEFERRED FEATURE 011 — SharePoint File Retrieval Automation

Do this **after** the core dynamic dashboard is accepted.

## Outcome

Approved RLLOT Excel workbooks can be retrieved from SharePoint programmatically without a developer manually downloading them.

## Dependencies

- Feature 001 ingestion contract.
- Approved enterprise identity/app registration.
- SharePoint/Graph permission approval.
- Feature 010 core application stability.

## Must cover

- approved workbook/library location,
- least-privilege file access,
- correct file selection,
- file naming/DR handling,
- secure credentials/identity,
- retrieval failure diagnostics,
- no activation when retrieval/import fails,
- idempotent handling of the same source version where appropriate.

## Ready-to-paste `/speckit.specify`

```text
/speckit.specify

Automate retrieval of the approved RLLOT Excel source workbooks from SharePoint.

Business outcome:
Operators should no longer need to manually download the approved DR Excel workbooks before ingestion.

Required behavior:

- Retrieve only the approved RLLOT source files from the approved SharePoint location.
- Use an approved enterprise identity and least-privilege access.
- Preserve the existing ingestion validation and active-dataset safety contract.
- A retrieval/authentication/permission/file-selection failure must be visible and must not change the active dataset.
- The retrieved files must enter the same ingestion path used by controlled/manual V1 import rather than creating a second parser.
- Reprocessing the same source version must not create unintended duplicate active data.

Do not add scheduling in this feature.
Do not change dashboard business formulas.
Do not invent tenant permissions or credentials; those are external prerequisites.
```

---

# 21. DEFERRED FEATURE 012 — Scheduled Ingestion

## Dependency

Feature 011.

## Outcome

SharePoint retrieval + ingestion executes on the approved schedule without manual operator action.

## Must cover

- agreed daily schedule/time zone,
- no overlapping runs,
- safe locking/concurrency,
- idempotency,
- missed/failed run handling,
- logging,
- last-success timestamp,
- active dataset unchanged on failure,
- manual re-run/recovery.

## Ready-to-paste `/speckit.specify`

```text
/speckit.specify

Add scheduled RLLOT source retrieval and ingestion.

Business outcome:
The approved SharePoint source workbooks should be retrieved, validated and ingested automatically on the agreed schedule without requiring a developer to run the pipeline manually.

Required behavior:

- Run at the approved business schedule and time zone.
- Prevent overlapping ingestion runs.
- Reuse the existing SharePoint retrieval and ingestion capabilities.
- Preserve idempotency and active-dataset safety.
- Failed scheduled runs must not replace the last-known-good data.
- Record last attempt, last success and failure diagnostics.
- Allow an authorized operator to safely retry/recover a failed run.

Do not create a second ingestion implementation.
Do not change dashboard calculations or source interpretation.
```

---

# 22. DEFERRED FEATURE 013 — Saved Views / Filter Profiles

## Dependencies

- Feature 002 identity if views are user-specific.
- Feature 009 global/initiative filters.

## Outcome

Users can save frequently used dashboard filter combinations and restore them later.

## Must cover

- save named view,
- current global + initiative filter context,
- load/apply view,
- rename/delete,
- optional default view if approved,
- private vs shared behavior must be clarified,
- authorization on shared views,
- behavior when a saved filter value no longer exists.

---

# 23. DEFERRED FEATURE 014 — ADO / Work-Item Drill-Through

## Dependencies

- Source data must contain or be mappable to valid work-item identifiers/links.
- Access/security behavior must be defined.

## Outcome

Users can navigate from relevant RLLOT detail to the corresponding approved ADO/work-item context.

## Important

Do not implement until the source of the work-item identifier and the authorization expectation are explicitly defined.

---

# 24. Items deliberately NOT in the current roadmap

Do not let Copilot add these unless the business explicitly asks:

- replacing Excel as the business source system,
- direct dashboard write-back to Excel,
- new data-entry workflow,
- Power BI rewrite,
- Power Automate dependency,
- new Azure App Service/Azure Functions/Azure SQL/Cosmos by default,
- microservices,
- event-driven architecture,
- Kafka/Event Hubs,
- distributed caching,
- Kubernetes,
- multi-region/multi-server architecture,
- AI/LLM features,
- advanced historical analytics,
- external-user access.

The current workload does not justify these by default.

---

# 25. Feature-to-requirement coverage matrix

This matrix is a final check that important captured behavior has a home.

| Requirement / behavior | Feature |
|---|---|
| Multiple DR Excel inputs | 001 |
| Six source sheets | 001 |
| Header/schema validation | 001 |
| Remove fixed column-index fragility | 001 |
| Status normalization | 001 |
| Customer Approval normalization | 001 |
| DR filename parsing / source identity | 001 |
| Pre/Post metric pairing | 001 |
| Sticky metric category | 001 |
| Explanation typo support | 001 |
| Missing/default values | 001 |
| Last-known-good import | 001 |
| Import diagnostics/history | 001 |
| Formula/cached-value behavior | 001 clarify |
| Duplicate semantics | 001 clarify |
| Retention/snapshot model | 001 clarify |
| No additional interactive login | 002 |
| Server-side authorization | 002 |
| Future data-scope access | 002 |
| Backend initiative | 003 |
| Phase completion formula | 003 shared |
| Initiative overall equal-phase average | 003 shared |
| Overdue rule | 003 shared |
| Productivity/Quality/Coverage | 003 shared |
| Lower-is-better | 003 shared |
| Post-average | 003 shared |
| Excluded summary metrics | 003 shared |
| Reported count | 003 shared |
| Aggregate totals vs row-average distinction | 003 shared |
| Plain-number Critical/High | 003 shared |
| `covergae` typo | 003 shared |
| P1/P2/P3 filters | 003 shared |
| Overdue filter | 003 shared |
| Search Team/Project/Org/PM/DR | 003 shared |
| Customer Approval filter | 003 clarify |
| Paging | 003 shared |
| Sorting | 003 shared |
| Team details | 003 shared |
| Explanation dedupe | 003 shared |
| Remarks | 003 shared |
| Safe rendering of imported text | Constitution + 003 + 010 |
| Generate Test Cases | 004 |
| Generate PostMan Scripts | 005 |
| Frontend Unit Testing | 006 |
| Integration Testing | 007 |
| Vulnerability Discovery | 008 |
| Six-initiative navigation | 009 |
| Overview Overall weighted formula | 009 |
| Total Team-Initiatives | 009 |
| Fully Completed | 009 |
| P1/P2/P3 completed KPIs | 009 |
| Teams with Due Phase(s) | 009 clarify |
| Organization global filter | 009 |
| DR global filter | 009 |
| PM/SM global filter | 009 |
| Reset Global | 009 |
| Global-filter aggregate recalculation | 009 |
| Activity cards | 009 |
| Overview concise Acceptance Rate quality | 009 |
| Activity descriptions | 009 |
| Existing IIS production deployment | 010 |
| Logging/health | 010 |
| Backup/recovery | 010 |
| Performance validation | 010 |
| Full parity matrix | 010 |
| Parallel legacy/new run | 010 |
| Cutover/rollback | 010 |
| SharePoint automatic download | 011 deferred |
| Scheduled ingestion | 012 deferred |
| Saved views | 013 deferred |
| ADO drill-through | 014 deferred |
| Pre=0 business rule | 003 clarify + 010 parity |
| Zero active: 0% vs N/A | 003/009 clarify |
| Single calculation authority | Constitution + 003 onward |

---

# 26. Rules for reviewing Copilot output at every feature

Before accepting any generated `spec.md`, check:

### Problem-space check

- Is it written in user/business terms?
- Did it avoid unnecessary technology?
- Did it invent a new business requirement?
- Did it accidentally copy a legacy implementation detail?

### Behavior check

- Are success criteria measurable?
- Are negative/error cases present?
- Are edge cases explicit?
- Did it preserve known RLLOT rules?
- Did it surface unresolved rules instead of guessing?

### Dependency check

- Does this feature depend on something not yet implemented?
- Did the generated spec pull a deferred capability forward?
- Is it trying to solve Overview before all initiative behavior exists?
- Is it trying to automate SharePoint before the main dashboard works?

### Architecture check at `/plan`

- Does plan follow the accepted ADR?
- Does it preserve database portability?
- Does it create one calculation authority?
- Does it avoid shipping the full dataset?
- Does it enforce authorization server-side?
- Does it keep manual Excel V1 unless this is Feature 011+?
- Does it avoid paid services unless approved?

### Implementation check

- Are tests generated before/with critical business logic?
- Are parity cases covered?
- Are frontend formulas duplicated?
- Are provider-specific shortcuts leaking into domain rules?
- Are imported strings treated as untrusted?
- Can a failed import change active data?
- Is pagination real server-side paging, not just slicing already-downloaded data?

---

# 27. Recommended working rhythm

Do not treat Spec Kit as one huge prompt chain.

Use this rhythm:

```text
Day / iteration N:

1. Run one /speckit.specify
2. Read the generated spec yourself
3. Run /speckit.clarify
4. Resolve business questions
5. Re-read spec
6. Run /speckit.plan only with validated architecture inputs
7. Review plan
8. Generate checklist/tasks
9. Analyze
10. Implement in bounded phases if large
11. Run tests and use the application
12. Converge
13. Capture lessons learned
14. Only then start next feature
```

For a large feature, implementation can also be staged by task phase rather than asking the agent to execute every task in a single context window.

---

# 28. Suggested SDD learning log

Because this RLLOT project is also intended to become a practical greenfield SDD reference, after every feature record:

```text
Feature:
What requirement was initially ambiguous?
What did /clarify expose?
What did Spec Kit generate well?
What did it infer incorrectly?
What did we manually correct?
What changed between spec and plan?
Did tasks respect dependencies?
What did /analyze catch?
What did /converge catch?
What business-rule parity issue was found?
What guideline would we give the next team?
```

This is what will later make the project useful as an organizational SDD guideline rather than merely a dashboard rewrite.

---

# 29. Current V1 completion boundary

V1 is complete after Feature 010 when:

```text
Manual/controlled Excel
        ↓
validated safe ingestion
        ↓
normalized relational active data
        ↓
server-side application
        ↓
all six dynamic initiatives
        ↓
cross-initiative Overview/global filters
        ↓
approved internal access model
        ↓
production IIS operation
        ↓
parity accepted
        ↓
cutover completed
```

SharePoint retrieval and scheduling are valuable follow-ons, but they are **not** allowed to block this primary modernization objective.

---

# 30. First action to take on the office laptop

After initializing Spec Kit and copying the reference docs:

1. Run the **constitution prompt** from Section 4.
2. Review/correct the generated constitution.
3. Run **Feature 001 `/speckit.specify`** from Section 9.
4. Stop.
5. Review the generated `spec.md`.
6. Then run `/speckit.clarify`.

Do **not** paste Feature 002 or later prompts yet.

---

**End of execution playbook**
