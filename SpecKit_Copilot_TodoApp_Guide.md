# Spec Kit + GitHub Copilot Chat — Token-Efficient Guide
### Todo App: Install → Iteration 2
**Stack:** ASP.NET Core Web API, EF Core, SQL Server, React

---

## 0. Why token efficiency matters here

Copilot Business bills **premium requests as credits** (your screenshot: `0 / 3,000 used`, resets monthly). Every Copilot **Chat** message, and every `/implement`-style agent turn, burns credits. Spec Kit's `specify`, `plan`, and `tasks` commands run **locally via CLI** — they don't touch Copilot at all. Only the actual code-generation step should go through Chat. Keep that separation strict.

**Golden rules (apply throughout):**
1. Turn **off** Ghost Text / Inline Suggestions while doing Spec Kit work — you don't need autocomplete while reviewing generated specs. (Settings → Copilot icon → Inline Suggestions → uncheck Ghost Text.)
2. Never paste full file contents into chat — reference files with `#filename` instead. Copilot reads them from the workspace at near-zero extra token cost vs. pasting.
3. One consolidated prompt > five small follow-ups. Batch every requirement into a single message.
4. Use `/clear` (or start a new chat) between unrelated tasks so old context isn't re-sent every turn.
5. Prefer the **base model** (GPT-4o-mini equivalent) for boilerplate; reserve premium models only for architecture-level reasoning if your org's credit multiplier makes that distinction.
6. Ask Copilot to generate **one layer at a time** (entity → repo → controller → tests), not the whole app in one giant prompt — smaller diffs are cheaper to regenerate if wrong.

---

## 1. Install Spec Kit CLI (one-time, no Copilot credits used)

```bash
# Requires Python 3.11+ and uv
pip install uv
uv tool install specify-cli --from git+https://github.com/github/spec-kit.git
```

Verify:
```bash
specify check
```

## 2. Initialize the project

```bash
specify init todo-app --ai copilot
cd todo-app
```

This scaffolds `.specify/` (templates + scripts) and a `.github/prompts/` folder wiring `/constitution`, `/specify`, `/plan`, `/tasks`, `/implement` as Copilot Chat slash commands. **No credits used yet** — this is filesystem scaffolding only.

Open the folder in VS Code:
```bash
code .
```

Confirm the GitHub Copilot Chat extension is installed and signed in to your org's Business plan.

---

## 3. `/constitution` — one-time, keep it short

Open Copilot Chat (Ctrl+Alt+I), type:

```
/constitution
Principles: tests written before implementation (Playwright E2E per requirement);
Clean Architecture layering (Domain/Application/Infrastructure/API);
EF Core code-first with migrations; no business logic in controllers;
React functional components with hooks only.
```

**Token tip:** this is a single message, run once per project. Don't re-paste it in later chats — it's saved to `.specify/memory/constitution.md` and auto-referenced by later commands.

---

## 4. Iteration 1 — `/specify`

```
/specify
Todo app. Users can:
- Add a task with title and optional due date
- Mark a task complete/incomplete
- Delete a task
- Filter tasks by status (all/active/completed)
```

Output: `spec.md` with FR-001–FR-004. **Review it in the editor, not in chat** — don't ask Copilot to "show me the spec again," just open the file. Saves a round trip.

## 5. Iteration 1 — `/plan`

```
/plan
Backend: ASP.NET Core Web API, EF Core, SQL Server, Clean Architecture layers.
Frontend: React, useState/useReducer, no external state library.
Endpoints: GET/POST/PUT/DELETE /api/todos.
```

## 6. Iteration 1 — `/tasks`

```
/tasks
```
(No extra prompt text needed — Spec Kit reads spec.md + plan.md automatically.)

## 7. Iteration 1 — `/implement` (this is where credits are spent)

Do this **layer by layer**, one Chat message per layer, referencing the tasks file instead of repeating requirements:

```
/implement
Generate only: Todo entity (Models/TodoItem.cs) and AppDbContext (Data/AppDbContext.cs).
Reference #tasks.md for field list. Nothing else yet.
```

Then, new turn:
```
Now generate the EF Core migration and repository/service layer for Todo, per #tasks.md.
```

Then:
```
Now generate TodosController with the 4 REST endpoints from #plan.md.
```

Then, switch context to frontend (start fresh chat to drop backend context):
```
Generate React components AddTodo, TodoList, TodoItem, FilterBar per #spec.md.
Functional components, hooks only, no external libraries.
```

Finally, tests (per your constitution — tests before/alongside code is fine for a demo):
```
Generate Playwright E2E tests, one per FR-001 to FR-004, in tests/todo.spec.ts.
```

**Why split like this:** each message is small and focused, so if one output is wrong you regenerate a 20-line file, not the whole app — cheaper in credits and in your own review time.

---

## 8. Iteration 2 — expand scope

### `/specify` (append, don't restate iteration 1)
```
/specify
Add to existing spec: task priority (Low/Medium/High), category/tag field,
sorting by due date or priority, and persistence check (state survives refresh).
```

### `/plan` (incremental)
```
/plan
Extend Todo entity with Priority enum and Category string.
Extend GET /api/todos with query params sortBy and category.
```

### `/tasks`
```
/tasks
```

### `/implement` — layer by layer again

```
/implement
Update TodoItem entity + create new EF Core migration for Priority and Category
fields only. Reference #tasks.md.
```

```
Update repository/service and TodosController to support sortBy and category
query params, per #plan.md.
```

New chat (frontend only):
```
Update AddTodo to include priority select and category input.
Add sort/filter controls to FilterBar. Reference #spec.md FR-006 to FR-009.
```

```
Add Playwright tests for: priority display, sort-by-priority ordering,
category filter, and refresh-persistence check.
```

---

## 9. Post-implementation credit hygiene

- Re-enable Ghost Text only when doing manual coding, not spec review.
- Delete/close unused chat sessions — some clients keep sending session history as context, which silently adds tokens per turn.
- If a generated file is 90% right, ask Copilot to **edit the specific function**, not regenerate the file:
  ```
  In TodosController.cs, only fix the sort parameter validation. Leave everything else unchanged.
  ```
- Track usage via Settings → Copilot → Credits panel (same screen as your screenshot) after each iteration to gauge burn rate before scaling this pattern to other EA team features.

---

## 10. Quick reference — command cheat sheet

| Command | Costs Copilot credits? | Purpose |
|---|---|---|
| `specify init` | No (CLI only) | Scaffold project |
| `/constitution` | Yes, once | Set project-wide rules |
| `/specify` | Yes, small | Define requirements |
| `/plan` | Yes, small | Define architecture |
| `/tasks` | Yes, small | Break into steps |
| `/implement` | Yes, per layer | Generate actual code |
