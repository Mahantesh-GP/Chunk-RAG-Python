# Tool Capability Assessment Report
## "How Good Is the Mythos Security Scanner?"
**Prepared for:** Engineering Manager
**Prepared by:** Fix Verification Engineer
**Date:** 2026-05-20
**Context:** Evaluating the Mythos Exploitation-Validator Pipeline (LLM + Skill-based tool)
**Codebase Tested On:** dotnet/eShop (341 C# files, .NET 8, Microservices)

---

## Bottom Line Up Front (For Manager)

> The tool is performing at a **strong mid-level to senior security engineer
> level** for detection. For fix generation, it is at a **junior-to-mid level**
> — fixes are directionally correct but require a human verification pass before
> applying. It is NOT doing extraordinary work, but it is doing genuinely useful
> work that would take a human engineer significantly longer.

**Recommended position to management:**
The tool is production-viable as a **first-pass scanner and fix suggester**,
NOT as an autonomous fix applier. A human verification step is required
before any fix goes to code review.

---

## What My Verification Job Covers

As the fix verifier, I checked each tool-generated fix for:

| Check | What I looked for |
|-------|------------------|
| Compile errors | Wrong types, missing using statements, wrong return types |
| Breaking changes | API signature changes that break callers |
| Flow breaks | Logic errors — null ref, wrong order of checks, infinite loops |
| Missing dependencies | New services injected but not registered in DI |
| Incomplete coverage | Fix solves one endpoint but misses another |
| Platform fit | Is the fix idiomatic for .NET 8 / Azure / Aspire? |

---

## Fix-by-Fix Verification Results

---

### F-001 — IDOR Fix
**Compile Issues:** None
**Breaking Changes:** Minor — return type signature changed
**Flow Issues:** None — logic order is correct

**Detail:**
The fix changes `GetOrderAsync` return type from:
```csharp
Task<Results<Ok<Order>, NotFound>>
```
to:
```csharp
Task<Results<Ok<Order>, NotFound, ForbidHttpResult>>
```
This is a **breaking change on the method signature** — any caller that
pattern-matches on the result type will need updating. However in Minimal
API style (which eShop uses), this is handled automatically by the framework.
No manual caller update needed.

The null check before ownership check is **intentional and correct**:
```csharp
if (order == null) return NotFound();       // Step 1 — correct
if (order.BuyerGuid != callerId) return Forbid(); // Step 2 — correct
```
Reversing these would cause a NullReferenceException. Tool got the order right.

**Verdict: Safe to apply. Minor signature change is framework-handled.**

---

### F-002 — Catalog Auth Fix
**Compile Issues:** Potential runtime error
**Breaking Changes:** Yes — all catalog writes become 401 for unauthenticated callers
**Flow Issues:** One gap

**Detail:**
The fix adds `.RequireAuthorization("admin")`. This compiles fine. BUT — if
the `"admin"` policy is not registered in `Program.cs` or `ServiceDefaults`,
the app will throw this at runtime:
```
InvalidOperationException: The AuthorizationPolicy named 'admin' was not found.
```
The tool did NOT include the policy registration code. This is a **deployment
risk** — the fix compiles but crashes at startup if policy is missing.

Required addition (not in tool's fix):
```csharp
builder.Services.AddAuthorizationBuilder()
    .AddPolicy("admin", policy => policy.RequireRole("admin"));
```

**Verdict: Needs one additional line before it is safe to deploy.**

---

### F-003 — SSRF Fix
**Compile Issues:** None
**Breaking Changes:** Yes — legitimate webhook URLs on private networks will
now be rejected
**Flow Issues:** Two logic gaps

**Detail:**

**Logic Gap 1 — Incomplete IP range:**
```csharp
host.StartsWith("172.16.") // Tool wrote this
```
RFC1918 `172.16.0.0/12` covers `172.16.x` through `172.31.x`.
The tool's check only blocks `172.16.x` — `172.17.x` through `172.31.x`
passes through. A `172.20.x` internal service is not blocked.

**Logic Gap 2 — Azure metadata endpoint missing:**
`169.254.169.254` is the Azure Instance Metadata Service (IMDS).
An attacker can use SSRF to call it and get managed identity tokens.
The tool's fix does not block this address. On Azure this is critical.

**Logic Gap 3 — Stored URLs not re-validated:**
The fix validates at subscription time. But `WebhooksSender.cs` uses
the stored `DestUrl` directly on every event. Webhooks registered
before this fix was deployed bypass the check entirely.

**Verdict: Do NOT apply as-is on Azure. Needs 3 additions before deployment.**

---

### F-004 — JWT Audience Fix
**Compile Issues:** None — it is a line deletion
**Breaking Changes:** Potentially yes — services that were accepting
cross-service tokens will now reject them
**Flow Issues:** None

**Detail:**
This is the simplest possible fix — remove one line:
```csharp
// DELETE this line
options.TokenValidationParameters.ValidateAudience = false;
```
No new code. No new dependencies. No signature changes.

The "breaking change" here is intentional and correct — services SHOULD
reject tokens not intended for them. Any integration test that was
passing a basket token to the orders service will now correctly fail.

**Confirmed:** This exact fix exists as open PR #808 in the real
dotnet/eShop GitHub repository. The tool independently reached the
same conclusion as the Microsoft team.

**Verdict: Safest fix in the report. Apply immediately.**

---

### F-005 — Hardcoded Secret Fix
**Compile Issues:** Possible NullReferenceException
**Breaking Changes:** Yes — app will not start without secrets configured
**Flow Issues:** One gap

**Detail:**
```csharp
new Secret(configuration["Clients:Maui:Secret"].Sha256())
```
If `configuration["Clients:Maui:Secret"]` returns null (key not set),
calling `.Sha256()` on null throws a NullReferenceException at startup.
Safe version:
```csharp
var secret = configuration["Clients:Maui:Secret"]
    ?? throw new InvalidOperationException("OIDC secret not configured");
new Secret(secret.Sha256())
```

Also — for Azure, the right approach is Key Vault, not environment
variables. The tool's fix works but is not the Azure-native pattern.

**Verdict: Fix direction is correct but needs null guard before applying.**

---

### F-006 — Seed Credentials Fix
**Compile Issues:** None
**Breaking Changes:** None — only affects startup in Production
**Flow Issues:** None

**Detail:**
```csharp
if (!env.IsProduction())
    await SeedAsync(app);
```
Clean, simple, correct. `IWebHostEnvironment` is already injected
in `Program.cs` in this codebase. No new dependency needed.
No compile issue. No flow issue.

**Verdict: Apply as-is. Zero risk.**

---

## Summary Verification Table

| Finding | Compiles? | Runtime Safe? | Logic Correct? | Apply As-Is? |
|---------|-----------|--------------|----------------|--------------|
| F-001 IDOR | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| F-002 Catalog Auth | ✅ Yes | ⚠️ Needs policy registration | ✅ Yes | ⚠️ One addition needed |
| F-003 SSRF | ✅ Yes | ❌ Logic gaps | ❌ Incomplete ranges | ❌ Extend first |
| F-004 JWT | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| F-005 Secret | ⚠️ Null risk | ⚠️ Needs null guard | ✅ Yes | ⚠️ Add null guard |
| F-006 Seed | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |

**3 fixes: Apply immediately**
**2 fixes: Apply with minor addition**
**1 fix: Extend before applying (F-003)**

---

## Honest Assessment: What Level Is This Tool?

### Detection Level: Senior Engineer (8.5/10)

| Capability | Assessment |
|-----------|-----------|
| Found real vulnerabilities | ✅ All 6 confirmed against source |
| Eliminated false positives | ✅ 4 dismissed with specific technical reasoning |
| Understood codebase context | ✅ Used existing abstractions, not generic advice |
| Produced PoC exploits | ✅ Working HTTP payloads shown for SSRF, JWT, IDOR |
| Cited exact file + line | ✅ Every finding has file path and line numbers |
| Cross-file analysis | ✅ Traced IDOR across OrdersApi → OrderQueries → CommandHandler |

This is NOT a simple pattern matcher. A pattern matcher flags every
`SELECT *` or every missing `[Authorize]`. This tool traced a 3-file
IDOR chain, understood that `GetOrdersByUserAsync` is safe while
`GetOrderAsync` is not, and explained WHY. That is meaningful analysis.

### Fix Generation Level: Junior-to-Mid Engineer (6.5/10)

| Capability | Assessment |
|-----------|-----------|
| Fix direction correct | ✅ All 6 point the right way |
| Uses existing codebase patterns | ✅ Did not invent new abstractions |
| Handles edge cases | ⚠️ Mixed — good on IDOR, weak on SSRF |
| Platform-aware (Azure) | ❌ Generic fixes, not Azure-native |
| DI registration awareness | ❌ Missed policy registration for F-002 |
| Null safety | ⚠️ Missed null guard on F-005 |
| Complete coverage | ❌ F-003 has 3 gaps |

The fixes read like code written by someone who knows the vulnerability
class well but has not personally shipped .NET 8 on Azure before.
Correct in structure, weak in production-hardening details.

### Overall Tool Rating

```
Detection Accuracy:     ████████░░  8.5/10  — Strong, trust the findings
Fix Quality:            ██████░░░░  6.5/10  — Use as starting point, verify before apply
False Positive Rate:    ████████░░  8/10    — 4 false positives correctly dismissed
Platform Awareness:     █████░░░░░  5/10    — Generic .NET, not Azure-specific
Production Readiness:   ███████░░░  7/10    — 3 of 6 fixes apply as-is
```

**Overall: 7/10 — Genuinely useful, not extraordinary**

---

## What This Tool Is vs What It Is Not

### What it IS
- A reliable first-pass security scanner
- A time-saver — finding these 6 issues manually across 341 files
  would take a human security engineer 1-2 days minimum
- A good starting point for fix implementation
- A documentation generator — the report format with CWE, CVSS,
  evidence, and PoC is production-quality

### What it is NOT
- An autonomous fix applier — human verification is required
- A replacement for a security engineer — it misses platform-specific
  depth (Azure IMDS, Key Vault patterns)
- Extraordinary — the gaps in F-002, F-003, F-005 are things a senior
  .NET Azure engineer would not miss
- A complete security audit — it covers OWASP Top 10 style issues but
  would not catch business logic flaws, race conditions, or
  infrastructure-level misconfigurations

---

## Recommendation to Manager

### Short version
> "The tool finds real bugs correctly and saves significant time.
> Its fixes need a human verification pass — 3 are apply-ready,
> 2 need minor additions, 1 needs meaningful extension.
> It is a strong assistant, not an autonomous engineer."

### For tooling investment decision
- **Worth using:** Yes — detection quality alone justifies it
- **Trust level:** High for detection, Medium for fixes
- **Required process:** Every fix must go through human verification
  before PR — this report demonstrates what that looks like
- **Gap to fill:** Azure-specific guidance (Key Vault, IMDS, Aspire
  service mesh trust) should be added to the tool's skill/prompt context
  to improve fix quality from 6.5 to 8+

---

*Assessment based on independent verification of Mythos Scan Report V1
against dotnet/eShop source code — May 2026*
