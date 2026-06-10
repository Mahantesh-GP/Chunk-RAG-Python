# Security Scan Report Comparison
## VulnDemo_Original — Report A vs Report B

**Prepared by:** Security Manager Review  
**Date:** June 10, 2026  
**Application:** VulnDemo_Original (V3) — .NET 8 ASP.NET Core Todo API  
**Purpose:** Compare two security scan outputs against the same codebase and determine which is more operationally useful, defensible, and actionable.

---

## 1. Report Identity

| Attribute | Report A — SoftPro Scan | Report B — Skill Exploitability Scan V1 |
|---|---|---|
| File | `VulnDemo_SecurityScan_Report.md` | `VulnDemo_SECURITY_SCAN_REPORT_V1 1.md` |
| Scan Date | June 9, 2026 | March 15, 2026 |
| Scan Engine | Security-scan skill (narrative mode) | Mythos Exploitability Pipeline (Stages 0→A→B→C→D→E) |
| Threat Model | Mythos ready — weaponize in hours | Mythos ready — CRITICAL FINDING PRESENT |
| Endpoints Scanned | 24 | Not explicitly stated |
| Vulnerable Endpoints | 21 of 24 | Not explicitly stated |
| Total Findings | **27** | **15** |
| False Positives Eliminated | Not tracked | **5 explicitly eliminated** |
| Report Version | V1 | V1 |

---

## 2. Finding Breakdown by Severity

| Severity | Report A (SoftPro) | Report B (Skill Scan) | Delta | Notes |
|---|---|---|---|---|
| CRITICAL | 6 | 3 | +3 in A | Report A counts more issues as Critical |
| HIGH | 9 | 10 | +1 in B | Roughly equivalent |
| MEDIUM | 12 | 2 | +10 in A | Major difference — Report A includes more medium-risk items |
| LOW | 0 | 0 | — | Neither found low-risk items |
| False Positives | 0 tracked | 5 eliminated | — | Report B is more disciplined |
| **TOTAL** | **27** | **15** | **+12 in A** | |

**Security Manager's Take:** The 12-finding gap is not necessarily a win for Report A. Report B explicitly eliminated 5 false positives and focuses only on confirmed exploitable issues. Report A's 12 extra MEDIUM findings likely include theoretical or context-dependent issues that would waste developer time investigating non-issues. Quality over quantity matters here.

---

## 3. Finding Coverage — Side-by-Side Mapping

Both reports cover the same codebase. Below is how individual findings map across reports.

| Vulnerability | Report A ID | Report B ID | In Both? |
|---|---|---|---|
| SQL Injection — Auth Bypass (Login) | F-001 | F-001 | ✅ Yes |
| SQL Injection — Full DB Read (Search) | F-002 | F-002 | ✅ Yes |
| No Authentication Middleware | F-003 | — | ⚠️ Only in A |
| SSRF via Webhook | F-004 | F-007 | ✅ Yes |
| Broken Access Control — Role from Header | F-005 | F-005 | ✅ Yes |
| Plaintext Passwords via List API | F-006 | — | ⚠️ Only in A |
| User-Controlled Role at Registration | F-007 | — | ⚠️ Only in A |
| Mass Assignment — Role/IsAdmin settable | F-008 | F-011 | ✅ Yes |
| Plaintext Password Storage | F-009 | — | ⚠️ Only in A |
| Path Traversal — Download | F-010 | F-006 | ✅ Yes |
| Weak Hash — MD5 | F-011 | F-014 (MEDIUM) | ✅ Yes |
| Weak Encryption — 3DES Hardcoded Key | F-012 | F-014 (MEDIUM) | ✅ Yes |
| XPath Injection | F-013 | F-009 | ✅ Yes |
| Unrestricted File Upload | F-014 | F-006/F-007 area | ✅ Partial |
| CORS Wildcard | F-015 | F-015 | ✅ Yes |
| Reflected XSS | F-016 | F-012 | ✅ Yes |
| Stored XSS in Comments | F-017 | F-012 | ✅ Yes |
| IDOR — No Ownership Check | F-018 | — | ⚠️ Only in A |
| Stack Trace in Error Response | F-019 | F-013 area | ✅ Partial |
| Debug Endpoint — Env Variables | F-020 | F-013 | ✅ Yes |
| Open Redirect — Password Reset | F-021 | — | ⚠️ Only in A |
| User Enumeration via Diff Response | F-022 | — | ⚠️ Only in A |
| HTTP Response Header Injection | F-023 | F-010 | ✅ Yes |
| Predictable Static Session Token | F-024 | F-001 compound | ✅ Partial |
| Swagger in Production | F-025 | — | ⚠️ Only in A |
| No Rate Limiting on Comments | F-026 | — | ⚠️ Only in A |
| No CSRF on State-Changing Operations | F-027 | — | ⚠️ Only in A |

**Findings only in Report A (8 unique):** F-003, F-006, F-007, F-009, F-018, F-021, F-022, F-025, F-026, F-027  
**Findings only in Report B (0 unique):** None — everything in B also appears in A

**Security Manager's Take:** Report B missed nothing critical that A found. Report A's unique findings (Swagger in prod, CSRF, rate limiting, IDOR, user enumeration) are real risks but largely infrastructure/config issues rather than code vulnerabilities — some are valid, but lower impact compared to the SQL injection chain that both catch.

---

## 4. Per-Finding Evidence Quality

This is the most critical difference between the two reports.

### Report A — Evidence Format (per finding)

```
F-001 — SQL Injection in Login
File: Controllers/AuthController.cs — Login()
cmd.CommandText = $"SELECT * FROM Users WHERE Username='{request.Username}'..."

If F-001 is exploited, the attacker can bypass authentication and gain full admin 
access. Blast radius: complete administrative control of the application.
```

**What you get:** File name, function name, vulnerable code line, blast radius description.  
**What you don't get:** Exact line numbers, CWE classification, CVSS score, working HTTP PoC request, fix code.

---

### Report B — Evidence Format (per finding)

```
F-002 ● CRITICAL — SQL Injection in Search (Full DB Exfiltration)
File: VulnDemoV3/VulnDemo.Api/Controllers/TodoController.cs
Lines: 16–23 (vulnerable line: 21) CWE: CWE-89

Evidence:
[HttpPost("search")]
public ActionResult Search([FromQuery] string keyword, [FromQuery] string category = "")
{
    // VULNERABLE: string concatenation - try: ' OR 1=1
    cmd.CommandText = $"SELECT * FROM Todos WHERE Title LIKE '%{keyword}%'...";

PoC — extract all Users table credentials via UNION:
GET /api/Todo/search?keyword=x%27 UNION SELECT 1,Username,Password,0,'','','','',''
FROM Users-- HTTP/1.1
Response includes rows: [{"Title":"admin", "Description":"admin123"}]

Fix:
cmd.CommandText = "SELECT * FROM Todos WHERE Title LIKE @kw AND Category LIKE @cat";
cmd.Parameters.AddWithValue("@kw", $"%{keyword}%");

CVSS Base Score: 9.8 (AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H)
```

**What you get:** Exact file path, exact line numbers, CWE number, vulnerable code block, working HTTP PoC that extracts real data, proof of data returned, fix code, CVSS score with full vector.  
**What you don't get:** Business impact narrative (brief only).

---

### Evidence Quality Scorecard

| Evidence Element | Report A | Report B | Winner |
|---|---|---|---|
| File name | ✅ | ✅ | Tie |
| Exact line numbers | ❌ | ✅ | **B** |
| CWE classification | ❌ | ✅ | **B** |
| Vulnerable code snippet | ✅ Partial | ✅ Full | **B** |
| Working PoC HTTP request | ❌ | ✅ | **B** |
| Proof of exploit (actual response) | ❌ | ✅ | **B** |
| Fix code per finding | ❌ | ✅ | **B** |
| CVSS score | ❌ | ✅ | **B** |
| CVSS vector string | ❌ | ✅ | **B** |
| Blast radius description | ✅ | ❌ | **A** |
| Business impact language | ✅ | ❌ Minimal | **A** |
| False positive status | ❌ | ✅ | **B** |

**Evidence Winner: Report B — 9 vs 2**

---

## 5. Attack Chains

Both reports include multi-step attack chains. Quality comparison:

### Report A — Chain 1 (Complete Takeover in Under 2 Minutes)
```
Step 1: POST /api/auth/login with ' OR '1'='1' — get TOKEN_1_Admin
Step 2: GET /api/user/list — returns all usernames + plaintext passwords
Step 3: Construct TOKEN_{n}_Admin for any user ID — impersonate anyone
No credentials. No tooling. No insider knowledge. Only the API hostname.
```
- ✅ Step-by-step HTTP with actual payloads
- ✅ Plain English framing ("No credentials, no tooling")
- ✅ Business impact clearly stated
- ❌ No CVSS chain score

### Report B — Chain 1 (Unauthenticated Full Compromise, F-004 + F-001)
```
Step 1: GET /api/user/click — returns admin:adminXXX in cleartext (F-004)
Step 2: POST /api/auth/login {"username":"admin","password":"adminXXX"} — 
        TOKEN_1_Admin obtained (token is guessable anyway)
Step 3: DELETE /api/user/5 with header role:Admin — deletes any user (F-005)
```
- ✅ Finding IDs referenced (cross-linked to evidence)
- ✅ Shows compound risk
- ❌ Less narrative explanation
- ❌ Not all chains have explicit HTTP payloads written out

### Attack Chain Scorecard

| Chain Feature | Report A | Report B |
|---|---|---|
| Number of chains | 5 | 5 |
| Step-by-step HTTP payloads | ✅ | ✅ Partial |
| Cross-references to finding IDs | ❌ | ✅ |
| Business narrative per chain | ✅ | ❌ Minimal |
| Cloud/infra takeover chain | ✅ Chain 3 (SSRF→IMDS) | ✅ Referenced |
| RCE chain | ✅ Chain 4 (File Upload) | ✅ Referenced |
| Architectural amplifiers table | ✅ | ❌ |

**Chain Winner: Tie — A wins on narrative, B wins on cross-linking**

---

## 6. Sections Unique to Each Report

### Sections Only in Report A

**Executive Summary — "What Breaks First"**  
Written in plain English for non-technical leadership. Starts with: *"Authentication breaks first — in under 30 seconds with a single HTTP POST."* Exactly the kind of hook that makes a CISO pay attention.

**Top 5 Critical Risks (Ranked by Speed to Exploit)**  
A prioritized table showing File, Time-to-Exploit. Example: *"SQL injection in login — AuthController.cs — < 30 seconds."* Extremely useful for sprint planning and triaging with product owners.

**Patch-Independent Containment Actions**  
This is a standout section. Lists infra-level blocks that stop attacks without any code change — WAF rules, firewall blocks, token rotation, CORS restriction. Critically useful for the hours between discovering a vulnerability and deploying a fix.

**exec-risk-brief.md — Leadership Summary**  
Four paragraphs: What breaks first, How quickly does compromise spread, Why current controls are insufficient, First containment moves if patching slips. This is a CTO/CISO document. Report B has nothing equivalent.

**Architectural Amplifiers Table**  
Maps each amplifier (No auth middleware, CORS wildcard, Plaintext passwords, etc.) to its cascading effect. Shows that removing one amplifier collapses multiple attack paths simultaneously.

**Scan Coverage Table**  
Shows every file scanned, every endpoint/function scanned. Gives stakeholders confidence that nothing was missed and provides an audit trail.

---

### Sections Only in Report B

**CVSS Scores with Full Vector Strings**  
Every finding has a CVSS 3.1 base score (e.g., 9.8 AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H). Required for SOC 2, ISO 27001, and any formal risk register. Report A has none.

**CWE Classifications**  
Every finding maps to a CWE number (CWE-89, CWE-22, CWE-79, etc.). Required for formal vulnerability management systems (Jira Security, DefectDojo, etc.) and compliance reporting.

**Working PoC HTTP Requests with Actual Responses**  
Not theoretical — Report B shows the actual HTTP request and the actual data returned. This is the difference between a finding a developer debates and one they cannot dismiss.

**Fix Code Per Finding**  
Every finding includes the exact parameterized or sanitized replacement code. Developer picks it up, applies it, done. No research required.

**Sprint/Backlog Prioritization**  
Findings split into P1 (This Sprint), P2 (Next Sprint), and Backlog. Maps to how engineering teams actually work.

**Stage Compliance Evidence (GATE 1–6)**  
Shows proof that each compliance gate was validated: GATE 1 (SAST/DUPE exploit), GATE 2 (Strict Sequence), GATE 3 (Checklist), GATE 4 (No Hedging), GATE 5 (Full Coverage), GATE 6 (Patch). Essential for audit documentation.

**False Positive Elimination Record**  
5 findings were investigated and ruled out. This matters — it means the 15 remaining findings are confirmed, not suspected. A finding list with no false positive tracking cannot make the same claim.

**Mythos Readiness Sign-Off Criteria**  
Checklist of what must be true before the report is considered production-grade.

---

## 7. Who Should Use Which Report

| Audience | Report A | Report B | Recommendation |
|---|---|---|---|
| CTO / CISO | ✅ Executive brief, business language | ❌ Technical only | **Use A** |
| Security Manager | ✅ Chain narrative, amplifiers | ✅ CVSS, gates, PoC | **Use Both** |
| Developer fixing the bug | ❌ No line numbers, no fix code | ✅ Line numbers, PoC, fix | **Use B** |
| Penetration tester / auditor | ❌ No PoC, no CVSS | ✅ Full evidence chain | **Use B** |
| Compliance / SOC2 / ISO27001 | ❌ No CWE, no CVSS | ✅ CWE + CVSS vectors | **Use B** |
| Sprint planning / product owner | ✅ Speed-to-exploit table | ✅ P1/P2/Backlog split | **Use Both** |
| Incident response (fire drill) | ✅ Containment actions (no code change) | ❌ No containment section | **Use A** |
| Risk register entry | ❌ No CVSS | ✅ CVSS required | **Use B** |

---

## 8. Defensibility Assessment

If this report goes to an external auditor, client, or regulatory body, which holds up better?

| Defensibility Criterion | Report A | Report B |
|---|---|---|
| Every finding has proven PoC | ❌ | ✅ |
| False positives tracked | ❌ | ✅ |
| Severity backed by CVSS standard | ❌ | ✅ |
| CWE references for vulnerability taxonomy | ❌ | ✅ |
| Exact file + line number | ❌ | ✅ |
| Can be imported into DefectDojo / Jira Security | ❌ Missing fields | ✅ All required fields |
| Audit trail (scan tracking, stages) | ❌ | ✅ |

**Defensibility Winner: Report B — unambiguously**

A finding without a PoC is an allegation. A finding with a working PoC and actual database response is a fact.

---

## 9. Completeness Assessment

| Completeness Criterion | Report A | Report B |
|---|---|---|
| More findings discovered | ✅ 27 vs 15 | — |
| Infra/config findings (Swagger, CSRF, rate limiting) | ✅ | ❌ |
| IDOR / ownership checks | ✅ F-018 | ❌ |
| User enumeration | ✅ F-022 | ❌ |
| Open redirect | ✅ F-021 | ❌ |
| Business impact per finding | ✅ | ❌ |
| Cloud-specific risk (Azure IMDS) | ✅ | ✅ |

**Completeness Winner: Report A** — it found more surface-level issues that are real, even if lower risk.

---

## 10. Overall Verdict — Security Manager's Decision

### Score Summary

| Category | Report A | Report B | Winner |
|---|---|---|---|
| Finding count | 27 | 15 | A (but quality matters more) |
| Evidence quality per finding | 2/12 | 9/12 | **B** |
| Attack chain quality | Narrative | Cross-linked + PoC | Tie |
| Executive communication | ✅ Strong | ❌ Absent | **A** |
| Developer usability | ❌ Low | ✅ High | **B** |
| Compliance / audit readiness | ❌ Fails | ✅ Passes | **B** |
| False positive discipline | ❌ | ✅ | **B** |
| Incident response guidance | ✅ | ❌ | **A** |
| Completeness of discovery | ✅ | Partial | **A** |

---

### Final Recommendation

> **Neither report alone is sufficient. Report B is technically superior. Report A is communicatively superior. The right answer is to use Report B as the primary security deliverable and extract Report A's executive brief + containment table as a separate leadership-facing document.**

**If forced to choose one:** Choose **Report B**.

Here is why. When a finding has no PoC, no CVSS, and no line number, a developer will challenge it, a manager will deprioritize it, and an auditor will reject it. Report B cannot be argued with — the SQL injection response literally shows `admin:admin123` in the payload. That evidence forces action. Report A describes the same issue in four sentences and a developer can say "we'll look at it next sprint."

Security findings that do not get fixed are worthless regardless of how many you found. Report B's 15 proven findings will get fixed faster than Report A's 27 described ones.

**What to do with Report A's unique findings (F-003, F-006, F-007, F-009, F-018, F-021–F-027):**  
Feed them into Report B's format. Write PoC for each, assign CVSS, get line numbers. Then they become actionable. Until then, treat them as a backlog of suspected issues pending confirmation.

---

## 11. Recommended Hybrid Report Structure

For the next scan cycle, a combined report should include:

1. **exec-risk-brief** (from Report A's format) — 1 page, C-suite language, "What breaks in 30 seconds"
2. **Top 5 by Speed-to-Exploit** (from Report A) — table with Time-to-Exploit
3. **Per-finding detail** (from Report B's format) — File, Line, CWE, Code, PoC, CVSS, Fix
4. **Attack chains** (Report A's narrative + Report B's finding cross-links)
5. **Architectural Amplifiers** (from Report A) — high-leverage architectural fixes
6. **Patch-Independent Containment** (from Report A) — infra actions, no code change
7. **Sprint/Backlog split** (from Report B) — P1/P2/Backlog
8. **Stage Compliance Gates** (from Report B) — audit evidence
9. **False Positive log** (from Report B) — what was investigated and cleared
10. **Scan Coverage table** (from Report A) — audit trail of what was scanned

This hybrid format serves every stakeholder: the CISO gets the exec brief, the developer gets line numbers and fix code, the auditor gets CVSS and PoC, and the incident responder gets the containment table.

---

*Report comparison prepared for internal security review. Both source reports cover VulnDemo_Original (V3) — the same ASP.NET Core codebase scanned on different dates with different tooling.*
