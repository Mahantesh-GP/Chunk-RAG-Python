# Master Prompt — Horizontal Consolidated Vulnerability Report Generator

**Purpose:** Use this prompt with GitHub Copilot Agent to generate a horizontal
consolidated vulnerability report from multiple scanner sources.  
**Author:** Mahantesh G | EA GenAI Team  
**Date:** June 2026  
**Version:** V1

---

## How to Use

1. Open **GitHub Copilot Agent** in VS Code
2. Attach **all scanner output files** (CSV or MD)
3. Copy the prompt below and paste it into Copilot Agent chat
4. Agent will generate `horizontal-consolidation-report.md`

---

## Supported Input Sources

| Scanner | Format | Notes |
|---------|--------|-------|
| Fortify | CSV | Columns: VulnId, Category, File, Line, Source, Sink |
| Mend SAST | CSV | Columns: sink, dataflow, deeplink, category |
| Skill-based Exploitation Validator | Markdown | Summary Table format |
| SpecKit | Markdown | Vulnerability blocks with severity/title/location |

---

## THE PROMPT — Copy Everything Below This Line

---

```
You are a security report consolidation agent.

I will provide you multiple vulnerability scan output files from different 
scanners. Your job is to generate a single horizontal consolidated vulnerability 
report in markdown format.

## Input Sources
Process ALL files I provide. Each file is from one of these scanners:
- Fortify (CSV format)
- Mend SAST (CSV format)
- Skill-based Exploitation Validator (markdown format)
- SpecKit (markdown format)

## Output
Generate a single file called: horizontal-consolidation-report.md

---

## REPORT STRUCTURE

---

### SECTION 1: Legend

Include this at the very top of the report:

| Symbol | Meaning |
|--------|---------|
| ✅ | Scanner flagged this issue |
| ❌ | Scanner did NOT flag this issue |
| ~~strikethrough~~ | Confirmed False Positive |

Confidence Score:
- 4/4 → Fix Immediately (all scanners agree)
- 3/4 → Fix (high confidence)
- 2/4 → Review before fixing
- 1/4 → Possible False Positive — verify first

---

### SECTION 2: Horizontal Matrix

Create a table with these exact columns:
| # | Severity | Issue Category | Location | Fortify | Mend SAST | Skill Validator | Speckit | Confidence | Action |

RULES:
- One row per unique issue family (deduplicated by OWASP meaning)
- Use ✅ if scanner flagged this issue, ❌ if not
- Apply ~~strikethrough~~ for confirmed false positives
- Confidence score: 4/4 = Fix Immediately | 3/4 = Fix | 2/4 = Review | 1/4 = Possible FP
- Action column values: "Fix Immediately" / "Fix" / "Review" / "Possible FP"
- Assign row IDs: H-001, H-002, H-003...
- Each H-ID must be a clickable link that jumps to the matching row in Summary Table
- SORT ORDER:
  1. Severity: Critical first, then High, Medium, Low
  2. Within same severity: Category (alphabetical)
  3. Within same category: Location (alphabetical)

---

### SECTION 3: Summary Table

Create a table with these exact columns:
| ID | Severity | Category | Location | Description | Scanner |

RULES:
- Assign sequential IDs: H-001, H-002... matching Horizontal Matrix
- Each ID row must have an internal HTML anchor: <a id="h-001"></a>
- IDs in Horizontal Matrix must be clickable links jumping to this anchor
- Include ALL findings from ALL scanners (do NOT deduplicate here)
- Keep each scanner finding as a separate row
- Mark confirmed false positives with ~~strikethrough~~ on entire row
- Same sort order as Horizontal Matrix

---

### SECTION 4: Confirmed False Positives / Strikeouts

List items that are:
- Exact duplicate rows (same issue, same file, same line number)
- Confirmed false positives with clear evidence

Format each entry as:
~~H-XXX — [Category] — [Description]~~
**Reason:** [Why this is a false positive or duplicate]

---

### SECTION 5: Full Consolidated Findings (F-001 to F-N)

For EVERY finding in the Summary Table, create a detailed entry block:

#### <a id="f-001"></a> F-001 — [Category] — [Severity]

| Field | Value |
|-------|-------|
| **File** | filename.cs:line |
| **Description** | What the vulnerability is |
| **Scanner** | Which tool found it |
| **Evidence** | Source/sink details from scanner output |
| **Recommendation** | How to fix it |

Repeat for every finding F-001 through F-N.

---

## PARSING RULES PER SCANNER

### Fortify CSV
- Extract columns: VulnId, Category, File, Line, Source, Sink
- Map Category to OWASP family
- Each row = one finding

### Mend SAST CSV  
- Extract columns: sink, dataflow, deeplink, category
- Map category to OWASP family
- Each row = one finding

### Skill-based Exploitation Validator (Markdown)
- Parse the Summary Table in the markdown
- Extract: Severity, Category, Location, Description
- Label scanner as: "Skill-based Exploitation Validator"

### SpecKit (Markdown)
- Parse vulnerability blocks with fields: severity, title, location, description, task ID
- Label scanner as: "Speckit"

---

## OWASP GROUPING RULES

Group these as the same family in Horizontal Matrix:
- SQL Injection → A03 Injection
- XPath Injection → A03 Injection  
- XSS / Stored XSS / Reflected XSS → A03 Injection (XSS)
- Path Traversal / Path Manipulation → A01 Broken Access Control
- IDOR / Broken Access Control → A01 Broken Access Control
- Hardcoded Credentials / Hardcoded Password → A07 Auth Failures
- Plaintext Password Storage → A02 Cryptographic Failures
- Weak Cryptography (MD5/3DES) → A02 Cryptographic Failures
- SSRF → A10 SSRF
- Security Misconfiguration / Debug Endpoint / CORS → A05 Security Misconfiguration
- Mass Assignment / Sensitive Field Exposure → A04 Insecure Design
- User Enumeration / Host Header Injection → A07 Auth Failures
- Error Messages / Stack Trace Exposure → A05 Security Misconfiguration
- Insecure Transport → A02 Cryptographic Failures
- No Rate Limiting / CSRF → A01 Broken Access Control

---

## INTERNAL LINKING RULES

1. Every finding ID (H-001, F-001 etc.) must use internal markdown anchors
2. Format for anchor: `<a id="h-001"></a>`
3. Format for link: `[H-001](#h-001)`
4. Clicking any ID in any section must jump to that exact row in the same file
5. Do NOT use external file links — everything must be internal to one file

---

## ADDITIONAL OUTPUT

Also generate a PowerShell script file: generate-handoff-report.ps1

The script must:
1. Accept paths to all scanner input files as parameters
2. Parse each file according to the rules above
3. Generate the horizontal-consolidation-report.md
4. Sort findings by Severity → Category → Location
5. Assign H-IDs sequentially
6. Print summary at end: total findings, findings per scanner, confidence breakdown

---

## FINAL CHECKLIST BEFORE DELIVERING

Verify the report contains:
- [ ] Legend section at top
- [ ] Horizontal Matrix with ✅/❌ per scanner
- [ ] Confidence scores (1/4 to 4/4)
- [ ] Action column (Fix Immediately / Fix / Review / Possible FP)
- [ ] Summary Table with all findings
- [ ] Internal anchor links working on all IDs
- [ ] Strikeout section for false positives
- [ ] Full consolidated findings F-001 to F-N
- [ ] Sorted by Severity → Category → Location

Now process the attached files and generate the report.
```

---

## Notes for Future Use

- Add new scanner sources by adding a new parsing rule section in the prompt
- Change sort order by modifying the SORT ORDER section
- Add new OWASP mappings in OWASP GROUPING RULES section
- The PowerShell script `generate-handoff-report.ps1` can be reused for automation

---

## Example Files to Attach

```
Fortify output:     FNF-Sandbox_EShop-POC_TODOSanbox_Scan-CodeFindings.csv
Mend SAST output:   IssuesExport-VX-1-20260520.csv
Skill Validator:    VulnDemo_SECURITY_SCAN_REPORT_V1.md
SpecKit review:     security-review-2026-05-20.md
```

---
*EA GenAI Team | Vulnerability Automation Project | 2026*
