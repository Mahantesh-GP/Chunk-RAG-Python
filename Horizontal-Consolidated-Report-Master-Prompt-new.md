# Master Prompt — Horizontal Consolidated Vulnerability Report Generator

**Purpose:** Use this prompt with GitHub Copilot Agent to generate a horizontal
consolidated vulnerability report from ANY number of scanner sources.  
**Author:** Mahantesh G | EA GenAI Team  
**Date:** June 2026  
**Version:** V2 — Generic (supports any scanner, any format)

---

## How to Use

1. Open **GitHub Copilot Agent** in VS Code
2. Attach **all scanner output files** (any format — CSV, JSON, MD, XML, TXT)
3. Copy the prompt below and paste into Copilot Agent chat
4. Agent will auto-detect scanners and generate `horizontal-consolidation-report.md`

---

## Supported Input Formats

| Format | Examples |
|--------|---------|
| CSV | Fortify, Mend SAST, Checkmarx, SonarQube, Snyk exports |
| Markdown (.md) | Mythos/Skill Validator reports, SpecKit reviews, manual reviews |
| JSON | Any scanner JSON export |
| XML | SARIF format, FindBugs XML, PMD XML |
| TXT | Any plain text vulnerability report |
| PDF text extract | Any scanner PDF converted to text |

> **Any new scanner can be added** — the agent auto-detects column/field names
> and maps them to the standard report format.

---

## THE PROMPT — Copy Everything Between The Lines

---

```
You are a security report consolidation agent.

I will provide you multiple vulnerability scan output files. Each file may come
from a DIFFERENT scanner tool. Your job is to:

1. AUTO-DETECT which scanner produced each file
2. AUTO-DETECT the format (CSV, JSON, Markdown, XML, TXT)
3. Extract all vulnerability findings from every file
4. Generate a single horizontal consolidated vulnerability report

## AUTO-DETECTION RULES

For each file I attach:

STEP 1 — Detect format:
- If file ends in .csv → parse as CSV, extract headers from first row
- If file ends in .md or .markdown → parse as Markdown, look for tables and blocks
- If file ends in .json → parse as JSON, detect vulnerability array/objects
- If file ends in .xml → parse as XML/SARIF, extract results/rules
- If file ends in .txt → parse as plain text, extract structured findings

STEP 2 — Detect scanner name:
- Look at filename for scanner name hints (e.g. "Fortify", "Mend", "Checkmarx", "Snyk", "SonarQube", "SpecKit", "Mythos", "Skill")
- Look at file content for scanner signatures (e.g. column headers, metadata fields)
- If scanner cannot be detected from filename or content, label it as "Scanner-N" (e.g. Scanner-1, Scanner-2)

STEP 3 — Extract fields from each file:
Extract as many of these fields as available:
- Severity (Critical / High / Medium / Low / Info)
- Category or Issue Type (SQL Injection, XSS, Path Traversal etc.)
- File / Location (filename and line number if available)
- Description (what the vulnerability is)
- Evidence (source, sink, dataflow, stack trace etc.)
- Recommendation (how to fix)
- Scanner name (detected in Step 2)
- Any additional metadata available

STEP 4 — Normalize severity:
Map any severity scale to: Critical / High / Medium / Low / Info
Examples:
- "Blocker" / "P1" / "CRITICAL" → Critical
- "Major" / "P2" / "ERROR" → High  
- "Minor" / "P3" / "WARNING" → Medium
- "Info" / "P4" / "NOTE" → Low

---

## OUTPUT

Generate a SINGLE file: horizontal-consolidation-report.md

---

## REPORT STRUCTURE

---

### SECTION 1: Report Header

Include:
- Project name (detect from filenames or ask me)
- Date generated
- List of all scanner sources detected with file names
- Total findings count per scanner
- Grand total findings

---

### SECTION 2: Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Scanner flagged this issue |
| ❌ | Scanner did NOT flag this issue |
| ~~strikethrough~~ | Confirmed False Positive |

Confidence Score (based on how many scanners flagged same issue):
- 4/4 or all scanners agree → Fix Immediately
- 3/4 → Fix (high confidence)
- 2/4 → Review before fixing
- 1/4 → Possible False Positive — verify first
- If only 1 scanner total provided → use description quality to assign confidence

---

### SECTION 3: Horizontal Matrix

Create a table with these columns:
| # | Severity | Issue Category | Location | [Scanner-1] | [Scanner-2] | [Scanner-N...] | Confidence | Action |

RULES:
- Column headers for scanners = auto-detected scanner names from Step 2
- Add one scanner column per unique scanner detected
- One row per UNIQUE issue family (deduplicated by OWASP meaning)
- Use ✅ if that scanner flagged this issue family, ❌ if not
- Apply ~~strikethrough~~ for confirmed false positives
- Assign row IDs: H-001, H-002... as clickable links to Summary Table
- ACTION values: "Fix Immediately" / "Fix" / "Review" / "Possible FP"
- SORT ORDER:
  1. Severity: Critical → High → Medium → Low → Info
  2. Within same severity: Category (alphabetical)
  3. Within same category: Location (alphabetical)

---

### SECTION 4: Summary Table

| ID | Severity | Category | Location | Description | Scanner |

RULES:
- Assign IDs: H-001, H-002... with internal HTML anchors <a id="h-001"></a>
- Include ALL findings from ALL scanners (do NOT deduplicate)
- Keep each scanner finding as separate row
- Mark confirmed false positives with ~~strikethrough~~
- Same sort order as Horizontal Matrix

---

### SECTION 5: Confirmed False Positives / Strikeouts

List items that are exact duplicates or confirmed false positives:

~~H-XXX — [Category] — [Description]~~
**Reason:** [Why this is FP or duplicate]

---

### SECTION 6: Full Consolidated Findings

For EVERY finding create a detailed block:

#### <a id="f-001"></a> F-001 — [Category] — [Severity]

| Field | Value |
|-------|-------|
| **File** | filename:line |
| **Description** | What the vulnerability is |
| **Scanner** | Which tool found it |
| **Evidence** | All evidence fields extracted from scanner |
| **Recommendation** | How to fix |

---

## OWASP GROUPING RULES (for deduplication in Horizontal Matrix)

Group these as same family:
- SQL Injection, XPath Injection, LDAP Injection → A03 Injection
- XSS, Stored XSS, Reflected XSS, DOM XSS → A03 Injection (XSS)
- Path Traversal, Path Manipulation, Directory Traversal → A01 Broken Access Control
- IDOR, Missing Authorization, Broken Access Control → A01 Broken Access Control
- Hardcoded Credentials, Hardcoded Password, Hardcoded Token → A07 Auth Failures
- Plaintext Password, Weak Password Storage → A02 Cryptographic Failures
- Weak Cryptography (MD5, 3DES, SHA1, RC4) → A02 Cryptographic Failures
- SSRF, Server-Side Request Forgery → A10 SSRF
- Security Misconfiguration, Debug Endpoint, CORS, Swagger → A05 Misconfiguration
- Mass Assignment, Sensitive Field Exposure, Over-posting → A04 Insecure Design
- User Enumeration, Host Header Injection → A07 Auth Failures
- Error Messages Exposure, Stack Trace Exposure → A05 Misconfiguration
- Insecure Transport, Missing TLS, Unencrypted → A02 Cryptographic Failures
- No Rate Limiting, CSRF, Missing Anti-forgery → A01 Broken Access Control
- Vulnerable Component, Outdated Library → A06 Vulnerable Components
- Insecure Deserialization → A08 Insecure Deserialization
- Logging Failures, Missing Audit Log → A09 Logging Failures

If a finding does not match any above → keep original category as-is.

---

## INTERNAL LINKING RULES

- Every ID must use internal anchors: `<a id="h-001"></a>`
- Every ID reference must be clickable link: `[H-001](#h-001)`
- Clicking any ID anywhere in the file must jump to that exact row
- Do NOT use external file links — single file only

---

## ADDITIONAL OUTPUT

Also generate: generate-handoff-report.ps1

Script requirements:
1. Accept any number of scanner input file paths as parameters
2. Auto-detect format and scanner name per file (same logic as above)
3. Parse all files and extract findings
4. Apply OWASP grouping for deduplication
5. Generate horizontal-consolidation-report.md
6. Print at end:
   - Total files processed
   - Scanner names detected
   - Total findings per scanner
   - Total unique issue families
   - Confidence score breakdown

---

## FINAL CHECKLIST

Before delivering verify report has:
- [ ] All scanner files processed and listed in header
- [ ] Scanner columns auto-generated in Horizontal Matrix
- [ ] ✅/❌ correctly assigned per scanner per issue
- [ ] Confidence scores assigned
- [ ] Action column filled
- [ ] Summary Table with all findings and anchors
- [ ] False positives struck through with reason
- [ ] Full consolidated findings F-001 to F-N
- [ ] All IDs are clickable internal links
- [ ] Sorted by Severity → Category → Location
- [ ] PowerShell script generated

Now process all attached files and generate the report.
```

---

## Examples of What You Can Attach

```
Any of these work:
✅ Fortify CSV export
✅ Mend SAST CSV export
✅ Checkmarx CSV/XML export
✅ SonarQube JSON export
✅ Snyk JSON/CSV export
✅ Skill-based Exploitation Validator MD report
✅ SpecKit security review MD
✅ Mythos scan report MD
✅ Manual review notes in any text format
✅ SARIF XML files
✅ Any custom scanner output
```

---

## Adding a New Scanner in Future

No prompt changes needed. Just:
```
1. Attach the new scanner file
2. Agent auto-detects it
3. New column appears in Horizontal Matrix automatically
```

---

## Version History

| Version | Change |
|---------|--------|
| V1 | Hardcoded 4 scanners only (Fortify, Mend, Skill Validator, SpecKit) |
| V2 | Generic — supports any scanner, any format, auto-detection |

---
*EA GenAI Team | Vulnerability Automation Project | 2026*
