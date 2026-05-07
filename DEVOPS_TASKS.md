https://learn.microsoft.com/en-us/training/modules/spec-driven-development-github-spec-kit-enterprise-developers/

https://learn.microsoft.com/en-us/training/modules/spec-driven-development-github-spec-kit-enterprise-developers/11-integrate-spec-kit-advanced-deployment

The two levels — clearly separated
Level 1 — Community extension (installable, works today)
This is the azure-devops extension that syncs tasks.md → ADO work items automatically via one command /speckit.adosync. One install, minimal config. This is what I described.
Level 2 — CI/CD pipeline integration (manual scripts, your responsibility)
This is what the Microsoft Learn page describes. It is not a packaged extension you install. It is a pattern and guidance — you write the PowerShell/Python scripts yourself, wire them into your ADO pipeline YAML, and configure the PR templates manually.


The community extension handles spec.md and tasks.md sync to ADO work items today with OAuth and zero PAT management. Full CI/CD enforcement — spec validation gates, constitution compliance checks, deployment gates — is a documented architecture pattern from Microsoft Learn that your team implements as pipeline scripts. Native out-of-the-box pipeline integration does not exist yet but is directionally coming given Microsoft's investment in official training content.

GitHub is the primary first-class integration — Copilot commands, Actions pipelines, PR templates, and Issue sync all work natively and are officially documented by Microsoft. Azure DevOps works via a community extension for task sync but requires manual engineering effort for pipeline enforcement. If your team can use GitHub, the full integration story is available today with minimal setup. If you are locked into Azure DevOps, the task sync works but CI/CD enforcement requires your own scripts.