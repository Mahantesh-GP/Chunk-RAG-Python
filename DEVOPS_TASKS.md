https://learn.microsoft.com/en-us/training/modules/spec-driven-development-github-spec-kit-enterprise-developers/


The two levels — clearly separated
Level 1 — Community extension (installable, works today)
This is the azure-devops extension that syncs tasks.md → ADO work items automatically via one command /speckit.adosync. One install, minimal config. This is what I described.
Level 2 — CI/CD pipeline integration (manual scripts, your responsibility)
This is what the Microsoft Learn page describes. It is not a packaged extension you install. It is a pattern and guidance — you write the PowerShell/Python scripts yourself, wire them into your ADO pipeline YAML, and configure the PR templates manually.
