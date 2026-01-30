# GitHub Repo Setup

The GitHub Repo Setup skill defines best practices for initializing and configuring GitHub repositories for CV/ML projects, covering branch protection, PR templates, CODEOWNERS, and automation via the `gh` CLI.

**Skill directory:** `skills/github-repo-setup/`

## Purpose

Repository configuration is infrastructure. Inconsistent settings across repos lead to unreviewed merges, broken main branches, and lost context in PRs. This skill teaches Claude Code to set up repositories with reproducible, scripted configuration that enforces quality gates from day one.

## When to Use

- Creating a new GitHub repository for any CV/ML project
- Adding branch protection and required status checks
- Setting up PR templates and issue templates for team workflows
- Configuring CODEOWNERS for automated review routing
- Scripting repository settings for consistency across multiple repos

## Key Patterns

### Branch Protection via `gh` CLI

```bash
gh api repos/{owner}/{repo}/branches/main/protection \
  --method PUT \
  --input - <<'EOF'
{
  "required_status_checks": {
    "strict": true,
    "contexts": ["test", "lint"]
  },
  "enforce_admins": true,
  "required_pull_request_reviews": {
    "required_approving_review_count": 1,
    "dismiss_stale_reviews": true
  },
  "restrictions": null,
  "allow_force_pushes": false
}
EOF
```

### Repository Settings

```bash
gh api repos/{owner}/{repo} \
  --method PATCH \
  --field delete_branch_on_merge=true \
  --field allow_squash_merge=true \
  --field allow_merge_commit=false \
  --field allow_rebase_merge=false
```

### Pydantic Configuration Model

```python
from pydantic import BaseModel, Field

class RepoConfig(BaseModel):
    owner: str
    name: str
    visibility: str = Field(default="private", pattern=r"^(public|private)$")
    branch_protection: BranchProtection = Field(default_factory=BranchProtection)
    merge_strategy: MergeStrategy = Field(default_factory=MergeStrategy)
```

## Anti-Patterns to Avoid

- Do not push directly to main without branch protection
- Do not use merge commits for feature branches (use squash merge)
- Do not configure repositories manually through the web UI (use `gh` CLI for reproducibility)
- Do not skip PR templates (leads to empty descriptions and lost context)
- Do not set too many required reviewers for small teams (1 is sufficient)

## Combines Well With

- **GitHub Actions** — CI workflows become required status checks in branch protection
- **Pre-commit** — Local hooks complement CI-level checks
- **Code Quality** — Ruff and mypy enforced through required status checks
- **Testing** — Test suite becomes a merge gate

## Full Reference

See [`skills/github-repo-setup/SKILL.md`](https://github.com/ortizeg/ai-cv-claude-skills/blob/main/skills/github-repo-setup/SKILL.md) for the complete setup script, Pydantic models, issue templates, CODEOWNERS patterns, and merge strategy guidance.
