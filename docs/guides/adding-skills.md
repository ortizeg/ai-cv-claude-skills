# Adding Skills

This guide explains how to contribute new skills to the whet framework.

## Skill Anatomy

Every skill directory contains:

```
skills/my-new-skill/
├── SKILL.md      # Comprehensive guide for Claude Code (required)
├── README.md     # Human-readable overview (required)
├── examples/     # Code examples (recommended)
└── configs/      # Template config files (if applicable)
```

## SKILL.md Requirements

The SKILL.md file is read by Claude Code agents. It must be:

- **1000+ words** with substantial, actionable content
- **Code examples** demonstrating correct and incorrect patterns
- **Headers** organized logically (purpose, patterns, best practices, anti-patterns)
- **Self-contained** -- Claude should be able to follow it without other context

### Structure Template

```markdown
# Skill Name

One-paragraph description of what this skill does.

## When to Use

Bullet list of scenarios where this skill applies.

## Patterns

### Pattern 1: [Name]
Code example with explanation.

### Pattern 2: [Name]
Code example with explanation.

## Configuration

Template configs if applicable.

## Anti-Patterns to Avoid

What NOT to do, with examples.

## Best Practices

Numbered list of key takeaways.
```

## README.md Requirements

The README.md is for humans reading the repository. It should include:

- **Overview paragraph** explaining the skill's purpose
- **When to Use** section with bullet points
- **Key Features** as a bulleted list
- **Related Skills** listing 3-4 skills that combine well with this one

Target length: 200-400 words.

## Step-by-Step Process

### 1. Create the Directory

```bash
mkdir -p skills/albumentations/{examples,configs}
```

### 2. Write SKILL.md

Write a comprehensive guide following the template above. Include:

- Multiple code examples with correct/incorrect patterns
- Configuration snippets
- Integration guidance with other skills (Lightning, Pydantic, etc.)
- At least one complete, realistic code example

### 3. Write README.md

Write a concise overview with purpose, usage guidance, features, and related skills.

### 4. Add Examples

Create example files in `examples/` demonstrating real usage:

```
examples/
├── basic_augmentation.py
├── detection_transforms.py
└── custom_transform.py
```

### 5. Update Tests

Add your skill to the expected set in `tests/test_skills_completeness.py`:

```python
expected = {
    "master-skill",
    # ... existing skills ...
    "albumentations",  # Add your skill
}
```

### 6. Add Documentation

Create `docs/skills/albumentations.md` with a documentation page.

Update `mkdocs.yml` navigation:

```yaml
nav:
  - Skills:
    # ... existing entries ...
    - Albumentations: skills/albumentations.md
```

### 7. Verify

```bash
# Run tests to confirm skill is complete
uv run pytest tests/test_skills_completeness.py -v

# Check documentation builds
uv run docs-build
```

## Checklist Before Submitting PR

- [ ] `SKILL.md` exists and is 1000+ words
- [ ] `SKILL.md` has code examples (```python blocks)
- [ ] `README.md` exists with overview, features, related skills
- [ ] Skill added to expected set in `test_skills_completeness.py`
- [ ] Documentation page created in `docs/skills/`
- [ ] `mkdocs.yml` nav updated
- [ ] All tests pass: `uv run test`
- [ ] Linting passes: `uv run lint`

## Example: Adding an "Albumentations" Skill

An albumentations skill would cover:

- **SKILL.md**: Integration with Lightning DataModules, transform composition patterns, detection-safe augmentations (bbox-aware), Pydantic configs for augmentation parameters, testing augmentation pipelines
- **README.md**: Purpose (image augmentation for training), when to use, key features (fast transforms, bbox support), related skills (pytorch-lightning, testing, opencv)
- **Examples**: Basic transforms, detection augmentations, custom transform class
- **Configs**: Default augmentation configs for classification and detection
