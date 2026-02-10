# Library Review

The Library Review skill provides a structured framework for evaluating and selecting ML/CV libraries, covering API quality, maintenance health, performance benchmarks, and integration fit.

**Skill directory:** `skills/library-review/`

## Purpose

Choosing the wrong library costs weeks of wasted effort. This skill teaches Claude Code to systematically evaluate ML libraries across multiple dimensions: API design, documentation quality, community activity, performance characteristics, and compatibility with your existing stack. Use it when deciding between competing options (e.g., Albumentations vs. torchvision transforms, FastAPI vs. Flask for serving).

## When to Use

- Selecting a new dependency for your project
- Comparing alternatives for a specific function (augmentation, serving, logging)
- Evaluating whether to adopt a new ML framework or tool
- Deciding between building custom code vs. using a library

## Key Patterns

### Evaluation Framework

When asked to review a library, Claude Code produces an assessment covering:

```markdown
## Library: [Name]

### Fitness Assessment
- **API Quality**: Is the API well-designed, typed, and consistent?
- **Documentation**: Are there comprehensive docs, tutorials, and API references?
- **Maintenance**: How active is development? When was the last release?
- **Community**: GitHub stars, issues response time, Stack Overflow presence
- **Performance**: Benchmark data, known bottlenecks

### Integration Check
- **Python Version**: Compatible with project's Python version?
- **Dependency Conflicts**: Any conflicts with existing dependencies?
- **Type Stubs**: Does it provide type stubs or inline types?
- **License**: Compatible with project license?

### Recommendation
[Use / Use with caveats / Avoid] -- [Reasoning]
```

### Comparison Template

```markdown
## Comparison: [Library A] vs [Library B]

| Criterion         | Library A | Library B |
|-------------------|-----------|-----------|
| API quality       | ...       | ...       |
| Type safety       | ...       | ...       |
| Performance       | ...       | ...       |
| Maintenance       | ...       | ...       |
| Documentation     | ...       | ...       |
| Stack integration | ...       | ...       |

### Verdict
[Which to use and under what conditions]
```

### Decision Criteria for CV/ML Projects

The skill emphasizes criteria specific to ML work:

1. **Does it support GPU acceleration?** -- Critical for training and inference libraries.
2. **Does it handle batched inputs?** -- Essential for production throughput.
3. **Is it deterministic?** -- Reproducibility matters for experiments.
4. **Does it work in containers?** -- Headless operation, no display server dependency.
5. **Does it integrate with PyTorch?** -- Tensor interop, gradient support if needed.

## Anti-Patterns to Avoid

- Do not choose libraries based solely on GitHub stars
- Do not adopt a library without checking its dependency tree for conflicts
- Avoid libraries with no type annotations if your project enforces strict typing
- Do not evaluate only the happy path -- check error handling and edge case documentation

## Combines Well With

- **Master Skill** -- Library must align with project conventions
- **Pixi** -- Verify library is available via conda-forge or PyPI
- **Code Quality** -- Library should not degrade type checking or linting
- **Abstraction Patterns** -- Wrap chosen library behind an interface for future replacement

## Full Reference

See [`skills/library-review/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/library-review/SKILL.md) for the complete evaluation rubric, including security assessment, license compatibility matrix, and migration cost estimation.
