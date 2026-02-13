# Interface Design

The Interface Design skill provides expert guidance for building crafted dashboards, admin panels, SaaS apps, tools, and interactive products with intent-driven design over template defaults.

**Skill directory:** `skills/interface-design/`

## Purpose

AI-generated interfaces tend toward generic output — the same sidebar widths, card grids, and metric layouts every time. This skill encodes craft principles that force deliberate choices at every level, ensuring each interface emerges from its specific context rather than defaulting to common patterns.

## When to Use

Use this skill whenever you need to:

- Build dashboards, admin panels, or data-heavy application interfaces
- Design SaaS UIs with navigation, cards, forms, and interactive controls
- Create a design system with token architecture, spacing scales, and elevation hierarchy
- Evaluate and improve AI-generated UI output that feels generic or templated

## Key Patterns

### Intent-First Workflow

Before writing any UI code, articulate three things:

```
Who is this human? (not "users" — the actual person and their context)
What must they accomplish? (the verb, not "use the dashboard")
What should this feel like? (specific qualities, not "clean and modern")
```

### Product Domain Exploration

Before proposing a visual direction, produce all four required outputs:

1. **Domain** — 5+ concepts, metaphors, vocabulary from the product's world
2. **Color world** — 5+ colors that exist naturally in the product's domain
3. **Signature** — one element that could only exist for THIS product
4. **Defaults** — 3 obvious choices you're explicitly rejecting

### Token Architecture

Every color traces back to primitives:

```css
/* Foreground: text hierarchy */
--fg-primary: ...;    /* default text */
--fg-secondary: ...;  /* supporting text */
--fg-tertiary: ...;   /* metadata */
--fg-muted: ...;      /* disabled/placeholder */

/* Background: surface elevation */
--bg-base: ...;       /* app canvas */
--bg-surface-100: ...; /* cards, panels */
--bg-surface-200: ...; /* dropdowns, popovers */

/* Border: separation hierarchy */
--border-default: ...;
--border-subtle: ...;
--border-strong: ...;
```

### Design Validation Checks

Run before presenting output:

- **Swap test** — would swapping your choices for common alternatives change anything?
- **Squint test** — can you still perceive hierarchy with blurred eyes?
- **Signature test** — can you point to 5 specific elements showing your signature?
- **Token test** — do CSS variable names sound like they belong to this product?

## Anti-Patterns to Avoid

- Harsh borders that demand attention instead of whispering structure
- Dramatic surface elevation jumps instead of subtle lightness shifts
- Mixed depth strategies (shadows AND heavy borders)
- Missing interaction states (hover, focus, disabled, loading, error)
- Generic token names (`--gray-700`) instead of domain-specific ones (`--ink`, `--parchment`)
- Same sidebar width, card grid, and metric layout every time

## Reference Files

The skill includes deep-dive references:

- **principles.md** — Surface architecture, spacing systems, depth strategies, typography
- **critique.md** — Post-build craft critique protocol
- **example.md** — Subtle layering decisions with real reasoning
- **validation.md** — Memory management and consistency checks

## Credits

Based on the [interface-design skill](https://github.com/Dammyjay93/interface-design) by Dammyjay93.

## Full Reference

See [`skills/interface-design/SKILL.md`](https://github.com/ortizeg/whet/blob/main/skills/interface-design/SKILL.md) for the complete skill with workflow, mandate, and all design principles.
