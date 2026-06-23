# Authoring Notes

Running list of gotchas to watch for when writing/editing posts. (Excluded from the built site via `_config.yml`.)

## Markdown / display gotchas

- **Always add YAML front matter to every post.** Without the `---` block (`layout: post`, `title`, `date`, ...), Jekyll copies the file verbatim instead of converting Markdown → HTML and applying the layout, so the whole page renders as raw text.
- **Math needs MathJax delimiters.** MathJax is loaded in `_layouts/default.html`. Inline math uses `$...$`, display math uses `$$...$$`. (kramdown converts `$$...$$` to `\[...\]`; literal `$...$` is passed through for MathJax.)
- **Never put a raw `|` inside math.** kramdown reads `|` as a table-cell separator and shreds the line into a `<table>`. Use `\lvert ... \rvert` for absolute value/cardinality and `\Vert` (or `\lVert ... \rVert`) for norms.
- **Inline `$...$` with braces-subscripts or brackets can break.** kramdown runs Markdown on the *inside* of inline `$...$`, so `_{...}` subscripts (with spaces) get eaten as `*emphasis*` and `[...]` as link syntax, which splits the delimiters and stops MathJax from rendering. Fix: wrap that expression in `$$...$$` (renders inline mid-sentence, but kramdown leaves the contents verbatim).
- **Simple inline tokens are fine** with single `$` — e.g. `$s_t$`, `$\pi_\theta$`, `$R(\tau)$`. The problems only show up with raw `|`, brace-subscripts containing spaces (`$\mathbb{E}_{a \sim p}$`), or brackets.
- **Rule of thumb:** if an inline expression contains `|`, `[ ]`, or `_{ ... }` with spaces, use `$$...$$` instead of `$...$`.
- **Escape stray dollar signs** in prose as `\$` — a lone `$` can otherwise be misread as the start of inline math.

## Security / infra

- **Don't use `polyfill.io`** (that CDN was hijacked to serve malware). MathJax 3 doesn't need it on modern browsers.

## Layout

- **Table of contents** (sticky left outline) is built client-side in `_layouts/post.html` from `h2`/`h3` headings and only appears at viewport width ≥ 1200px. Below that it's hidden and the article layout is unchanged.

## How to verify locally

```bash
bundle exec jekyll build      # or: bundle exec jekyll serve
```

After a build, inspect the generated HTML in `_site/.../index.html` — a quick check for stray `<table>` tags or literal `$...$` left in paragraphs catches most math/kramdown issues.
