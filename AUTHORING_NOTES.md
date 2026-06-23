# Authoring Notes

Running list of gotchas to watch for when writing/editing posts. (Excluded from the built site via `_config.yml`.)

## Markdown / display gotchas

- **Always add YAML front matter to every post.** Without the `---` block (`layout: post`, `title`, `date`, ...), Jekyll copies the file verbatim instead of converting Markdown → HTML and applying the layout, so the whole page renders as raw text.
- **Math needs MathJax delimiters.** MathJax is loaded in `_layouts/default.html`. Inline math uses `$...$`, display math uses `$$...$$`. (kramdown converts `$$...$$` to `\[...\]`; literal `$...$` is passed through for MathJax.)
- **Never put a raw `|` inside math.** kramdown reads `|` as a table-cell separator and shreds the line into a `<table>`. Use `\lvert ... \rvert` for absolute value/cardinality and `\Vert` (or `\lVert ... \rVert`) for norms.
- **Inline `$...$` with braces-subscripts or brackets can break.** kramdown runs Markdown on the *inside* of inline `$...$`, so `_{...}` subscripts (with spaces) get eaten as `*emphasis*` and `[...]` as link syntax, which splits the delimiters and stops MathJax from rendering. Fix: wrap that expression in `$$...$$` (renders inline mid-sentence, but kramdown leaves the contents verbatim).
- **Watch for underscores that *pair* into emphasis.** Two or more "loose" underscores in the same paragraph can be matched as `_emphasis_`. The usual culprits are the positive-part subscript `_+` (e.g. `$(p - q)_+$`) and primed-index subscripts like `_{v'}` — when two of them appear, kramdown wraps the text between them in `<em>`. Intra-word subscripts like `$s_t$`, `$x_i$`, `$\pi_{\theta_{\text{old}}}$` are safe because the underscore sits between word characters. Fix the loose ones by switching them to `$$...$$`.
- **Simple inline tokens are fine** with single `$` — e.g. `$s_t$`, `$\pi_\theta$`, `$R(\tau)$`. The problems only show up with raw `|`, brace-subscripts containing spaces (`$\mathbb{E}_{a \sim p}$`), brackets, or paragraphs with multiple loose `_+` / `_{v'}` subscripts.
- **Rule of thumb:** if an inline expression contains `|`, `[ ]`, `_{ ... }` with spaces, or a loose `_+` / `_{...'}` subscript (especially more than one per paragraph), use `$$...$$` instead of `$...$`.
- **Escape stray dollar signs** in prose as `\$` — a lone `$` can otherwise be misread as the start of inline math.
- **Don't use display delimiters `$$...$$` for short inline bits unless needed.** It works (kramdown renders inline `$$...$$` as `\(...\)`), but prefer single `$...$` for readability and only reach for `$$...$$` to dodge the kramdown issues above.

## Security / infra

- **Don't use `polyfill.io`** (that CDN was hijacked to serve malware). MathJax 3 doesn't need it on modern browsers.

## Images

- **Put post images under `assets/images/<post-slug>/`** and reference them with an absolute, site-root path: `![alt](/assets/images/<post-slug>/foo.png)`.
- **Do not use relative paths** like `images/foo.png` in posts — the page lives at a permalink URL (e.g. `/<category>/<yyyy>/<mm>/<dd>/<slug>/`), so a relative path resolves under that URL and 404s. Anything under `assets/` is copied to the site root on build, so absolute paths always resolve.
- Always include descriptive alt text (it doubles as the caption-ish hover text and helps accessibility).

## Citations / references

- Style used across posts: inline author-year links like `([Author, 2024](#ref-author-2024))` that point to a numbered **References** section at the end.
- Each reference entry carries an inline anchor so the links resolve: `1. <a id="ref-author-2024"></a>Author. "Title." *Venue*, Year. [[link]](https://arxiv.org/abs/...)`.
- After editing, sanity-check that every `href="#ref-..."` has a matching `id="ref-..."` in the built HTML.

## Layout

- **Table of contents** (sticky left outline) is built client-side in `_layouts/post.html` from `h2`/`h3` headings and only appears at viewport width ≥ 1200px. Below that it's hidden and the article layout is unchanged.
- **Syntax highlighting** lives in `css/syntax.css` (a GitHub-style Rouge theme) and is linked from `_layouts/default.html`. Rouge tokenizes fenced code (` ```python `) into `<span>`s; `syntax.css` colors them. Regenerate/swap themes with `bundle exec rougify style <theme> > css/syntax.css` (e.g. `github`, `github.dark`, `monokai`).

## How to verify locally

```bash
bundle exec jekyll build      # or: bundle exec jekyll serve
```

After a build, inspect the generated HTML in `_site/.../index.html`. Quick greps that catch most issues:

- Stray `<table>` tags (from a raw `|` in math).
- `<em>`/`</em>` fused with math characters — signatures like `</em>{`, `)<em>`, `<em>+` mean an inline `$...$` got chewed up by emphasis pairing.
- Literal `$...$` or `$$` left sitting in a paragraph (math that didn't get picked up).
- `href="#ref-..."` without a matching `id="ref-..."` (broken citation links).
