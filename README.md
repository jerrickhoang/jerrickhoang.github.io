# Jerrick Hoang's Blog

Personal blog built with Jekyll and hosted on GitHub Pages.

## Setup

1. Install Ruby and Bundler:
   ```bash
   gem install bundler
   ```

2. Install dependencies:
   ```bash
   bundle install
   ```

3. Run locally:
   ```bash
   bundle exec jekyll serve
   ```

   Visit `http://localhost:4000` in your browser.

## Writing Posts

Create new posts in the `_posts/` directory with the following filename format:
```
YYYY-MM-DD-post-title.md
```

Frontmatter example:
```yaml
---
layout: post
title: "Your Post Title"
date: 2024-01-01
description: "A brief description"
category: "Category Name"
tags: ["tag1", "tag2"]
---
```

## Deployment

This site is automatically built by GitHub Pages when you push to the `main` branch (or `gh-pages` branch). Just commit and push your changes!

## License

MIT

