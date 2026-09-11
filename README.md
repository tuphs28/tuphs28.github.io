# tuphs28.github.io

Personal academic website and blog — DPhil student in Artificial Intelligence at the University of Oxford.

Built with [Jekyll](https://jekyllrb.com/) and the [minima](https://github.com/jekyll/minima) theme, hosted on [GitHub Pages](https://pages.github.com/).

## Local development

```sh
bundle install
bundle exec jekyll serve
```

Then open http://localhost:4000.

## Deployment

Push to `main`. In the repo's **Settings → Pages**, set:

- Source: Deploy from a branch
- Branch: `main` / `(root)`

The site will be published at https://tuphs28.github.io.

## Structure

- `index.md` — homepage
- `about.md`, `research.md`, `publications.md`, `cv.md` — static pages
- `_posts/` — blog posts (Markdown, filename `YYYY-MM-DD-title.md`)
- `_config.yml` — site configuration
- `assets/` — images, CSS overrides, CV PDF, etc.

## TODO

- [ ] Fill in name, bio, and research details in `index.md`, `about.md`, `research.md`
- [ ] Add real links (Google Scholar, ORCID, Twitter/X, LinkedIn)
- [ ] Add CV PDF to `assets/cv.pdf`
- [ ] Add publications to `publications.md`
- [ ] Set a custom domain (optional) via a `CNAME` file
