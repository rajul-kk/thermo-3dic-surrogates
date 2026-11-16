# Releasing fieldlint

Publishing uses PyPI trusted publishing from `.github/workflows/fieldlint-release.yml` (repository root): no API token is
stored anywhere. Steps 1-2 are one-time and need the owner's browser.

| step | where | what |
|---|---|---|
| 1 | test.pypi.org and pypi.org, Account settings, Publishing, "Add a new pending publisher" | project `fieldlint`, owner `rajul-kk`, repository `thermo-3dic-surrogates`, workflow `fieldlint-release.yml`, environment `testpypi` (TestPyPI) / `pypi` (PyPI) |
| 2 | GitHub, thermo-3dic-surrogates, Settings, Environments | create `testpypi` and `pypi` (optionally: required reviewer on `pypi`) |
| 3 | `pyproject.toml` and `src/fieldlint/__init__.py` | the new version, the same in both (a test enforces it) |
| 4 | git | tag `fieldlint-v<version>` on main and push the tag: the workflow builds, checks the sdist and wheel, checks the tag matches the version, and runs the installed wheel |
| 5 | GitHub, Actions, fieldlint-release, "Run workflow" (on main) | target `testpypi`; then `pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple fieldlint==<version>` and run `fieldlint --list-rules` |
| 6 | same | target `pypi` |
