# Release procedure

The project uses `MAJOR.MINOR.PATCH` versions. During the 0.x series, collect
compatibility changes and feature milestones in a minor release; use patch
releases for compatible fixes. Keep migration guidance with each release.

## Prepare a release PR

1. Select a tested commit on `main` and review current dependency alerts.
2. Synchronize `pyproject.toml`, the package `__version__`, `CITATION.cff`,
   `.zenodo.json`, `frontend/package.json`, and both root version fields in
   `frontend/package-lock.json`.
3. Set `CITATION.cff`'s `date-released` and `.zenodo.json`'s `publication_date` to
   the planned publication date. Move the changelog entries to a matching dated
   version heading and retain an empty `Unreleased` section.
4. Add upgrade notes under `docs/releases/`, including compatibility changes and
   unfinished capabilities. Refresh the citation dates if publication is delayed.
5. Regenerate the documentation using the current base revision, and run the
   checks in [CONTRIBUTING.md](../CONTRIBUTING.md), the MLflow integration tests,
   frontend audit/build, and package build.
6. Run `poetry run python scripts/check_release.py --tag v0.7.0`, substituting
   the intended version. CI checks agreement without a tag; the Release workflow
   also checks the tag on a tag-triggered run. The validator reads metadata
   without importing application code and supports Python 3.10+.
7. Merge the release PR after its checks pass. The Release workflow can also be
   dispatched against the PR branch to validate its artifacts before tagging.

## Tag and publish

On the merged commit, verify the metadata again and confirm that its CI and
security checks pass. Create and push an annotated tag named `v<version>` that
points to this exact commit. Do not move or reuse an existing published tag.

Wait for the tag's Release workflow to succeed. It validates metadata, runs the
demo, builds the Python source distribution and wheel, builds the frontend, and
uploads build artifacts. It does not publish a GitHub release, upload to PyPI,
or deploy the application.

Create a GitHub release from the existing tag, attach the Python distributions
from that workflow, and use the corresponding `docs/releases/` file as the
release notes. Include a link to the full changelog. Verify the attached wheel's
version matches the tag before publishing.

## Verify Zenodo

Confirm the repository's GitHub integration is enabled in Zenodo before
publishing. After the GitHub release is published, verify that the resulting
Zenodo record has the intended title, version, publication date, authors,
license, and GitHub archive. Repository metadata alone does not prove the
deposit succeeded.

Zenodo uses `.zenodo.json` when both it and `CITATION.cff` are present; keep both
consistent for GitHub citations and archiving. See
[Zenodo's metadata guidance](https://help.zenodo.org/docs/github/describe-software/).

Keep the all-versions DOI `10.5281/zenodo.21804219` in the README and citation
metadata. Record the newly assigned version-specific DOI in the published
release notes after checking the deposit. Do not put the concept DOI in the
`.zenodo.json` `doi` field, which would identify an existing deposit rather than
requesting a new version DOI.
