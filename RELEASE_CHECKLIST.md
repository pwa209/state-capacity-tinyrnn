# Release Checklist

Use this checklist before final manuscript submission.

## GitHub software release

- [ ] Confirm all intended code files are present in the repository.
- [ ] Confirm no raw data, private files, local virtual environments or large generated outputs are committed.
- [ ] Confirm `README.md`, `LICENSE`, `CITATION.cff`, `DATA_AVAILABILITY.md`, `environment.yml` and `pyproject.toml` are current.
- [ ] Create a GitHub release tag, suggested: `v1.0.0`.
- [ ] Connect the repository to Zenodo and archive the GitHub release.
- [ ] Record the GitHub URL, release tag, commit hash and Zenodo software DOI.

## Zenodo derived-data release

- [x] Upload `state_capacity_tinyrnn_zenodo_derived_data_package.zip` as a Zenodo Dataset record.
- [x] Use a clear title, for example: `Derived data and source tables for machine-defined state and capacity profiles in human cognition`.
- [x] Include raw-data provenance in the Zenodo description.
- [x] Confirm raw public datasets are not redistributed in the derived-data archive.
- [x] Record the Zenodo derived-data DOI: 10.5281/zenodo.21028379.

## Manuscript update

- [x] Replace provisional Data availability text with the derived-data DOI.
- [ ] Replace provisional Code availability text with the GitHub URL, release tag, commit hash and Zenodo software DOI.
- [ ] Confirm the manuscript does not contain `[PLACEHOLDER]`, `[GitHub URL]`, `[Zenodo DOI]`, `[hash]` or `vX.X.X`.
