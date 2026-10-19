# RealtimeTTS release contract

- Build the wheel and sdist once from a clean release checkout with the standard build tooling. Preserve unrelated worktrees and reuse the exact artifacts through validation and upload.
- Run the relevant test suite once before building, then run:
  python -m build --sdist --wheel
  python -m twine check --strict dist/*
- Install the exact built wheel and smoke-test the affected engines on their supported platforms. For a service deployment, verify that it runs the installed wheel. Compare performance with the existing baseline when inference or native builds change. Do not copy edited Python files directly into site-packages.
- Upload the unchanged wheel and sdist with Twine. When promoting through TestPyPI, reuse the same files for PyPI after fresh-install and runtime acceptance.
- After each upload, compare the package-index SHA-256 values with the hashes of the local artifacts.
