# Releasing RealtimeTTS

RealtimeTTS releases promote the exact artifacts already exercised by the target runtime. The standard build and package-index checks cover artifact parity; they do not repeat the full test suite.

1. Build from a clean release checkout containing the intended changes; preserve unrelated worktrees.

2. Run the relevant tests once, then build the wheel and sdist once:

~~~powershell
python -m build --sdist --wheel
python -m twine check --strict dist/*
~~~

3. Install the newly built wheel and smoke-test the affected engines on their supported platforms. For a service deployment, verify that it runs the installed wheel. Compare performance with the existing baseline when inference or native builds change. Do not copy individual .py files into site-packages.

4. Upload the unchanged artifacts with Twine. For a TestPyPI rehearsal, use:

~~~powershell
python -m twine upload --repository testpypi dist/<wheel> dist/<sdist>
~~~

After the fresh-install and runtime checks pass, upload those same files to PyPI:

~~~powershell
python -m twine upload dist/<wheel> dist/<sdist>
~~~

Use the appropriate index for a direct PyPI release when TestPyPI is not part of the release.

5. Verify each uploaded file against the package-index JSON for the selected index. Compare its reported digests.sha256 with the SHA-256 of the unchanged local artifact, and download the published files again for an install check. If a fix is needed, commit it and build a new artifact pair; never replace files under an existing package filename.

The release is complete only when the uploaded artifact set, local hashes, and fresh installation all match.
