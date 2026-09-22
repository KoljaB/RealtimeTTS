from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import subprocess
import tarfile
import tempfile
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
GUARD_PATH = ROOT / "tools" / "release_guard.py"
SPEC = importlib.util.spec_from_file_location("release_guard_under_test", GUARD_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import setup failure
    raise RuntimeError(f"cannot load release guard from {GUARD_PATH}")
release_guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release_guard)


class ReleaseGuardTests(unittest.TestCase):
    @staticmethod
    def _git(repo: Path, *args: str) -> None:
        result = subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise AssertionError(
                f"git {' '.join(args)} failed:\n{result.stdout}\n{result.stderr}"
            )

    def _init_repo(self, root: Path) -> Path:
        repo = root / "repo"
        repo.mkdir()
        self._git(repo, "init", "--quiet")
        source_packages = {
            "demo_pkg": repo / "demo_pkg",
            "demo_server": repo / "src" / "demo_server",
        }
        for package, source_package in source_packages.items():
            source_package.mkdir(parents=True)
            (source_package / "__init__.py").write_bytes(
                f"PACKAGE = '{package}'\n".encode()
            )
        self._git(repo, "add", "demo_pkg", "src/demo_server")
        self._git(
            repo,
            "-c",
            "user.name=release-guard-test",
            "-c",
            "user.email=release-guard-test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "initial",
        )
        return repo

    @staticmethod
    def _package_bytes(package: str) -> bytes:
        return f"PACKAGE = '{package}'\r\n".encode()

    def _write_artifacts(self, root: Path) -> tuple[Path, Path]:
        wheel = root / "demo_distribution-1.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for package in ("demo_pkg", "demo_server"):
                archive.writestr(
                    f"{package}/__init__.py", self._package_bytes(package)
                )
            archive.writestr(
                "demo_distribution-1.0.dist-info/METADATA",
                "Metadata-Version: 2.1\nName: demo-distribution\nVersion: 1.0\n",
            )

        sdist = root / "demo_distribution-1.0.tar.gz"
        with tarfile.open(sdist, "w:gz") as archive:
            metadata = b"Name: demo-distribution\nVersion: 1.0\n"
            metadata_info = tarfile.TarInfo("demo_distribution-1.0/PKG-INFO")
            metadata_info.size = len(metadata)
            archive.addfile(metadata_info, io.BytesIO(metadata))
            for package in ("demo_pkg", "demo_server"):
                data = self._package_bytes(package)
                info = tarfile.TarInfo(
                    f"demo_distribution-1.0/{package}/__init__.py"
                )
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        return wheel, sdist

    def _runtime(self, root: Path) -> tuple[list[Path], list[str]]:
        runtime_root = root / "runtime"
        runtime_dirs: list[Path] = []
        modules: list[str] = []
        for package in ("demo_pkg", "demo_server"):
            package_dir = runtime_root / package
            package_dir.mkdir(parents=True)
            (package_dir / "__init__.py").write_bytes(self._package_bytes(package))
            runtime_dirs.append(package_dir)
            modules.append(package)
        return runtime_dirs, modules

    @staticmethod
    def _probe(runtime_dirs: list[Path], modules: list[str]):
        imports = {
            module: str(package_dir.resolve())
            for module, package_dir in zip(modules, runtime_dirs, strict=True)
        }

        def probe(
            runtime_python: Path,
            requested_modules: list[str],
            distributions: list[str],
        ) -> dict[str, object]:
            if requested_modules != modules:
                raise AssertionError(requested_modules)
            versions = {
                "demo_distribution": "1.0",
                "demo-dependency": "2.0",
            }
            return {
                "python": str(release_guard._absolute_path(runtime_python)),
                "prefix": str(runtime_dirs[0].parent.resolve()),
                "imports": imports,
                "versions": {name: versions[name] for name in distributions},
            }

        return probe

    @staticmethod
    def _attest_args(
        repo: Path,
        wheel: Path,
        sdist: Path,
        runtime_dirs: list[Path],
        modules: list[str],
        signing_key_file: Path,
        manifest: Path,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            repo=repo,
            package_dir=["demo_pkg", "demo_server"],
            source_package_dir=["demo_pkg", "src/demo_server"],
            wheel=wheel,
            sdist=sdist,
            artifact=[],
            component="demo",
            distribution="demo_distribution",
            runtime_package_dir=runtime_dirs,
            runtime_module=modules,
            runtime_python=runtime_dirs[0].parent / "python",
            dependency=["demo-dependency"],
            runtime_label="test-runtime",
            signing_key_file=signing_key_file,
            signer="test-runtime",
            output=manifest,
        )

    @staticmethod
    def _verify_args(
        repo: Path,
        wheel: Path,
        sdist: Path,
        allowed_signers_file: Path,
        manifest: Path,
        *,
        allow_remote_attestation: bool = False,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            repo=repo,
            package_dir=["demo_pkg", "demo_server"],
            source_package_dir=["demo_pkg", "src/demo_server"],
            wheel=wheel,
            sdist=sdist,
            artifact=[],
            manifest=manifest,
            signature_file=Path(str(manifest) + ".sig"),
            allowed_signers_file=allowed_signers_file,
            signer="test-runtime",
            max_attestation_age_seconds=1800,
            allow_remote_attestation=allow_remote_attestation,
        )

    @staticmethod
    def _sign(manifest: Path, signing_key: Path) -> Path:
        if not signing_key.is_file():
            raise AssertionError(signing_key)
        signature = Path(str(manifest) + ".sig")
        signature.write_text(
            hashlib.sha256(manifest.read_bytes()).hexdigest(), encoding="ascii"
        )
        return signature

    @staticmethod
    def _verify_signature(
        manifest: Path,
        signature: Path,
        allowed_signers: Path,
        signer: str,
        expected_fingerprint: object,
    ) -> None:
        if not allowed_signers.is_file() or signer != "test-runtime":
            raise release_guard.GuardError("deployment manifest signature is invalid")
        expected = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if signature.read_text(encoding="ascii") != expected:
            raise release_guard.GuardError("deployment manifest signature is invalid")

    def test_dirty_linked_worktree_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = self._init_repo(root)
            linked = root / "linked"
            self._git(repo, "worktree", "add", "--detach", str(linked), "HEAD")
            try:
                (linked / "uncommitted.py").write_text("DIRTY = True\n", encoding="utf-8")
                stdout = io.StringIO()
                stderr = io.StringIO()
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    result = release_guard.main(
                        ["check-worktrees", "--repo", str(repo)]
                    )
                self.assertEqual(result, 2)
                self.assertIn("publication blocked", stderr.getvalue())
                self.assertIn("uncommitted.py", stderr.getvalue())
            finally:
                self._git(repo, "worktree", "remove", "--force", str(linked))

    def test_multi_package_signed_attestation_rechecks_current_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = self._init_repo(root)
            wheel, sdist = self._write_artifacts(root)
            runtime_dirs, modules = self._runtime(root)
            key_file = root / "signing.key"
            key_file.write_text("private test key", encoding="ascii")
            manifest = root / "deployment.json"
            attest_args = self._attest_args(
                repo, wheel, sdist, runtime_dirs, modules, key_file, manifest
            )
            verify_args = self._verify_args(
                repo, wheel, sdist, key_file, manifest
            )
            with mock.patch.object(
                release_guard,
                "_runtime_probe",
                side_effect=self._probe(runtime_dirs, modules),
            ), mock.patch.object(
                release_guard, "_sign_manifest_file", side_effect=self._sign
            ), mock.patch.object(
                release_guard,
                "_verify_manifest_signature",
                side_effect=self._verify_signature,
            ):
                attested = release_guard.command_attest(attest_args)
                verified = release_guard.command_verify(verify_args)
                self.assertEqual(attested["schema_version"], 3)
                self.assertEqual(
                    verified["packages"], ["demo_pkg", "demo_server"]
                )
                self.assertTrue(verified["live_runtime_rechecked"])

                (runtime_dirs[1] / "__init__.py").write_bytes(b"CHANGED = True\r\n")
                with self.assertRaisesRegex(
                    release_guard.GuardError, "current runtime differs"
                ):
                    release_guard.command_verify(verify_args)

    def test_manifest_edit_is_rejected_and_remote_mode_requires_fresh_signature(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = self._init_repo(root)
            wheel, sdist = self._write_artifacts(root)
            runtime_dirs, modules = self._runtime(root)
            key_file = root / "signing.key"
            key_file.write_text("private test key", encoding="ascii")
            manifest = root / "deployment.json"
            with mock.patch.object(
                release_guard,
                "_runtime_probe",
                side_effect=self._probe(runtime_dirs, modules),
            ), mock.patch.object(
                release_guard, "_sign_manifest_file", side_effect=self._sign
            ):
                release_guard.command_attest(
                    self._attest_args(
                        repo, wheel, sdist, runtime_dirs, modules, key_file, manifest
                    )
                )

            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["source_commit"] = "0" * 40
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            verify_args = self._verify_args(
                repo,
                wheel,
                sdist,
                key_file,
                manifest,
                allow_remote_attestation=True,
            )
            with mock.patch.object(
                release_guard,
                "_verify_manifest_signature",
                side_effect=self._verify_signature,
            ), self.assertRaisesRegex(release_guard.GuardError, "signature is invalid"):
                release_guard.command_verify(verify_args)

    def test_remote_mode_still_compares_signed_runtime_hashes_to_wheel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = self._init_repo(root)
            wheel, sdist = self._write_artifacts(root)
            runtime_dirs, modules = self._runtime(root)
            key_file = root / "signing.key"
            key_file.write_text("private test key", encoding="ascii")
            manifest = root / "deployment.json"
            with mock.patch.object(
                release_guard,
                "_runtime_probe",
                side_effect=self._probe(runtime_dirs, modules),
            ), mock.patch.object(
                release_guard, "_sign_manifest_file", side_effect=self._sign
            ):
                release_guard.command_attest(
                    self._attest_args(
                        repo, wheel, sdist, runtime_dirs, modules, key_file, manifest
                    )
                )

            payload = json.loads(manifest.read_text(encoding="utf-8"))
            package_files = payload["runtime"]["package_files"]["demo_server"]
            package_files[next(iter(package_files))] = "0" * 64
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            self._sign(manifest, key_file)
            verify_args = self._verify_args(
                repo,
                wheel,
                sdist,
                key_file,
                manifest,
                allow_remote_attestation=True,
            )
            with mock.patch.object(
                release_guard,
                "_verify_manifest_signature",
                side_effect=self._verify_signature,
            ), self.assertRaisesRegex(
                release_guard.GuardError, "deployed runtime package demo_server and wheel"
            ):
                release_guard.command_verify(verify_args)

    def test_component_profile_and_source_containment_are_mandatory(self) -> None:
        profile = release_guard.COMPONENT_PROFILES["RealtimeSTT"]
        self.assertEqual(
            tuple(item[0] for item in profile["packages"]),
            ("RealtimeSTT", "RealtimeSTT_server", "example_fastapi_server"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary).resolve()
            with self.assertRaisesRegex(release_guard.GuardError, "stay inside"):
                release_guard._repo_member(repo, "../outside", "source package")

        native = release_guard.COMPONENT_PROFILES["RealtimeTTSQwenNative"]
        self.assertEqual(native["distribution"], "realtimetts-qwen-native")
        self.assertEqual(
            native["required_wheel_platforms"],
            (
                "manylinux_2_35_x86_64",
                "manylinux_2_35_aarch64",
                "win_amd64",
            ),
        )
        self.assertFalse(native["publish_sdist"])
        self.assertEqual(native["binary_package_prefixes"], {"qwentts_cpp": ("lib/",)})

    def test_component_signer_key_fingerprint_is_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "deployment.json"
            signature = root / "deployment.json.sig"
            allowed = root / "allowed_signers"
            manifest.write_text("{}", encoding="utf-8")
            signature.write_text("invalid", encoding="ascii")
            encoded = base64.b64encode(b"different-key").decode("ascii")
            allowed.write_text(
                f"linux-services ssh-ed25519 {encoded} test\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(release_guard.GuardError, "trust profile"):
                release_guard._verify_manifest_signature(
                    manifest,
                    signature,
                    allowed,
                    "linux-services",
                    release_guard.COMPONENT_PROFILES["RealtimeTTS"][
                        "signer_fingerprint"
                    ],
                )

    def test_cpu_deployment_cannot_attest_gpu_or_publish(self) -> None:
        profile = release_guard._component_profile("RealtimeTTSCPU")
        self.assertEqual(
            release_guard._service_name(profile, "wwz-qwen3-tts-cpu.service"),
            "wwz-qwen3-tts-cpu.service",
        )
        self.assertEqual(profile["packages"], release_guard.COMPONENT_PROFILES["RealtimeTTS"]["packages"])
        self.assertEqual(profile["signer_fingerprint"], release_guard.COMPONENT_PROFILES["RealtimeTTS"]["signer_fingerprint"])
        with self.assertRaises(release_guard.GuardError):
            release_guard._service_name(profile, "wwz-qwen3-tts.service")
        with self.assertRaisesRegex(release_guard.GuardError, "not publishable"):
            release_guard._component_profile("RealtimeTTSCPU", publishing=True)
        native = release_guard._component_profile("RealtimeTTSQwenNativeCPU")
        self.assertEqual(native["service_name"], profile["service_name"])
        self.assertEqual(native["required_wheel_platforms"], ("linux_x86_64",))
        self.assertEqual(native["sdist_package_dirs"], {"qwentts_cpp": "src/qwentts_cpp"})
        with self.assertRaisesRegex(release_guard.GuardError, "not publishable"):
            release_guard._component_profile("RealtimeTTSQwenNativeCPU", publishing=True)
        with self.assertRaisesRegex(release_guard.GuardError, "realtimetts-qwen-native==0.2.0\\+cpu2"):
            release_guard._validate_package_artifacts(
                Path("."), Path("missing.whl"), Path("missing.tar.gz"), [],
                {"versions": {"realtimetts-qwen-native": "0.2.0"}}, profile,
            )

    def test_overlap_deployment_profiles_stay_private_and_cpu_only(self) -> None:
        for name in ("RealtimeTTSCPUOverlap", "RealtimeTTSQwenNativeCPUOverlap",
                     "RealtimeTTSCPUStartup", "RealtimeTTSQwenNativeCPUStartup"):
            profile = release_guard._component_profile(name)
            self.assertEqual(profile["service_name"], "wwz-qwen3-tts-cpu.service")
            self.assertEqual(profile["signer_fingerprint"],
                             release_guard.COMPONENT_PROFILES["RealtimeTTS"]["signer_fingerprint"])
            with self.assertRaises(release_guard.GuardError):
                release_guard._service_name(profile, "wwz-qwen3-tts.service")
            with self.assertRaisesRegex(release_guard.GuardError, "not publishable"):
                release_guard._component_profile(name, publishing=True)
        self.assertEqual(
            release_guard.COMPONENT_PROFILES["RealtimeTTSCPUOverlap"]["required_dependencies"],
            {"realtimetts-qwen-native-cpu": "0.3.0+overlap20260920"},
        )
        self.assertEqual(
            release_guard.COMPONENT_PROFILES["RealtimeTTSCPUStartup"]["required_dependencies"],
            {"realtimetts-qwen-native-cpu": "0.3.0+startup20260920.1"},
        )
        self.assertEqual(
            release_guard.COMPONENT_PROFILES["RealtimeTTSQwenNativeCPUStartup"]["native_revision"],
            "0b7862a660cafaf550f50b8da880e8eace90d425",
        )

    def test_public_cpu_native_profile_is_separate_and_publishable(self) -> None:
        private = release_guard._component_profile("RealtimeTTSQwenNativeCPU")
        public = release_guard._component_profile(
            "RealtimeTTSQwenNativeCPUPublic", publishing=True
        )
        self.assertFalse(private["publishable"])
        self.assertTrue(public["publishable"])
        self.assertEqual(public["distribution"], "realtimetts-qwen-native-cpu")
        self.assertEqual(
            public["packages"],
            (("qwentts_cpp_cpu", "src/qwentts_cpp_cpu", "qwentts_cpp_cpu"),),
        )
        self.assertEqual(
            public["required_wheel_platforms"],
            (
                "manylinux_2_35_x86_64",
                "win_amd64",
                "macosx_13_0_x86_64",
                "macosx_11_0_arm64",
            ),
        )
        self.assertEqual(
            public["binary_package_prefixes"],
            {"qwentts_cpp_cpu": ("lib/", ".dylibs/")},
        )
        self.assertEqual(
            public["native_revision"],
            "ec5336154a68f9e17f95c3d99b3b97489cb090a6",
        )
        self.assertTrue(public["publish_sdist"])

    def test_remote_branch_and_tag_must_both_equal_release_head(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = self._init_repo(root)
            bare = root / "remote.git"
            self._git(root, "init", "--bare", "--quiet", str(bare))
            self._git(repo, "remote", "add", "origin", str(bare))
            self._git(
                repo,
                "-c",
                "user.name=release-guard-test",
                "-c",
                "user.email=release-guard-test@example.invalid",
                "tag",
                "-a",
                "v1.0",
                "-m",
                "release",
            )
            self._git(repo, "push", "origin", "HEAD:refs/heads/main", "v1.0")
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            profile = {
                "remote_repository": "example.invalid/demo",
                "remote_branch": "main",
            }
            with mock.patch.object(
                release_guard,
                "_canonical_remote_url",
                return_value="example.invalid/demo",
            ):
                result = release_guard._assert_remote_release_refs(
                    repo, "origin", "main", "v1.0", head, "1.0", profile
                )
            self.assertEqual(result["branch"], "refs/heads/main")
            with mock.patch.object(
                release_guard,
                "_canonical_remote_url",
                return_value="example.invalid/demo",
            ), self.assertRaisesRegex(release_guard.GuardError, "does not match"):
                release_guard._assert_remote_release_refs(
                    repo, "origin", "main", "v2.0", head, "1.0", profile
                )

    def test_package_index_confirmation_compares_exact_hashes(self) -> None:
        expected = [
            {"filename": "demo.whl", "sha256": "abc", "size": 123},
            {"filename": "demo.tar.gz", "sha256": "def", "size": 456},
        ]
        response = {
            "urls": [
                {
                    "filename": record["filename"],
                    "digests": {"sha256": record["sha256"]},
                    "size": record["size"],
                }
                for record in expected
            ]
        }
        encoded = json.dumps(response).encode()
        with mock.patch.object(
            release_guard.urllib.request,
            "urlopen",
            return_value=io.BytesIO(encoded),
        ):
            confirmed = release_guard._confirm_published(
                "pypi",
                {"name": "demo", "version": "1.0"},
                expected,
                "",
                1,
            )
        self.assertEqual(confirmed, expected)

        response["urls"].append(
            {
                "filename": "unexpected.zip",
                "digests": {"sha256": "123"},
                "size": 9,
            }
        )
        encoded_with_extra = json.dumps(response).encode()
        with mock.patch.object(
            release_guard.urllib.request,
            "urlopen",
            return_value=io.BytesIO(encoded_with_extra),
        ), mock.patch.object(
            release_guard.time, "monotonic", side_effect=[0.0, 2.0]
        ), mock.patch.object(
            release_guard.time, "sleep"
        ), self.assertRaisesRegex(
            release_guard.GuardError, "artifact set is not exact"
        ):
            release_guard._confirm_published(
                "pypi",
                {"name": "demo", "version": "1.0"},
                expected,
                "",
                1,
            )

    def test_native_platform_wheels_require_exact_python_source_parity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "src" / "qwentts_cpp"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("VERSION = '0.2.0'\n", encoding="utf-8")
            (package / "lib").mkdir()
            (package / "lib" / ".gitkeep").write_bytes(b"")

            platforms = (
                "manylinux_2_35_x86_64",
                "manylinux_2_35_aarch64",
                "win_amd64",
            )
            wheels = []
            for platform_tag in platforms:
                wheel = root / (
                    "realtimetts_qwen_native-0.2.0-py3-none-"
                    f"{platform_tag}.whl"
                )
                with zipfile.ZipFile(wheel, "w") as archive:
                    archive.writestr(
                        "qwentts_cpp/__init__.py", "VERSION = '0.2.0'\r\n"
                    )
                    if platform_tag.startswith("manylinux"):
                        archive.writestr(
                            "qwentts_cpp/lib/libqwen.so", b"b91bca4-native"
                        )
                        archive.writestr(
                            "qwentts_cpp/lib/libggml-cuda.so.0", b"cuda"
                        )
                    else:
                        archive.writestr(
                            "qwentts_cpp/lib/qwen.dll", b"b91bca4-native"
                        )
                        archive.writestr(
                            "qwentts_cpp/lib/ggml-cuda.dll", b"cuda"
                        )
                    archive.writestr(
                        "realtimetts_qwen_native-0.2.0.dist-info/METADATA",
                        "Name: realtimetts-qwen-native\nVersion: 0.2.0\n",
                    )
                    archive.writestr(
                        "realtimetts_qwen_native-0.2.0.dist-info/WHEEL",
                        f"Wheel-Version: 1.0\nTag: py3-none-{platform_tag}\n",
                    )
                wheels.append(wheel)

            profile = release_guard.COMPONENT_PROFILES["RealtimeTTSQwenNative"]
            metadata = release_guard._wheel_metadata(wheels[0])
            specs = [
                {
                    "package_dir": "qwentts_cpp",
                    "source_package_dir": "src/qwentts_cpp",
                    "runtime_module": "qwentts_cpp",
                }
            ]
            release_guard._validate_platform_wheels(
                profile, metadata, wheels[0], wheels[1:], root, specs
            )

            original_windows = wheels[2].read_bytes()
            with zipfile.ZipFile(wheels[2], "w") as archive:
                archive.writestr(
                    "qwentts_cpp/__init__.py", "VERSION = '0.2.0'\r\n"
                )
                archive.writestr(
                    "qwentts_cpp/lib/qwen.dll", b"b91bca4-native"
                )
                archive.writestr(
                    "realtimetts_qwen_native-0.2.0.dist-info/METADATA",
                    "Name: realtimetts-qwen-native\nVersion: 0.2.0\n",
                )
                archive.writestr(
                    "realtimetts_qwen_native-0.2.0.dist-info/WHEEL",
                    "Wheel-Version: 1.0\nTag: py3-none-win_amd64\n",
                )
            with self.assertRaisesRegex(
                release_guard.GuardError, "lacks required native library"
            ):
                release_guard._validate_platform_wheels(
                    profile, metadata, wheels[0], wheels[1:], root, specs
                )
            wheels[2].write_bytes(original_windows)

            with self.assertRaisesRegex(
                release_guard.GuardError, "platform set is not exact"
            ):
                release_guard._validate_platform_wheels(
                    profile, metadata, wheels[0], wheels[1:2], root, specs
                )

            wheel = wheels[0]
            sdist = root / "release.tar.gz"
            extra = wheels[1:]
            self.assertEqual(
                release_guard._publication_artifacts(profile, wheel, sdist, extra),
                [wheel, *extra],
            )
            self.assertEqual(
                release_guard._publication_artifacts(
                    release_guard.COMPONENT_PROFILES["RealtimeTTS"],
                    wheel,
                    sdist,
                    extra,
                ),
                [wheel, sdist, *extra],
            )

    def test_sdist_name_and_version_must_match_wheel(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = self._init_repo(root)
            wheel, sdist = self._write_artifacts(root)
            specs = [
                {
                    "package_dir": "demo_pkg",
                    "source_package_dir": "demo_pkg",
                    "runtime_module": "demo_pkg",
                },
                {
                    "package_dir": "demo_server",
                    "source_package_dir": "src/demo_server",
                    "runtime_module": "demo_server",
                },
            ]
            release_guard._validate_package_artifacts(
                repo,
                wheel,
                sdist,
                specs,
                None,
                release_guard.COMPONENT_PROFILES["demo"],
            )

            with tarfile.open(sdist, "w:gz") as archive:
                metadata = b"Name: demo-distribution\nVersion: 2.0\n"
                info = tarfile.TarInfo("demo_distribution-2.0/PKG-INFO")
                info.size = len(metadata)
                archive.addfile(info, io.BytesIO(metadata))
            with self.assertRaisesRegex(
                release_guard.GuardError, "name/version metadata differ"
            ):
                release_guard._validate_package_artifacts(
                    repo,
                    wheel,
                    sdist,
                    specs,
                    None,
                    release_guard.COMPONENT_PROFILES["demo"],
                )


if __name__ == "__main__":
    unittest.main()
