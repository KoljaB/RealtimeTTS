"""Public CPU releases retain exact live-runtime and artifact requirements."""
from pathlib import Path

import pytest

from tools import release_guard


def test_public_cpu_uses_real_service_and_exact_candidate_dependency():
    profile = release_guard._component_profile("RealtimeTTSPublicCPU", publishing=True)
    assert profile["service_required"] is True
    assert profile["service_name"] == "wwz-qwen3-tts-cpu.service"
    assert profile["remote_branch"] == "master"
    assert profile["required_dependencies"] == {"realtimetts-qwen-native-cpu": "0.3.0"}
    assert profile["signer_fingerprint"] == release_guard.COMPONENT_PROFILES["RealtimeTTS"]["signer_fingerprint"]
    with pytest.raises(release_guard.GuardError, match="requires realtimetts-qwen-native-cpu"):
        release_guard._validate_package_artifacts(
            Path("."), Path("missing.whl"), Path("missing.tar.gz"), [],
            {"versions": {"realtimetts-qwen-native-cpu": "0.3.0rc0"}}, profile,
        )


def test_new_cpu_publication_keeps_source_distribution_required():
    profile = release_guard._component_profile("RealtimeTTSQwenNativeCPUPublic", publishing=True)
    wheel, sdist = Path("candidate.whl"), Path("candidate.tar.gz")
    assert release_guard._publication_artifacts(profile, wheel, sdist, []) == [wheel, sdist]
    for private in ("RealtimeTTSCPU", "RealtimeTTSQwenNativeCPU"):
        with pytest.raises(release_guard.GuardError, match="not publishable"):
            release_guard._component_profile(private, publishing=True)
