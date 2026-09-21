"""Realism/quality profile loader.

Code reads a `Quality` object, never hardware. Switching hardware means
switching the active profile name (`fast` -> `balanced` -> `high` -> `reference`),
optionally with point overrides.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import yaml

_DEFAULT_PROFILES_PATH = Path(__file__).parent / "default_profiles.yaml"


@dataclass
class Quality:
    profile: str
    render: dict
    sun: dict
    material: dict
    camera: dict
    lidar: dict
    terrain: dict
    rocks: dict
    dust: dict


def load_quality(path: str | None = None, profiles_path: str | None = None) -> Quality:
    """Load a Quality config.

    `path` points to a user config with a top-level `quality:` block
    (`profile:` name + optional `overrides:`); if omitted, defaults to
    `profile: fast` with no overrides. `profiles_path` points to the YAML
    defining the named profiles (defaults to the bundled reference set).
    """
    profiles_file = Path(profiles_path) if profiles_path else _DEFAULT_PROFILES_PATH
    with open(profiles_file) as f:
        profile_defs = yaml.safe_load(f)["profiles"]

    if path:
        with open(path) as f:
            cfg = yaml.safe_load(f)["quality"]
    else:
        cfg = {"profile": "fast", "overrides": {}}

    profile_name = cfg["profile"]
    if profile_name not in profile_defs:
        raise ValueError(f"unknown quality profile: {profile_name!r} (have: {list(profile_defs)})")

    q = copy.deepcopy(profile_defs[profile_name])
    for key, val in cfg.get("overrides", {}).items():
        section, name = key.split(".", 1)
        if section not in q:
            raise ValueError(f"override section {section!r} not in profile {profile_name!r}")
        q[section][name] = val

    return Quality(profile=profile_name, **q)


def quality_from_dict(cfg: dict, profiles_path: str | None = None) -> Quality:
    """Same as load_quality but `cfg` is an already-parsed `{"quality": {...}}` dict."""
    profiles_file = Path(profiles_path) if profiles_path else _DEFAULT_PROFILES_PATH
    with open(profiles_file) as f:
        profile_defs = yaml.safe_load(f)["profiles"]

    q_cfg = cfg["quality"]
    profile_name = q_cfg["profile"]
    q = copy.deepcopy(profile_defs[profile_name])
    for key, val in q_cfg.get("overrides", {}).items():
        section, name = key.split(".", 1)
        q[section][name] = val
    return Quality(profile=profile_name, **q)
