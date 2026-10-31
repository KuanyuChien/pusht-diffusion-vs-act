"""Run LeRobot's migrate_policy_normalization.py with a fix for old configs.

LeRobot's migration script builds the policy config from raw config.json values. JSON has no
tuples, so fields typed as tuples (e.g. DiffusionConfig.crop_shape = [84, 84]) stay lists, and
saving the config fails with `Couldn't encode 84`. This wrapper decodes every list value to the
field's declared type (the way normal config loading does) before the config is built.

Usage: same arguments as migrate_policy_normalization.py, e.g.
    python migrate_official.py --pretrained-path lerobot/diffusion_pusht --output-dir <dir>
"""

from typing import get_type_hints

import draccus

import lerobot.processor.migrate_policy_normalization as migrate
from lerobot.configs.policies import PreTrainedConfig

_make_policy_config = migrate.make_policy_config


def make_policy_config(policy_type, **kwargs):
    hints = get_type_hints(PreTrainedConfig.get_choice_class(policy_type))
    for name, value in kwargs.items():
        if isinstance(value, list) and name in hints:
            kwargs[name] = draccus.decode(hints[name], value)
    return _make_policy_config(policy_type, **kwargs)


migrate.make_policy_config = make_policy_config

if __name__ == "__main__":
    migrate.main()
