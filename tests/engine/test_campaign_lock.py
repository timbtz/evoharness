from pathlib import Path

import pytest

from evoharness.engine.campaign_lock import CampaignLock


def test_campaign_lock_rejects_second_evaluator(tmp_path: Path):
    path = tmp_path / "stellar.lock"
    with CampaignLock(path):
        with pytest.raises(RuntimeError, match="another Stellar"):
            with CampaignLock(path):
                pass
    with CampaignLock(path):
        pass
