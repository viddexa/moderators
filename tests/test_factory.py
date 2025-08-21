def test_factory_mappings():
    from moderators import Moderator

    archs = set(Moderator._ARCH_TO_CLASS_PATH.keys())
    assert "TransformersModerator" in archs
    assert "UltralyticsModerator" in archs
    assert "OnnxModerator" in archs
