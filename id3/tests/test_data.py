from id3.data import load_contact_lenses


def test_contact_lenses():
    X, y, targets = load_contact_lenses()
    assert X.shape == (24, 4)
    assert y.shape == (24,)
    assert targets.shape == (5,)
