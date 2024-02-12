import pupil_labs.recover_recording as this_project


def test_package_metadata() -> None:
    assert hasattr(this_project, "__version__")
