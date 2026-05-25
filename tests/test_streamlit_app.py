"""
Streamlit App Test Suite
=========================

Basic tests to verify the Streamlit application components.

Run with: pytest test_streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))


def test_imports():
    """Test that all page modules can be imported."""
    try:
        from app.pages import (
            data_selection,
            feature_engineering,
            fsp_selection,
            model_training,
            predictions_viz,
            model_comparison
        )
        assert True
    except ImportError as e:
        assert False, f"Import failed: {e}"


def test_config_loading():
    """Test configuration loading."""
    from src.config_loader import load_config

    config = load_config()
    assert config is not None
    assert 'data' in config
    assert 'training' in config


def test_data_preprocessing_imports():
    """Test data preprocessing module imports."""
    from src.data.preprocessing import (
        pivot_fsp_data,
        get_fsp_forecast_columns,
        FSP_PROVIDERS
    )

    assert FSP_PROVIDERS is not None
    assert len(FSP_PROVIDERS) > 0


def test_feature_engineering_imports():
    """Test feature engineering module imports."""
    from src.features.feature_engineering import (
        create_temporal_split,
        create_rolling_features,
        create_time_features,
        encode_categorical_features
    )

    assert True


def test_streamlit_app_structure():
    """Test that main app file exists and has required components."""
    app_file = PROJECT_DIR / "streamlit_app.py"
    assert app_file.exists(), "streamlit_app.py not found"

    content = app_file.read_text()
    assert "streamlit" in content
    assert "def main()" in content
    assert "initialize_session_state" in content


def test_page_modules_exist():
    """Test that all page modules exist."""
    pages = [
        'data_selection',
        'feature_engineering',
        'fsp_selection',
        'model_training',
        'predictions_viz',
        'model_comparison'
    ]

    for page in pages:
        page_file = PROJECT_DIR / "app" / "pages" / f"{page}.py"
        assert page_file.exists(), f"{page}.py not found"

        # Check for show() function
        content = page_file.read_text()
        assert "def show()" in content, f"show() function not found in {page}.py"


def test_requirements_file():
    """Test that requirements.txt includes streamlit."""
    req_file = PROJECT_DIR / "requirements.txt"
    assert req_file.exists(), "requirements.txt not found"

    content = req_file.read_text()
    assert "streamlit" in content.lower(), "streamlit not in requirements.txt"
    assert "plotly" in content.lower(), "plotly not in requirements.txt"


if __name__ == "__main__":
    print("Running Streamlit App Tests...")
    print("=" * 70)

    tests = [
        ("Imports", test_imports),
        ("Config Loading", test_config_loading),
        ("Data Preprocessing", test_data_preprocessing_imports),
        ("Feature Engineering", test_feature_engineering_imports),
        ("App Structure", test_streamlit_app_structure),
        ("Page Modules", test_page_modules_exist),
        ("Requirements", test_requirements_file)
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            test_func()
            print(f" {name}")
            passed += 1
        except AssertionError as e:
            print(f" {name}: {e}")
            failed += 1
        except Exception as e:
            print(f" {name}: Unexpected error - {e}")
            failed += 1

    print("=" * 70)
    print(f"\nResults: {passed} passed, {failed} failed")

    if failed == 0:
        print(" All tests passed!")
        sys.exit(0)
    else:
        print(" Some tests failed")
        sys.exit(1)
