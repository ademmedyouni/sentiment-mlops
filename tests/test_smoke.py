import sys
import os
sys.path.insert(0, os.path.abspath("."))

from src.preprocess import clean


def test_clean_removes_html():
    assert clean("<b>Hello</b> WORLD!") == "hello world!"


def test_clean_handles_none():
    assert clean(None) == ""


def test_clean_strips_whitespace():
    assert clean("  hello  ") == "hello"
