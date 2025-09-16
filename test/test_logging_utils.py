"""Tests for centralized logging utilities."""
from __future__ import annotations

import logging
import importlib

from pyblinker.logging import get_logger, set_log_level, set_log_file, verbose


def test_set_log_level_and_coercion():
    """set_log_level should accept multiple verbosity representations."""
    logger = get_logger()
    old = set_log_level("ERROR", return_old_level=True)
    assert old is not None
    assert logger.level == logging.ERROR
    set_log_level(True)
    assert logger.level == logging.INFO


def test_environment_default(monkeypatch):
    """Environment variable should initialize the log level when unset."""
    monkeypatch.setenv("APP_LOGGING_LEVEL", "ERROR")
    import sys

    sys.modules.pop("pyblinker.logging", None)
    logging_mod = importlib.import_module("pyblinker.logging")
    logger = logging_mod.get_logger()
    assert logger.level == logging.ERROR


def test_verbose_context_manager(caplog):
    """verbose used as a context manager should temporarily elevate level."""
    logger = get_logger("pyblinker.test")
    root = get_logger()
    root.propagate = True
    set_log_level("WARNING")
    with caplog.at_level(logging.DEBUG):
        with verbose("DEBUG"):
            logger.debug("msg")
    root.propagate = False
    assert "msg" in caplog.text


def test_verbose_decorator(caplog):
    """@verbose should enable per-call level overrides."""

    @verbose
    def _func(*, verbose=None):
        logger = get_logger("pyblinker.test")
        logger.debug("inside")

    root = get_logger()
    root.propagate = True
    with caplog.at_level(logging.DEBUG):
        _func(verbose="DEBUG")
    root.propagate = False
    assert "inside" in caplog.text


def test_set_log_file(tmp_path):
    """set_log_file should redirect output exclusively to the file."""
    log_file = tmp_path / "out.log"
    logger = get_logger()
    set_log_file(log_file)
    logger.warning("file")
    set_log_file(None)
    assert "file" in log_file.read_text()


def test_child_logger_emits_once():
    """Child loggers should not duplicate log records."""
    import io

    stream = io.StringIO()
    root = get_logger()
    set_log_level("INFO")
    handler = logging.StreamHandler(stream)
    root.addHandler(handler)
    try:
        child = get_logger("pyblinker.child")
        child.info("hello")
        handler.flush()
        assert stream.getvalue().count("hello") == 1
    finally:
        root.removeHandler(handler)
