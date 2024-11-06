import contextlib
import io
import logging
import os

from autogluon.tabular import TabularDataset

from ..constants import TEXT_EXTENSIONS


def is_text_file(filename):
    _, ext = os.path.splitext(filename)
    return ext.lower() in TEXT_EXTENSIONS


@contextlib.contextmanager
def suppress_tabular_logs():
    """Context manager to suppress logging output from TabularDataset"""
    # Save the current logging level
    logger = logging.getLogger()
    old_level = logger.getEffectiveLevel()

    # Temporarily increase logging level to suppress INFO logs
    logger.setLevel(logging.WARNING)

    # Redirect stdout to capture any print statements
    temp_stdout = io.StringIO()
    with contextlib.redirect_stdout(temp_stdout):
        try:
            yield
        finally:
            # Restore the original logging level
            logger.setLevel(old_level)


def load_pd_quietly(filename):
    with suppress_tabular_logs():
        return TabularDataset(filename)
