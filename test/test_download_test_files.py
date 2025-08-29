import sys
import shutil
import tempfile
import types
import zipfile
from pathlib import Path
from unittest import TestCase, mock

import test.data_setup as data_setup


class TestAdditionalTestFileDownload(TestCase):
    """Ensure additional test files can be downloaded."""

    def setUp(self) -> None:
        self.temp_parent = Path(tempfile.mkdtemp())
        self.test_dir = self.temp_parent / "test_files"
        self.required_files = [self.test_dir / "ear_eog_raw.fif"]
        self.dir_patch = mock.patch.object(data_setup, "TEST_FILES_DIR", self.test_dir)
        self.files_patch = mock.patch.object(
            data_setup, "TEST_REQUIRED_FILES", self.required_files
        )
        self.dir_patch.start()
        self.files_patch.start()
        self.original_gdown = sys.modules.get("gdown")
        sys.modules["gdown"] = types.SimpleNamespace(download=self._fake_download)

    def tearDown(self) -> None:
        self.dir_patch.stop()
        self.files_patch.stop()
        if self.original_gdown is not None:
            sys.modules["gdown"] = self.original_gdown
        else:
            sys.modules.pop("gdown", None)
        shutil.rmtree(self.temp_parent, ignore_errors=True)

    def _fake_download(self, url: str, output: str, quiet: bool = False) -> str:
        """Create a zip archive with test files for extraction."""
        with zipfile.ZipFile(output, "w") as zf:
            for file in self.required_files:
                zf.writestr(f"test_files/{file.name}", "data")
        return output

    def test_download_and_extract(self) -> None:
        """Ensure test files are downloaded to the expected directory."""
        data_setup.download_test_files()
        for path in self.required_files:
            self.assertTrue(path.exists(), f"File {path.name} was not downloaded")
