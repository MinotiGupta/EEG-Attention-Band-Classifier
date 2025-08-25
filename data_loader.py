from pathlib import Path
import mne
from typing import List
import os


class DataLoader:
    """Handles loading and finding EEG data files"""

    def __init__(self):
        self.supported_extensions = ['.bdf', '.BDF']

    def find_bdf_files(self, root_dir: str) -> List[str]:
        """
        Recursively find all .bdf files in the directory structure

        Args:
            root_dir: Root directory to search

        Returns:
            List of paths to .bdf files
        """
        root_path = Path(root_dir)
        bdf_files = []

        if not root_path.exists():
            print(f"Directory does not exist: {root_dir}")
            return bdf_files

        print(f"Scanning directory: {root_dir}")

        # Recursively find all .bdf files
        for ext in self.supported_extensions:
            found_files = list(root_path.rglob(f"*{ext}"))
            bdf_files.extend(found_files)
            print(f"Found {len(found_files)} files with extension {ext}")

        # Convert to strings and sort
        bdf_files = [str(f) for f in bdf_files]
        bdf_files.sort()

        print(f"Total .bdf files found: {len(bdf_files)}")

        # Print first few files for debugging
        if bdf_files:
            print("Sample files found:")
            for i, file in enumerate(bdf_files[:5]):
                print(f"  {i + 1}. {file}")
            if len(bdf_files) > 5:
                print(f"  ... and {len(bdf_files) - 5} more files")

        return bdf_files

    def get_file_info(self, file_path: str) -> dict:
        """
        Get basic information about a .bdf file without loading all data

        Args:
            file_path: Path to .bdf file

        Returns:
            Dictionary with file information
        """
        try:
            # Read info without loading data
            info = mne.io.read_raw_bdf(file_path, preload=False, verbose=False).info

            return {
                'channels': len(info['ch_names']),
                'sampling_rate': info['sfreq'],
                'channel_names': info['ch_names'],
                'file_size': os.path.getsize(file_path)
            }
        except Exception as e:
            return {'error': str(e)}

    def validate_bdf_file(self, file_path: str) -> bool:
        """
        Validate if a .bdf file can be loaded

        Args:
            file_path: Path to .bdf file

        Returns:
            True if file is valid, False otherwise
        """
        try:
            # Try to read the file header
            mne.io.read_raw_bdf(file_path, preload=False, verbose=False)
            return True
        except Exception as e:
            print(f"Invalid file {file_path}: {e}")
            return False

    def get_directory_structure(self, root_dir: str) -> dict:
        """
        Get the directory structure for display

        Args:
            root_dir: Root directory to analyze

        Returns:
            Dictionary with directory structure info
        """
        root_path = Path(root_dir)

        if not root_path.exists():
            return {'error': 'Directory does not exist'}

        structure = {
            'total_files': 0,
            'total_bdf_files': 0,
            'subdirectories': [],
            'files_by_extension': {}
        }

        for item in root_path.rglob('*'):
            if item.is_file():
                structure['total_files'] += 1
                ext = item.suffix.lower()

                if ext in ['.bdf']:
                    structure['total_bdf_files'] += 1

                if ext in structure['files_by_extension']:
                    structure['files_by_extension'][ext] += 1
                else:
                    structure['files_by_extension'][ext] = 1

            elif item.is_dir() and item != root_path:
                rel_path = item.relative_to(root_path)
                structure['subdirectories'].append(str(rel_path))

        return structure
