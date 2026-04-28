"""
CSV file reading utility module
Responsible for reading harmful behavior data from CSV files
"""
import csv
import os
from typing import List, Dict, Any
from pathlib import Path


class CSVReader:
    """CSV file reader, specifically for reading harmful behavior data"""
    
    def __init__(self, file_path: str):
        """
        Initialize CSV reader
        
        Args:
            file_path: Path to CSV file
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file does not exist: {file_path}")
    
    def read_harmful_behaviors(self) -> List[Dict[str, str]]:
        """
        Read harmful behavior data
        
        Supports two CSV formats:
        1. Old format: Contains 'goal' and 'target' columns
        2. New format: Contains 'query' column (mapped to 'goal'), 'target' is empty string
        
        Returns:
            List of dictionaries containing harmful behavior data, each dict contains 'goal' and 'target' fields
        """
        harmful_behaviors = []
        
        try:
            for encoding in ('utf-8', 'utf-8-sig', 'latin-1', 'cp1252'):
                try:
                    with open(self.file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            else:
                raise UnicodeDecodeError('all', b'', 0, 1, 'Could not decode file with any supported encoding')

            import io
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                # Support new format (query column) and old format (goal column)
                if 'query' in row:
                    goal = row['query'].strip()
                    target = row.get('target', '').strip() if 'target' in row else ''
                elif 'goal' in row:
                    goal = row['goal'].strip()
                    target = row.get('target', '').strip() if 'target' in row else ''
                else:
                    raise ValueError(
                        f"CSV file format error: Missing 'goal' or 'query' column. "
                        f"Available columns: {list(row.keys())}"
                    )

                harmful_behaviors.append({
                    'goal': goal,
                    'target': target
                })
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV file: {str(e)}")
        
        return harmful_behaviors
    
    def read_single_goal(self, index: int = 0) -> str:
        """
        Read a single harmful goal
        
        Args:
            index: Index of the goal to read (default 0, i.e., the first one)
        
        Returns:
            Text content of the harmful goal
        """
        behaviors = self.read_harmful_behaviors()
        if index >= len(behaviors):
            raise IndexError(f"Index {index} out of range (total {len(behaviors)} records)")
        return behaviors[index]['goal']

