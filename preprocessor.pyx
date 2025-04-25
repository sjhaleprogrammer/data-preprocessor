# cython: language_level=3
# distutils: language = c

import re
import pandas as pd
import logging
import warnings
import time
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from typing import List, Dict, Any, Optional, Tuple
import os
import functools
import multiprocessing
import platform
import sys
import gc
import psutil
import math
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import shutil

# Cython imports
cimport cython
# No C standard library imports needed directly in this version
# from libc.stdlib cimport malloc, free
# from libc.string cimport strlen, strcpy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(process)d - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=MarkupResemblesLocatorWarning)
# Filter the specific ChainedAssignmentError warning if it becomes too noisy during development
# warnings.filterwarnings('ignore', category=FutureWarning, message=".*ChainedAssignmentError.*")

# --- Constants ---
ENCODINGS: List[str] = ['utf-8', 'latin-1', 'cp1252']
MEMORY_SAFETY_FACTOR: float = 0.8
MIN_MEMORY_PER_WORKER_GB: float = 0.4
DEFAULT_AVAILABLE_MEMORY_GB: float = 4.0
MIN_BATCH_SIZE: cython.int = 200
MAX_BATCH_SIZE: cython.int = 15000
FILE_PROCESSING_CHUNK_SIZE: cython.int = 100
BATCH_PROCESSING_CHUNK_SIZE: cython.int = 100
EXPORT_CHUNK_SIZE: cython.int = 100000
TRAIN_SPLIT_RATIO: float = 1
RANDOM_SEED: cython.int = 42
TEMP_DIR_BASE: str = "temp_parser_batches"
PREVIEW_ROWS: cython.int = 5 # Number of rows to show in preview

# Dtypes
DTYPES_CONTENT: Dict[str, Any] = {
    'id': str, 'content': str, 'file': str,
    'file_order': np.uint32, 'line_num': np.uint32, 'sequence': np.int64
}
DTYPES_SOURCES: Dict[str, Any] = {
    'id': str, 'words': str, 'country': str,
    'genre': 'category', 'source': str, 'title': str
}
DTYPES_MERGED: Dict[str, Any] = {**DTYPES_CONTENT, **DTYPES_SOURCES}

# Precompiled Regex
_RE_MULTI_SPACE = re.compile(r'\s+')
_RE_UNK = re.compile(r'(@\s*)+')
# Refined cleanup patterns
_RE_CLEANUP = {
    # General spacing before punctuation
    re.compile(r" \?"): "?",
    re.compile(r" !"): "!",
    re.compile(r" \."): ".", # Note: Escaped dot
    re.compile(r" ,"): ",",
    re.compile(r" ;"): ";",
    re.compile(r" :"): ":",
    # Specific cases
    re.compile(r" --"): "",
    re.compile(r" \.\. \."): "...", # Or rely on multi-space cleanup? Specific is safer.
    re.compile(r" n't"): "n't",
    re.compile(r" &"): "&",
     # --- Rules for spaces INSIDE quotes and parentheses ---
    re.compile(r'\(\s+'): '(', # Remove space(s) after (
    re.compile(r'\s+\)'): ')', # Remove space(s) before )
    #re.compile(r'"\s+'): '"', # Remove space(s) after "
    #re.compile(r"\s+\""): '"', # Remove space(s) before "
    re.compile(r"'\s+"): "'", # Remove space(s) after '
    re.compile(r"\s+'"): "'", # Remove space(s) before '
    # Note: The rule re.compile(r" '"): "'" might be redundant now but harmless
} 


# --- Platform & Resource Utils ---

@cython.cfunc
@cython.inline
def get_platform_info() -> str:
    try:
        s = platform.system()
        if s == "Linux": return "WSL" if "microsoft" in platform.uname().release.lower() else "Linux"
        return s
    except Exception: return "Unknown"

@cython.cfunc
def get_available_memory() -> cython.double:
    try: return max(0.1, psutil.virtual_memory().available / (1024.0 ** 3))
    except Exception: return DEFAULT_AVAILABLE_MEMORY_GB

def configure_multiprocessing() -> None:
    """Sets the multiprocessing start method based on platform."""
    platform_type: str = get_platform_info()
    # Use 'fork' on Unix-like systems (Linux, macOS, WSL) as requested.
    # 'spawn' is required on Windows.
    if platform_type in ["Linux", "WSL", "Darwin"]: # Darwin is macOS
        desired_method: str = 'fork'
    elif platform_type == "Windows":
         desired_method: str = 'spawn'
    else:
        logger.warning(f"Unknown platform '{platform_type}', defaulting to 'spawn' start method.")
        desired_method: str = 'spawn'

    try:
        current_method = multiprocessing.get_start_method(allow_none=True)
        if current_method is None or current_method != desired_method:
            # Force setting only if changing method OR if current is None
            force_set = current_method is not None and current_method != desired_method
            multiprocessing.set_start_method(desired_method, force=force_set)
            logger.info(f"Set multiprocessing start method to '{desired_method}' (Platform: {platform_type})")
        else:
             logger.info(f"Multiprocessing start method already set to '{current_method}'")

        # freeze_support() is crucial for creating executables on Windows
        if platform_type == "Windows" and getattr(sys, 'frozen', False):
             multiprocessing.freeze_support()

    except ValueError as e: # Catch specific error if method is already set and force=False
        logger.warning(f"Could not set start method to '{desired_method}' (may be already set?): {e}. Using: {multiprocessing.get_start_method(allow_none=True)}")
    except Exception as e:
        logger.error(f"Failed configuring multiprocessing '{desired_method}': {e}. Using method: {multiprocessing.get_start_method(allow_none=True)}")
        if platform_type == "Windows" and getattr(sys, 'frozen', False):
             multiprocessing.freeze_support() # Still attempt freeze_support

@cython.cdivision(True)
def calculate_processing_parameters(total_items: cython.int, item_type: str = "lines") -> Tuple[cython.int, cython.int]:
    available_memory_gb: float = get_available_memory()
    logical_cpu_count: int = psutil.cpu_count(logical=True) or 2
    est_mem_kb = 15 if item_type == "lines" else 512
    max_mem_batch_gb = available_memory_gb * 0.10
    calc_batch = <cython.int>((max_mem_batch_gb * 1024**2) / est_mem_kb) if est_mem_kb > 0 else MAX_BATCH_SIZE
    # batch_size = max(MIN_BATCH_SIZE, min(calc_batch, MAX_BATCH_SIZE))
    batch_size = 1000

    mem_workers = max(1, <cython.int>((available_memory_gb * MEMORY_SAFETY_FACTOR) / MIN_MEMORY_PER_WORKER_GB))
    cpu_workers = max(1, logical_cpu_count - 1)
    workers = max(1, min(mem_workers, cpu_workers))
    workers = max(1, min(workers, (total_items + batch_size - 1) // batch_size, total_items))

    logger.info(f"Params ({available_memory_gb:.1f}GB RAM, {logical_cpu_count} CPU): batch={batch_size}, workers={workers}")
    return batch_size, workers

# --- Text Cleaning ---

@functools.lru_cache(maxsize=16384)
@cython.cfunc
@cython.inline
def clean_text_cython(text_obj: str) -> str:
    # Use 'object' for BeautifulSoup result as it's a Python object
    cdef object soup, pattern, replacement
    # Use 'str' for known string types
    cdef str cleaned_text, new_text, text_before_bs

    if not isinstance(text_obj, str): text_obj = str(text_obj)

    # 1. Initial strip and @ mentions
    cleaned_text = text_obj.strip()
    cleaned_text = _RE_UNK.sub('<unk>', cleaned_text)
    text_before_bs = cleaned_text # Store before potential BS modification

    # 2. BeautifulSoup for HTML stripping
    try:
        soup = BeautifulSoup(cleaned_text, 'html.parser')
        new_text = soup.get_text()
        # Only update if BS actually changed the text AND the result is not empty
        if new_text != text_before_bs and new_text:
            cleaned_text = new_text
            # *Crucial*: Re-run basic whitespace cleanup after HTML stripping
            cleaned_text = _RE_MULTI_SPACE.sub(' ', cleaned_text).strip()
    except Exception as e:
        # Log warning only if input likely contained HTML tags
        if '<' in text_before_bs and '>' in text_before_bs:
             logger.warning(f"BSoup cleaning failed (text: '{text_before_bs[:50]}...'): {e}")

    # 3. Apply specific pattern replacements from _RE_CLEANUP
    # This now includes the quote and parenthesis spacing rules
    for pattern, replacement in _RE_CLEANUP.items():
         cleaned_text = pattern.sub(replacement, cleaned_text)

    # 4. Final pass to catch any multi-spaces possibly introduced by replacements
    cleaned_text = _RE_MULTI_SPACE.sub(' ', cleaned_text)

    # 5. Final strip
    return cleaned_text.strip()

# --- File Handling & Parsing ---

@cython.boundscheck(False)
@cython.wraparound(False)
def read_file_lines(file_path: str) -> Tuple[Optional[List[str]], str]:
    if not os.path.exists(file_path): return None, ""
    cdef str encoding, used_encoding = ""
    cdef list lines = None
    cdef object f # File handle is a Python object
    for encoding in ENCODINGS:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
            used_encoding = encoding
            break
        except (UnicodeDecodeError, TypeError): continue
        except Exception as e:
            logger.error(f"Error reading {file_path} with {encoding}: {e}")
            return None, ""
    if lines is None:
        logger.warning(f"Could not decode {file_path} with {ENCODINGS}")
    return lines, used_encoding

@cython.boundscheck(False)
@cython.wraparound(False)
def parse_sources_file(sources_file: str) -> pd.DataFrame:
    # Declare df as Python object within cdef if needed, otherwise rely on Python typing
    cdef list lines = [] # Initialize
    cdef list sources_data = []
    cdef str encoding = "" # Initialize
    cdef str line = "" # Initialize
    cdef list parts = [] # Initialize
    cdef object df # Use object for DataFrame

    lines, encoding = read_file_lines(sources_file)
    # Return type hint handles the expected output type for Python
    if lines is None: return pd.DataFrame(columns=DTYPES_SOURCES.keys()).astype(DTYPES_SOURCES)

    logger.info(f"Parsing sources file {sources_file} (encoding: {encoding})")
    for line in lines:
        line = line.strip()
        if not line: continue
        
        parts = line.split(maxsplit=5)
        if len(parts) == 6:
            sources_data.append({'id': parts[0], 'words': parts[1], 'country': parts[2],
                               'genre': parts[3], 'source': parts[4], 'title': parts[5]})
        else:
            logger.warning(f"Skipping malformed line in sources: '{line[:80]}...'")

    if not sources_data: return pd.DataFrame(columns=DTYPES_SOURCES.keys()).astype(DTYPES_SOURCES)

    try:
        # df is assigned a Python object here
        df = pd.DataFrame(sources_data).astype(DTYPES_SOURCES)
        logger.info(f"Loaded {len(df)} sources. RAM: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
        return df
    except Exception as e:
        logger.error(f"Error creating sources DataFrame: {e}")
        return pd.DataFrame(columns=DTYPES_SOURCES.keys()).astype(DTYPES_SOURCES)

# --- Parallel Content Collection ---

@cython.boundscheck(False)
@cython.wraparound(False)
def process_single_file(file_info_tuple: Tuple[str, str, int, int]) -> Tuple[bool, List[Dict[str, Any]], str]:
    cdef:
        str root_dir = file_info_tuple[0]
        str filename = file_info_tuple[1]
        cython.uint file_order = <cython.uint>file_info_tuple[2]
        cython.longlong sequence_base = <cython.longlong>file_info_tuple[3]
        str file_path = os.path.join(root_dir, filename)
        list content = None # Initialized
        list result_lines = []
        str encoding = ""
        str line = ""
        str line_stripped = ""
        cython.uint line_num = 0
        cython.int sequence_counter = 0

    try:
        content, encoding = read_file_lines(file_path)
        if content is None: return False, [], file_path

        for line_num, line in enumerate(content):
            line_stripped = line.strip()
            if line_stripped.startswith('##'):
                sequence_counter += 1
                result_lines.append({
                    'line': line_stripped,
                    'file_order': file_order,
                    'line_num': <cython.uint>line_num,
                    'sequence': sequence_base + sequence_counter
                })
        return True, result_lines, file_path
    except Exception as e:
        logger.error(f"Error processing {file_path} (worker {os.getpid()}): {e}")
        return False, [], file_path

@cython.boundscheck(False)
@cython.wraparound(False)
def collect_content_lines_parallel(input_folder: str, num_workers: cython.int) -> List[Dict[str, Any]]:
    cdef:
        list file_infos = []
        cython.int file_order = 0, total_files = 0
        str root = "" # Initialize
        str filename = "" # Initialize
        list txt_files = []
        list all_lines = []
        cython.int processed_files = 0, skipped_files = 0
        cython.int actual_workers = 0 # Initialize
        cython.int chunk_size_files = 0 # Initialize
        cython.int num_file_chunks = 0 # Initialize
        cython.int chunk_idx = 0 # Initialize
        cython.int start_idx = 0 # Initialize
        list file_chunk = []
        object pool # Use object for multiprocessing.Pool
        bint success = False # Initialize
        list lines = []
        str file_path_processed = ""
        cython.longlong SEQUENCE_GAP = 2_000_000
        object results_iterator # Use object for iterator type

    logger.info(f"Collecting content from: {input_folder}")
    for root, _, files in os.walk(input_folder):
        txt_files = sorted([f for f in files if f.lower().endswith('.txt')])
        for filename in txt_files:
            file_infos.append((root, filename, file_order, file_order * SEQUENCE_GAP))
            file_order += 1
    total_files = len(file_infos)

    if total_files == 0: logger.warning("No .txt files found."); return []
    logger.info(f"Found {total_files} .txt files.")

    _, actual_workers = calculate_processing_parameters(total_files, "files")
    if num_workers > 0: actual_workers = max(1, min(num_workers, actual_workers))
    chunk_size_files = max(1, min(FILE_PROCESSING_CHUNK_SIZE, (total_files + actual_workers - 1) // actual_workers))
    num_file_chunks = (len(file_infos) + chunk_size_files - 1) // chunk_size_files
    logger.info(f"Using {actual_workers} workers, {num_file_chunks} file chunks...")

    for chunk_idx in range(num_file_chunks):
        start_idx = chunk_idx * chunk_size_files
        file_chunk = file_infos[start_idx : min(start_idx + chunk_size_files, total_files)]
        if not file_chunk: continue
        logger.info(f"Processing file chunk {chunk_idx + 1}/{num_file_chunks}...")
        try:
            # Use 'object' type hint for pool for robustness
            with multiprocessing.Pool(processes=actual_workers, maxtasksperchild=25) as pool:
                results_iterator = pool.imap_unordered(process_single_file, file_chunk, chunksize=1)
                # Explicitly type loop variables if needed, though Cython often infers correctly
                for success, lines, file_path_processed in results_iterator:
                    if success:
                        # line_dict is implicitly created in the loop
                        for line_dict in lines: line_dict['file'] = file_path_processed
                        all_lines.extend(lines)
                        processed_files += 1
                    else: skipped_files += 1
        except Exception as e:
            logger.error(f"Error processing file chunk {chunk_idx + 1}: {e}")
            skipped_files += len(file_chunk)
        finally:
            del file_chunk; gc.collect()

    if not all_lines: logger.warning("No '@@' lines collected."); return []
    logger.info(f"Collected {len(all_lines)} lines (Processed: {processed_files}, Skipped: {skipped_files}). Sorting...")
    all_lines.sort(key=lambda x: x['sequence'])
    logger.info("Sorting complete.")
    return all_lines

# --- Content Parsing (from Collected Lines) ---

@cython.boundscheck(False)
@cython.wraparound(False)
def process_content_batch(batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cdef list rows = []
    cdef dict line_dict # Loop variable type
    cdef str line, content_raw, content_cleaned, line_id
    cdef cython.Py_ssize_t space_index
    for line_dict in batch_data:
        line = line_dict['line']
        space_index = line.find(' ', 2)
        if space_index != -1:
            line_id = line[2:space_index]
            content_raw = line[space_index+1:]
            content_cleaned = clean_text_cython(content_raw)
            rows.append({
                'id': line_id, 'content': content_cleaned,
                'file': line_dict.get('file', 'unknown'),
                'file_order': line_dict.get('file_order', 0),
                'line_num': line_dict.get('line_num', 0),
                'sequence': line_dict.get('sequence', 0)
            })
    return rows

def _get_temp_dir(suffix: str) -> Optional[str]:
    temp_dir = f"{TEMP_DIR_BASE}_{suffix}_{os.getpid()}_{int(time.time())}"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir, ignore_errors=True)
    try:
        os.makedirs(temp_dir)
        return temp_dir
    except OSError as e:
        logger.error(f"Cannot create temp dir {temp_dir}: {e}")
        return None

def _cleanup_temp_dir(temp_dir: Optional[str]):
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
        logger.debug(f"Removed temp dir: {temp_dir}")

def _save_temp_parquet(df: pd.DataFrame, temp_file: str) -> bool:
    cdef object table # Use object for Arrow Table
    try:
        # Use object for table as well
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, temp_file, compression='snappy')
        return True
    except Exception as e: logger.error(f"Failed to save {temp_file}: {e}"); return False


def _combine_temp_parquets(temp_files: List[str], dtypes: Dict[str, Any]) -> pd.DataFrame:
    if not temp_files: 
        return pd.DataFrame(columns=dtypes.keys()).astype(dtypes)
    
    logger.info(f"Combining {len(temp_files)} temporary Parquet files...")
    cdef list final_cols = list(dtypes.keys())
    cdef str temp_dir = None
    cdef int batch_size = 5  # Process 5 files at a time - smaller batch for less memory use
    cdef int total_batches = (len(temp_files) + batch_size - 1) // batch_size
    cdef object final_df = pd.DataFrame(columns=final_cols).astype(dtypes)
    
    try:
        # Process files in batches to avoid memory overload
        for batch_num in range(1, total_batches + 1):
            start_idx = (batch_num - 1) * batch_size
            end_idx = min(start_idx + batch_size, len(temp_files))
            batch_files = temp_files[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_num}/{total_batches} with {len(batch_files)} files")
            
            # Process this batch into a dataframe directly
            batch_dfs = []
            for file_path in batch_files:
                try:
                    # Read just a single file into pandas
                    file_df = pq.read_table(file_path, columns=final_cols).to_pandas()
                    batch_dfs.append(file_df)
                    del file_df
                except Exception as e:
                    logger.warning(f"Error reading file {file_path}: {e}")
            
            # Combine this batch if we have any dataframes
            if batch_dfs:
                # Concatenate dataframes in this batch
                batch_df = pd.concat(batch_dfs, ignore_index=True)
                
                # Append to the final dataframe
                final_df = pd.concat([final_df, batch_df], ignore_index=True)
                
                # Clean up
                del batch_df
                del batch_dfs
                gc.collect()
                
            logger.info(f"Progress: {batch_num}/{total_batches} batches processed")
        
        # Apply final dtypes at the end
        if not final_df.empty:
            logger.info("Applying final dtypes...")
            final_df = final_df.astype(dtypes, errors='ignore')
            
            # Special handling for categorical columns
            if 'type' in final_df.columns and final_df['type'].dtype == 'object':
                final_df.loc[:, 'type'] = final_df['type'].astype('category')
                
            logger.info(f"Combined {len(final_df)} rows. RAM: {final_df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
        
        return final_df
            
    except Exception as e:
        logger.error(f"Error combining temp files: {e}")
        return pd.DataFrame(columns=final_cols).astype(dtypes)
        
    except Exception as e:
        logger.error(f"Error combining temp files with Arrow Dataset: {e}")
        return pd.DataFrame(columns=final_cols).astype(dtypes)


@cython.boundscheck(False)
@cython.wraparound(False)
def parse_content_lines_sequential(line_dicts: List[Dict[str, Any]], batch_size: cython.int) -> pd.DataFrame:
    cdef:
        cython.int total_lines = len(line_dicts)
        list temp_files = []
        # Use 'object' for DataFrame references in cdef
        object temp_df = None # Initialize to None
        object combined_df = None # Initialize to None
        str temp_dir = None
        str temp_file = ""
        list batch = []
        list rows = []
        cython.int i = 0 # Initialize loop variable
        cython.int batch_num = 0 # Initialize
        cython.int batch_count = 0 # Initialize

    if total_lines == 0: return pd.DataFrame(columns=DTYPES_CONTENT.keys()).astype(DTYPES_CONTENT)
    logger.info(f"Starting sequential parsing ({total_lines} lines, batch size {batch_size}).")

    temp_dir = _get_temp_dir("seq")
    if not temp_dir: return pd.DataFrame(columns=DTYPES_CONTENT.keys()).astype(DTYPES_CONTENT)

    try:
        batch_count = (total_lines + batch_size - 1) // batch_size
        for i in range(0, total_lines, batch_size):
            batch_num = (i // batch_size) + 1
            logger.info(f"Processing sequential batch {batch_num}/{batch_count}")
            batch = line_dicts[i : min(i + batch_size, total_lines)]
            rows = process_content_batch(batch)
            del batch; gc.collect()
            if rows:
                try:
                    # Assign Python object to temp_df
                    temp_df = pd.DataFrame(rows).astype(DTYPES_CONTENT, errors='ignore')
                    temp_file = os.path.join(temp_dir, f"batch_{batch_num}.parquet")
                    if _save_temp_parquet(temp_df, temp_file): temp_files.append(temp_file)
                except Exception as e: logger.error(f"Error saving sequential batch {batch_num}: {e}")
                finally:
                    # 'del temp_df' works correctly on the object reference
                    if temp_df is not None: del temp_df; temp_df = None
                    if rows: del rows; rows = [] # Clear list too
                    gc.collect()

        combined_df = _combine_temp_parquets(temp_files, DTYPES_CONTENT)
    except Exception as e:
        logger.error(f"Error during sequential processing loop: {e}")
        combined_df = pd.DataFrame(columns=DTYPES_CONTENT.keys()).astype(DTYPES_CONTENT)
    finally:
        _cleanup_temp_dir(temp_dir)
    # Return the Python object
    return combined_df

@cython.boundscheck(False)
@cython.wraparound(False)
def parse_content_lines_parallel(line_dicts: List[Dict[str, Any]], batch_size: cython.int, num_workers: cython.int) -> pd.DataFrame:
    cdef:
        cython.int total_lines = len(line_dicts)
        double available_memory_gb_start = get_available_memory()
        double estimated_mem_needed_gb
        list batches = []
        list all_rows = []
        str temp_dir = None
        # Use object for DataFrame references
        object df = None
        object sequential_df = None
        object partial_df = None
        object final_df = None
        bint parallel_completed_successfully = False
        cython.int chunk_size_batches = 0 # Initialize
        cython.int num_batch_chunks = 0 # Initialize
        double mem_before_chunk = 0.0 # Initialize
        double mem_after_chunk = 0.0 # Initialize
        double low_mem_threshold = 0.0 # Initialize
        cython.int batch_chunk_idx = 0 # Initialize
        cython.int start_idx = 0 # Initialize
        cython.int end_idx = 0 # Initialize
        list batch_chunk = []
        object pool # multiprocessing.Pool object
        object results_iterator # Iterator object
        list batch_rows = []
        str temp_fallback_file = None
        cython.longlong first_remaining_line_idx = 0 # Initialize
        list remaining_line_dicts = []
        cython.Py_ssize_t imap_chunksize = 1 # Initialize with a default

    if total_lines == 0: return pd.DataFrame(columns=DTYPES_CONTENT.keys()).astype(DTYPES_CONTENT)

    estimated_mem_needed_gb = (total_lines * 15) / (1024**2)
    if estimated_mem_needed_gb > available_memory_gb_start * 0.7:
        logger.warning(f"High estimated memory need. Switching to sequential parsing.")
        return parse_content_lines_sequential(line_dicts, batch_size)

    logger.info(f"Starting parallel parsing ({total_lines} lines, Batch={batch_size}, Workers={num_workers})")
    temp_dir = _get_temp_dir("par")

    try:
        batches = [line_dicts[i : min(i + batch_size, total_lines)] for i in range(0, total_lines, batch_size)]
        logger.info(f"Created {len(batches)} batches.")
    except MemoryError:
        logger.error("MemoryError creating batches. Falling back to sequential.")
        gc.collect()
        _cleanup_temp_dir(temp_dir)
        return parse_content_lines_sequential(line_dicts, batch_size)

    chunk_size_batches = max(1, min(BATCH_PROCESSING_CHUNK_SIZE, (len(batches) + num_workers - 1) // num_workers))
    num_batch_chunks = (len(batches) + chunk_size_batches - 1) // chunk_size_batches
    logger.info(f"Processing in {num_batch_chunks} batch chunks...")

    try:
        mem_before_chunk = get_available_memory()
        for batch_chunk_idx in range(num_batch_chunks):
            start_idx = batch_chunk_idx * chunk_size_batches
            end_idx = min(start_idx + chunk_size_batches, len(batches))
            batch_chunk = batches[start_idx:end_idx]
            if not batch_chunk: continue

            logger.info(f"Processing batch chunk {batch_chunk_idx + 1}/{num_batch_chunks}...")
            try:
                with multiprocessing.Pool(processes=num_workers, maxtasksperchild=30) as pool:
                    # Calculate imap_chunksize inside the loop where variables are defined
                    imap_chunksize = max(1, len(batch_chunk) // (num_workers * 2))
                    results_iterator = pool.imap_unordered(process_content_batch, batch_chunk, chunksize=imap_chunksize)
                    for batch_rows in results_iterator:
                         if batch_rows: all_rows.extend(batch_rows)
            finally:
                 # Ensure cleanup even if inner try fails
                 if 'batch_chunk' in locals(): del batch_chunk
                 if 'results_iterator' in locals(): del results_iterator
                 gc.collect()

            mem_after_chunk = get_available_memory()
            logger.info(f"Finished chunk {batch_chunk_idx + 1}. Rows: {len(all_rows)}. Available RAM: {mem_after_chunk:.1f} GB")

            low_mem_threshold = max(0.8, available_memory_gb_start * 0.15)
            if mem_after_chunk < low_mem_threshold:
                logger.warning(f"Low memory detected ({mem_after_chunk:.1f} GB). Falling back to sequential for remaining.")
                temp_fallback_file = None
                if all_rows and temp_dir:
                    try:
                        # Use object for partial_df
                        partial_df = pd.DataFrame(all_rows).astype(DTYPES_CONTENT, errors='ignore')
                        temp_fallback_file = os.path.join(temp_dir, "fallback_partial.parquet")
                        if _save_temp_parquet(partial_df, temp_fallback_file): logger.info("Saved partial results.")
                        else: temp_fallback_file = None
                        # 'del partial_df' works on the object
                        del partial_df; partial_df = None
                        all_rows = []; gc.collect()
                    except Exception as e: logger.error(f"Failed saving partial results: {e}"); temp_fallback_file = None

                first_remaining_line_idx = <cython.longlong>end_idx * batch_size
                remaining_line_dicts = line_dicts[first_remaining_line_idx:] if first_remaining_line_idx < total_lines else []
                # Use object for sequential_df
                sequential_df = parse_content_lines_sequential(remaining_line_dicts, batch_size)
                del remaining_line_dicts; gc.collect()

                if temp_fallback_file:
                    # Use object for partial_df
                    partial_df = _combine_temp_parquets([temp_fallback_file], DTYPES_CONTENT)
                    # Use object for final_df
                    final_df = pd.concat([partial_df, sequential_df], ignore_index=True).astype(DTYPES_CONTENT, errors='ignore')
                    # 'del partial_df' works
                    del partial_df; partial_df = None
                else: final_df = sequential_df

                logger.info(f"Low memory fallback complete. Final rows: {len(final_df)}")
                parallel_completed_successfully = False
                del line_dicts, batches; gc.collect()
                _cleanup_temp_dir(temp_dir)
                return final_df

            mem_before_chunk = mem_after_chunk

        parallel_completed_successfully = True

    except Exception as e:
        logger.error(f"Fatal error during parallel processing: {e}", exc_info=True)
        logger.info("Attempting full sequential fallback...")
        parallel_completed_successfully = False
        try:
            # Use object for final_df
            final_df = parse_content_lines_sequential(line_dicts, batch_size)
        except Exception as fallback_err:
            logger.error(f"Sequential fallback also failed: {fallback_err}")
            final_df = pd.DataFrame(columns=DTYPES_CONTENT.keys()).astype(DTYPES_CONTENT)
        # Cleanup lists and objects that might hold significant memory
        if 'line_dicts' in locals(): del line_dicts
        if 'batches' in locals(): del batches
        if 'all_rows' in locals(): del all_rows
        gc.collect()
        _cleanup_temp_dir(temp_dir)
        return final_df

    # ---- If parallel completed fully ----
    if parallel_completed_successfully:
        logger.info("Parallel processing successful. Cleaning up.")
        # Cleanup lists and objects that might hold significant memory
        if 'line_dicts' in locals(): del line_dicts
        if 'batches' in locals(): del batches
        gc.collect()
        _cleanup_temp_dir(temp_dir)

        if not all_rows: return pd.DataFrame(columns=DTYPES_CONTENT.keys()).astype(DTYPES_CONTENT)
        logger.info(f"Creating final DataFrame from {len(all_rows)} rows...")
        try:
            # Use object for df
            df = pd.DataFrame(all_rows).astype(DTYPES_CONTENT, errors='ignore')
            del all_rows # Free list memory after df creation
            gc.collect()
            logger.info(f"Final DF RAM: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
            return df
        except Exception as e:
            logger.error(f"Error creating final DataFrame: {e}")
            if 'all_rows' in locals(): del all_rows # Ensure cleanup on error too
            gc.collect()
            return pd.DataFrame(columns=DTYPES_CONTENT.keys()).astype(DTYPES_CONTENT)
    else:
        logger.error("Reached end of parallel function unexpectedly.")
        _cleanup_temp_dir(temp_dir)
        return pd.DataFrame(columns=DTYPES_CONTENT.keys()).astype(DTYPES_CONTENT)


# --- Merging & Formatting ---

@cython.boundscheck(False)
@cython.wraparound(False)
def merge_data(content_df: pd.DataFrame, sources_df: pd.DataFrame) -> pd.DataFrame:
    if content_df.empty or sources_df.empty: return pd.DataFrame(columns=DTYPES_MERGED.keys()).astype(DTYPES_MERGED)
    logger.info(f"Merging {len(content_df)} content rows with {len(sources_df)} sources.")
    # Use object for DataFrame reference
    cdef object merged_df = None
    try:
        # Use .loc for modifying columns to avoid ChainedAssignmentWarning
        if 'id' in content_df.columns and content_df['id'].dtype != 'object':
             content_df.loc[:, 'id'] = content_df['id'].astype(str)
        if 'id' in sources_df.columns and sources_df['id'].dtype != 'object':
             sources_df.loc[:, 'id'] = sources_df['id'].astype(str)

        merged_df = pd.merge(content_df, sources_df, on='id', how='inner')
        logger.info(f"Merge complete ({len(merged_df)} rows). Sorting by sequence...")
        if 'sequence' in merged_df.columns:
            merged_df = merged_df.sort_values('sequence', ignore_index=True)

        # First apply general dtypes
        merged_df = merged_df.astype(DTYPES_MERGED, errors='ignore')
        # Then use .loc for specific column modifications if needed (e.g., category)
        if 'type' in merged_df.columns and merged_df['type'].dtype == 'object':
             merged_df.loc[:, 'type'] = merged_df['type'].astype('category')
        logger.info(f"Final Merged DF RAM: {merged_df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
        return merged_df
    except Exception as e:
        logger.error(f"Error during merge/sort: {e}")
        return pd.DataFrame(columns=DTYPES_MERGED.keys()).astype(DTYPES_MERGED)

@cython.boundscheck(False)
@cython.wraparound(False)
def format_for_validation(merged_df: pd.DataFrame) -> List[Dict[str, Any]]:
    if merged_df.empty: return []
    cdef list required_cols = ['content', 'source', 'country', 'title', 'id']
    if not all(col in merged_df.columns for col in required_cols):
        logger.error(f"Missing columns for formatting: {set(required_cols) - set(merged_df.columns)}")
        return []

    logger.info(f"Formatting {len(merged_df)} rows for validation structure...")
    cdef int chunk_size = 50000  # Process in smaller chunks
    cdef int total_chunks = (len(merged_df) + chunk_size - 1) // chunk_size
    cdef list validation_format = []
    cdef list chunk_format = []
    cdef object chunk_df = None
    cdef int i, start_idx, end_idx
    cdef object row  # Use object for itertuples result
    cdef str content_val, source_val, country_val, title_val, gpt_response
    cdef list conversation

    try:
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, len(merged_df))
            logger.info(f"Formatting chunk {i+1}/{total_chunks} (rows {start_idx}-{end_idx})")
            
            # Get chunk of the dataframe
            chunk_df = merged_df.iloc[start_idx:end_idx]
            chunk_format = []
            
            # Process each row in the chunk
            for row in chunk_df.itertuples(index=False):
                content_val = str(getattr(row, 'content', ''))
                
                # Start building gpt_response
                gpt_response = "The sample you provided "
                
                # Add country info if available
                if hasattr(row, 'country') and getattr(row, 'country') and str(getattr(row, 'country')) != 'nan':
                    country_val = str(getattr(row, 'country'))
                    gpt_response += f"appears to come from {country_val}"
                
                # Add title info if available
                if hasattr(row, 'title') and getattr(row, 'title') and str(getattr(row, 'title')).strip() != 'nan':
                    title_val = str(getattr(row, 'title')).strip()
                    if "come from" in gpt_response:
                        gpt_response += f" and has the title '{title_val}'"
                    else:
                        gpt_response += f"has the title '{title_val}'"
                
                # Add source info if available
                source_val = str(getattr(row, 'source', 'unknown_source'))
                
                # Finalize the response with a period if needed
                if not gpt_response.endswith('.'):
                    gpt_response += "."
                
                # Create conversation structure
                conversation = [
                    {"from": "human", "value": "Given the following example text, identify the dialect of the speaker: " + content_val},
                    {"from": "gpt", "value": gpt_response}
                ]
                
                # Add to chunk results
                chunk_format.append({"conversations": conversation, "source": source_val, "score": 0})
            
            # Append chunk results to main results and free memory
            validation_format.extend(chunk_format)
            del chunk_df
            del chunk_format
            gc.collect()
            
            logger.info(f"Processed {len(validation_format)} records so far")
            
        logger.info(f"Formatted {len(validation_format)} records total.")
    except Exception as e:
        logger.error(f"Error during validation formatting: {e}", exc_info=True)
    
    return validation_format

# --- Main Pipeline ---

@cython.boundscheck(False)
@cython.wraparound(False)
def process_data(input_folder: str, sources_file: str, num_workers: cython.int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    logger.info("="*20 + " Starting Data Processing Pipeline " + "="*20)
    start_time = time.time()
    cdef list train_formatted = [], eval_formatted = []
    # Use object for DataFrame references in cdef
    cdef object sources_df = None
    cdef object content_df = None
    cdef object merged_df = None
    cdef object train_df = None
    cdef object eval_df = None
    cdef list all_line_dicts = []
    cdef cython.int batch_size = 0 # Initialize
    cdef cython.int actual_workers = 0 # Initialize
    cdef cython.int split_idx = 0 # Initialize
    cdef double duration = 0.0 # Initialize

    try:
        sources_df = parse_sources_file(sources_file)
        if sources_df.empty: raise ValueError("Failed to load sources.")
        gc.collect()

        all_line_dicts = collect_content_lines_parallel(input_folder, num_workers)
        if not all_line_dicts: raise ValueError("Failed to collect content lines.")
        gc.collect()

        batch_size, actual_workers = calculate_processing_parameters(len(all_line_dicts), "lines")
        if num_workers > 0: actual_workers = max(1, min(num_workers, actual_workers))
        content_df = parse_content_lines_parallel(all_line_dicts, batch_size, actual_workers)
        # 'del' works on list object
        del all_line_dicts; all_line_dicts = [] # Ensure list is cleared
        gc.collect()
        if content_df.empty: raise ValueError("Failed to parse content lines.")
        gc.collect()

        merged_df = merge_data(content_df, sources_df)
        # 'del' works on object references
        del content_df; content_df = None
        del sources_df; sources_df = None
        gc.collect()
        if merged_df.empty: raise ValueError("Merging failed or resulted in empty DataFrame.")
        gc.collect()

        logger.info(f"Shuffling {len(merged_df)} rows and splitting ({TRAIN_SPLIT_RATIO*100:.0f}% train)...")
        # Assignment works correctly with object types
        merged_df = merged_df.iloc[np.random.RandomState(seed=RANDOM_SEED).permutation(len(merged_df))].reset_index(drop=True)
        split_idx = int(len(merged_df) * TRAIN_SPLIT_RATIO)
        train_df = merged_df.iloc[:split_idx]
        eval_df = merged_df.iloc[split_idx:]
        logger.info(f"Split complete: Train={len(train_df)}, Eval={len(eval_df)}")
        # 'del' works on object reference
        del merged_df; merged_df = None
        gc.collect()

        logger.info("Formatting train data...")
        train_formatted = format_for_validation(train_df)
        # 'del' works on object reference
        del train_df; train_df = None
        gc.collect()
        logger.info("Formatting eval data...")
        eval_formatted = format_for_validation(eval_df)
        # 'del' works on object reference
        del eval_df; eval_df = None
        gc.collect()

        if not train_formatted and not eval_formatted:
            logger.warning("Formatting resulted in no data for train or eval.")

    except Exception as e:
        logger.critical(f"Pipeline failed: {e}", exc_info=True)
        # Cleanup on error - check if variables exist before deleting
        if 'sources_df' in locals() and sources_df is not None: del sources_df
        if 'content_df' in locals() and content_df is not None: del content_df
        if 'merged_df' in locals() and merged_df is not None: del merged_df
        if 'train_df' in locals() and train_df is not None: del train_df
        if 'eval_df' in locals() and eval_df is not None: del eval_df
        if 'all_line_dicts' in locals() and all_line_dicts is not None: del all_line_dicts
        gc.collect()
        return [], []
    finally:
        duration = time.time() - start_time
        logger.info(f"Pipeline finished in {duration:.2f} seconds.")
        logger.info("="*20 + " Pipeline Finished " + "="*20)

    return train_formatted, eval_formatted

# --- Exporting ---

def _export_single_parquet(data: List[Dict[str, Any]], output_file: str):
    cdef cython.int total_rows = len(data)
    if total_rows == 0: return
    logger.info(f"Exporting {total_rows} records to {output_file}...")

    cdef:
        # Use object for Schema, Writer, DataFrame, Table
        object schema = None
        object writer = None
        cython.int i = 0 # Initialize
        cython.int chunk_num = 0 # Initialize
        cython.int start_idx = 0 # Initialize
        cython.int end_idx = 0 # Initialize
        cython.int num_chunks = 0 # Initialize
        list chunk_data = []
        object df_chunk = None
        object table = None

    try:
        # Use smaller chunks for less memory usage
        chunk_size = min(EXPORT_CHUNK_SIZE, 10000)  # Smaller chunk size
        num_chunks = (total_rows + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk_data = data[start_idx:end_idx]
            if not chunk_data: continue
            chunk_num = i + 1
            logger.info(f"Exporting chunk {chunk_num}/{num_chunks} ({len(chunk_data)} records)")

            try:
                df_chunk = pd.DataFrame(chunk_data)
                table = pa.Table.from_pandas(df_chunk, preserve_index=False)
                # 'del' works on object
                del df_chunk; df_chunk = None
                del chunk_data; chunk_data = []  # Clear the chunk data immediately
                gc.collect()

                if writer is None:
                    schema = table.schema
                    logger.info(f"Inferred schema for {os.path.basename(output_file)}: {schema}")
                    writer = pq.ParquetWriter(output_file, schema, compression='snappy')

                if table.schema != schema:
                    logger.warning(f"Schema mismatch chunk {chunk_num}. Casting.")
                    table = table.cast(schema, safe=False)

                writer.write_table(table)
                del table; table = None
                gc.collect()
            except Exception as chunk_err:
                logger.error(f"Error writing chunk {chunk_num} to {os.path.basename(output_file)}: {chunk_err}")
            finally:
                # Clear any remaining objects
                if 'table' in locals() and table is not None: del table; table = None
                if 'df_chunk' in locals() and df_chunk is not None: del df_chunk; df_chunk = None
                if 'chunk_data' in locals() and chunk_data: del chunk_data; chunk_data = []
                gc.collect()

    except Exception as e:
        logger.error(f"Failed to export {output_file}: {e}", exc_info=True)
    finally:
        # Check if writer object exists before closing
        if writer is not None:
            try: writer.close()
            except Exception as close_err: logger.warning(f"Error closing writer: {close_err}")
        logger.info(f"Finished export attempt for {output_file}")


def export_datasets(train_data: List[Dict[str, Any]], eval_data: List[Dict[str, Any]], output_basename: str):
    logger.info("--- Exporting Datasets ---")
    if train_data: _export_single_parquet(train_data, f"{output_basename}_train.parquet")
    if eval_data: _export_single_parquet(eval_data, f"{output_basename}_eval.parquet")


# --- Main Execution ---

def _get_validated_path(env_var: str, default: str, check_dir: bool = False) -> str:
    path = os.path.abspath(os.getenv(env_var, default))
    exists = os.path.isdir(path) if check_dir else os.path.isfile(path)
    if not exists:
        file_type = "directory" if check_dir else "file"
        logger.warning(f"{file_type.capitalize()} specified by ${env_var} or default not found: {path}")
    return path

def _preview_output(output_file: str) -> None:
    """Previews first few rows using Pandas read_parquet and df.head()"""
    if not os.path.exists(output_file):
        logger.warning(f"Preview file not found: {output_file}")
        return
    try:
        logger.info(f"--- Output Preview ({os.path.basename(output_file)}, Top {PREVIEW_ROWS}): ---")
        preview_cols = ['source', 'conversations'] # Adjust columns as needed
   
        df_full = pd.read_parquet(output_file, columns=preview_cols)
        df_preview = df_full.head(PREVIEW_ROWS)

        with pd.option_context('display.max_colwidth', 80, 'display.max_rows', PREVIEW_ROWS + 2, 'display.width', 120):
            logger.info("\n" + df_preview.to_string())
    except Exception as e:
        logger.warning(f"Could not preview {output_file}: {e}")


def main():
    # Declare Cython variables at the top of the function scope
    cdef:
        double start_time = time.time()
        bint success = False
        int exit_code = 1
        str input_folder = ""
        str sources_txt = ""
        str output_base = ""
        int num_workers = 0
        list train_data = []
        list eval_data = []
        double duration = 0.0
        str num_workers_str = "" # For reading env var
        str item = "" # For loop in finally block

    configure_multiprocessing()

    logger.info("="*50)
    logger.info(f"Platform: {get_platform_info()}, RAM: {get_available_memory():.1f}GB, CPUs: {psutil.cpu_count(logical=True)}")
    logger.info("="*50)

    try:
        input_folder = _get_validated_path('INPUT_FOLDER', 'corpora', check_dir=True)
        sources_txt = _get_validated_path('SOURCES_FILE', 'sources.txt', check_dir=False)
        output_base = os.path.abspath(os.getenv('OUTPUT_BASENAME', 'output'))

        logger.info(f"Input Folder : {input_folder}")
        logger.info(f"Sources File : {sources_txt}")
        logger.info(f"Output Base  : {output_base}")

        if not os.path.isdir(input_folder): raise FileNotFoundError(f"Input folder not found: {input_folder}")
        if not os.path.isfile(sources_txt): raise FileNotFoundError(f"Sources file not found: {sources_txt}")

        # Assign Python variable value to Cython variable
        num_workers_str = os.getenv('NUM_WORKERS', '0')
        num_workers = int(num_workers_str) if num_workers_str and num_workers_str.isdigit() else 0 # Check not None/empty


        train_data, eval_data = process_data(input_folder, sources_txt, num_workers)

        if train_data or eval_data:
            export_datasets(train_data, eval_data, output_base)
            if train_data: _preview_output(f"{output_base}_train.parquet")
            success = True
            exit_code = 0
        else:
            logger.warning("Processing completed but yielded no data.")
            exit_code = 2

    except FileNotFoundError as e:
        logger.critical(f"Input path error: {e}")
        exit_code = 3
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        exit_code = 130
    except MemoryError:
        logger.critical("Critical MemoryError occurred. Aborting.")
        exit_code = 137
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {type(e).__name__} - {e}", exc_info=True)
        exit_code = 1
    finally:
        duration = time.time() - start_time
        status = "Success" if success else ("Failed" if exit_code != 2 else "Completed (No Data)")
        logger.info("="*50)
        logger.info(f"Run Status: {status}, Total Time: {duration:.2f}s")
        logger.info("="*50)
        # Final temp dir cleanup attempt
        # Use standard Python loop
        for item in os.listdir('.'):
             if item.startswith(TEMP_DIR_BASE) and os.path.isdir(item):
                 _cleanup_temp_dir(item)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())