# cython: language_level=3
# distutils: language = c

import re
import pandas as pd
import logging
from bs4 import BeautifulSoup  # Keep import, even if not directly used now, for potential future use in clean_text
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
# Cython imports (ensure these are available)
cimport cython
# Cython Standard Library Imports (no Python overhead for these)
from libc.stdlib cimport malloc, free
from libc.string cimport strlen, strcpy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Define constants
ENCODINGS: List[str] = ['utf-8', 'latin-1', 'cp1252']
# Memory optimization constants
MEMORY_SAFETY_FACTOR: float = 0.8  # Use up to 80% of *available* memory estimate
MIN_MEMORY_PER_WORKER_GB: float = 0.5 # Minimum estimated memory per worker process
DEFAULT_AVAILABLE_MEMORY_GB: float = 4.0 # Fallback if psutil fails
# Processing constants
MIN_BATCH_SIZE: cython.int = 100
MAX_BATCH_SIZE: cython.int = 5000 # Increased max batch size for potentially faster processing if memory allows
FILE_PROCESSING_CHUNK_SIZE: cython.int = 50 # Process N files at a time in parallel collection step
BATCH_PROCESSING_CHUNK_SIZE: cython.int = 50 # Process N batches at a time in parallel parsing step
EXPORT_CHUNK_SIZE: cython.int = 50000 # Rows per chunk for Parquet/Excel export

# --- Platform and Resource Detection ---

@cython.cfunc
@cython.inline
def get_platform_info() -> str:
    """Detect the platform and return information about it. (Cythonized for potential minor speedup if called often)"""
    cdef str system_str
    try:
        system_str = platform.system()
        if system_str == "Linux":
            # Check if running in WSL
            uname_release: str = platform.uname().release.lower()
            if "microsoft" in uname_release or "wsl" in uname_release:
                return "WSL"
            return "Linux"
        elif system_str == "Darwin":
            return "MacOS"
        elif system_str == "Windows":
            return "Windows"
    except Exception as e:
        logger.warning(f"Could not detect platform accurately: {e}")
    return "Unknown"

@cython.cfunc
def get_available_memory() -> cython.double:
    """Get available system memory in GB."""
    cdef:
        double available_gb, total_gb
        object mem_info # Use object for psutil result

    try:
        mem_info = psutil.virtual_memory()
        available_gb = mem_info.available / (1024.0 ** 3)
        total_gb = mem_info.total / (1024.0 ** 3)
        # Log less frequently or only if significantly changed? For now, keep it.
        # logger.debug(f"System memory: {total_gb:.1f} GB total, {available_gb:.1f} GB available")
        return max(0.1, available_gb) # Ensure returns at least a small positive value
    except Exception as e:
        logger.warning(f"Could not determine available memory via psutil: {e}. Using default: {DEFAULT_AVAILABLE_MEMORY_GB} GB")
        return DEFAULT_AVAILABLE_MEMORY_GB

def configure_multiprocessing() -> None:
    """Configure multiprocessing based on the platform and Python version."""
    cdef str platform_type = get_platform_info()

    # Set multiprocessing start method
    # 'spawn' is generally safer cross-platform, especially with complex imports or global state,
    # though it has higher overhead than 'fork'. 'fork' is Linux/macOS default but can be problematic.
    try:
        current_method = multiprocessing.get_start_method(allow_none=True)
        # Only set if not already set or if we need to force 'spawn'
        if platform_type in ["Windows", "WSL"]:
            if current_method != 'spawn':
                multiprocessing.set_start_method('fork', force=True)
                logger.info(f"Forcing multiprocessing start method to 'spawn' for {platform_type}")
            # freeze_support() is essential for Windows frozen executables
            multiprocessing.freeze_support()
        elif platform_type == "MacOS":
            # Newer Python versions on macOS default to 'spawn', which is safer.
            # Let's prefer 'spawn' unless explicitly on an old Python version.
            if sys.version_info >= (3, 8):
                 if current_method != 'spawn':
                    multiprocessing.set_start_method('spawn', force=True)
                    logger.info("Forcing multiprocessing start method to 'spawn' for MacOS (Python >= 3.8)")
            else: # Older Python on macOS might default to 'fork'
                 if current_method != 'fork':
                    multiprocessing.set_start_method('fork', force=True)
                    logger.info("Forcing multiprocessing start method to 'fork' for MacOS (Python < 3.8)")
        else: # Linux and others
             if current_method != 'fork':
                multiprocessing.set_start_method('fork', force=True)
                logger.info(f"Forcing multiprocessing start method to 'fork' for {platform_type}")
    except Exception as e:
        logger.error(f"Failed to configure multiprocessing start method: {e}. Using default.")
        if platform_type == "Windows":
             # Ensure freeze_support is called anyway for Windows bundling
             multiprocessing.freeze_support()

@cython.cdivision(True)
def calculate_processing_parameters(total_items: cython.int, item_type: str = "lines") -> Tuple[cython.int, cython.int]:
    """
    Calculate optimal batch size and worker count based on system resources and item count.

    Args:
        total_items: The total number of items (e.g., lines, files) to process.
        item_type: A string descriptor for logging (e.g., "lines", "files").

    Returns:
        A tuple containing (optimal_batch_size, optimal_workers).
    """
    cdef:
        double available_memory_gb = get_available_memory()
        str platform_type = get_platform_info()
        # Estimate memory per item (adjust based on expected item size)
        # Let's use a slightly higher estimate for safety margin
        cython.int estimated_memory_per_item_kb = 15 if item_type == "lines" else 1024 # 15KB/line, 1MB/file (rough estimates)
        double total_memory_needed_gb
        double max_memory_per_batch_gb
        cython.int optimal_batch_size, physical_cpu_count, logical_cpu_count
        cython.int base_workers, memory_based_workers, optimal_workers

    logger.info(f"Calculating parameters based on {available_memory_gb:.1f} GB RAM and {total_items} {item_type}")

    # Calculate memory requirements (rough estimate)
    total_memory_needed_gb = (total_items * estimated_memory_per_item_kb) / (1024.0 * 1024.0)
    logger.info(f"Estimated total memory needed for {item_type}: {total_memory_needed_gb:.2f} GB (very approximate)")

    # Determine optimal batch size based on available memory
    # Use a fraction of *available* memory per batch, leaving room for OS and other processes.
    # Let's target ~10-15% of available memory per batch.
    max_memory_per_batch_gb = available_memory_gb * 0.15

    # Calculate optimal batch size
    optimal_batch_size = MIN_BATCH_SIZE # Default to minimum
    if estimated_memory_per_item_kb > 0:
        calculated_batch_size = <cython.int>((max_memory_per_batch_gb * 1024.0 * 1024.0) / estimated_memory_per_item_kb)
        # Clamp between MIN_BATCH_SIZE and MAX_BATCH_SIZE
        optimal_batch_size = max(MIN_BATCH_SIZE, min(calculated_batch_size, MAX_BATCH_SIZE))

    # Determine optimal worker count
    try:
        physical_cpu_count = psutil.cpu_count(logical=False) or 1
        logical_cpu_count = psutil.cpu_count(logical=True) or 2
    except Exception:
        logger.warning("Could not get CPU counts via psutil. Using defaults (1 physical, 2 logical).")
        physical_cpu_count = 1
        logical_cpu_count = 2

    # Base worker count heuristic based on platform (more conservative)
    if platform_type == "WSL":
        base_workers = max(1, min(physical_cpu_count, logical_cpu_count // 2, 4)) # WSL often shares resources heavily
    elif platform_type == "Windows":
        base_workers = max(1, min(logical_cpu_count - 1, physical_cpu_count, 6)) # Windows 'spawn' overhead
    else: # Linux, MacOS
        base_workers = max(1, min(logical_cpu_count - 1, physical_cpu_count * 2)) # Can often utilize hyperthreading better

    # Adjust worker count based on available memory per worker
    memory_based_workers = max(1, <cython.int>((available_memory_gb * MEMORY_SAFETY_FACTOR) / MIN_MEMORY_PER_WORKER_GB))

    # Final worker count: minimum of CPU-based and memory-based, ensuring at least 1
    optimal_workers = max(1, min(base_workers, memory_based_workers))

    # Further reduce workers if total items are very few compared to workers
    optimal_workers = max(1, min(optimal_workers, (total_items + optimal_batch_size - 1) // optimal_batch_size )) # No more workers than batches
    optimal_workers = max(1, min(optimal_workers, total_items)) # No more workers than items

    logger.info(f"Optimal processing parameters: batch_size={optimal_batch_size}, workers={optimal_workers}")
    return optimal_batch_size, optimal_workers

# --- Text Cleaning ---

# Public function that users will call
@cython.boundscheck(False)
@cython.wraparound(False)
# @functools.lru_cache(maxsize=1024) # Caching might be useful if the same text appears often, but adds overhead. Consider if needed.
cpdef str clean_text(str text_obj):
    """
    Clean text by applying various text cleaning operations.
    This version focuses on basic cleaning. Add more rules as needed.

    Args:
        text_obj: The text to clean

    Returns:
        Cleaned text
    """
    cdef str cleaned_text

    if not isinstance(text_obj, str):
        # Handle potential non-string input gracefully
        logger.debug(f"clean_text received non-string input: {type(text_obj)}. Converting to string.")
        text_obj = str(text_obj)

    # Basic cleaning:
    # 1. Remove leading/trailing whitespace
    cleaned_text = text_obj.strip()

    # 2. Replace multiple spaces with a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    # 3. (Optional) Remove HTML tags if they might be present
    # If HTML is common, enable this. Requires BeautifulSoup.
    # try:
    #     # Use 'html.parser' for built-in, no extra dependency
    #     soup = BeautifulSoup(cleaned_text, 'html.parser')
    #     cleaned_text = soup.get_text()
    #     # Re-apply whitespace cleaning after tag removal
    #     cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    # except Exception as e:
    #     logger.warning(f"BeautifulSoup failed during cleaning (text: '{cleaned_text[:50]}...'): {e}")
    #     # Fallback: keep the text as is if parsing fails

    # Add more cleaning rules here if needed (e.g., removing special characters, normalization)

    return cleaned_text


# --- File Handling and Parsing ---

@cython.boundscheck(False)
@cython.wraparound(False)
def read_file_with_multiple_encodings(file_path: str, encodings: List[str] = None) -> Tuple[Optional[List[str]], str]:
    """Try to read a file with multiple encodings, line by line for memory efficiency."""
    cdef:
        str encoding
        list lines = []
        object f # File handle object
        str line

    if encodings is None:
        encodings = ENCODINGS

    for encoding in encodings:
        try:
            # Read line by line to avoid loading huge files entirely into memory
            lines = [] # Reset lines list for each encoding attempt
            with open(file_path, 'r', encoding=encoding) as f:
                for line in f:
                    lines.append(line)
            # If we successfully read the whole file
            logger.debug(f"Successfully read {file_path} with encoding {encoding}")
            return lines, encoding
        except UnicodeDecodeError:
            logger.debug(f"Failed to decode {file_path} with {encoding}")
            # If decoding fails partway, 'lines' might contain partial data.
            # We discard it by continuing to the next encoding.
            continue
        except FileNotFoundError:
             logger.error(f"File not found: {file_path}")
             return None, "" # File doesn't exist
        except IOError as e:
             logger.error(f"IOError reading file {file_path} with {encoding}: {e}")
             # Consider if retrying makes sense for some IOErrors
             return None, "" # Treat as unreadable
        except Exception as e:
            logger.error(f"Unexpected error reading file {file_path} with {encoding}: {e}")
            # Don't try other encodings if an unexpected error occurs
            return None, ""

    logger.warning(f"Could not decode {file_path} with any specified encoding: {encodings}")
    return None, ""

@cython.boundscheck(False)
@cython.wraparound(False)
def parse_sources_file(sources_file: str) -> pd.DataFrame:
    """Parse the sources file into a DataFrame."""
    cdef:
        list lines = None
        str line, encoding
        list parts
        dict record
        list sources_data = []
        object df # Use object for Pandas DataFrame type

    lines, encoding = read_file_with_multiple_encodings(sources_file)
    if lines is None: # Changed check from 'not lines' to 'is None' for clarity
        logger.error(f"Could not read or decode sources file: {sources_file}")
        return pd.DataFrame() # Return empty DataFrame

    logger.info(f"Parsing sources file {sources_file} (detected encoding: {encoding})")
    for line in lines:
        line = line.strip()
        if not line: # Skip empty lines
            continue
        parts = line.split(maxsplit=5) # Split efficiently, max 5 splits needed
        if len(parts) == 6:
            # Basic validation (e.g., check if year is numeric) can be added here
            record = {
                'id': parts[0],
                'year': parts[1],
                'type': parts[2],
                'pages': parts[3],
                'source': parts[4],
                'title': parts[5] # The rest is title
            }
            sources_data.append(record)
        else:
            logger.warning(f"Skipping malformed line in sources file: '{line[:100]}...' (found {len(parts)} parts)")

    if not sources_data:
        logger.warning("No valid data found in sources file")
        return pd.DataFrame()

    try:
        df = pd.DataFrame(sources_data)
        # Optimize DataFrame memory usage
        df.loc['id'] = df['id'].astype(str) # Ensure 'id' is string for merging
        # Consider converting 'year', 'pages' to numeric if appropriate
        # df['year'] = pd.to_numeric(df['year'], errors='coerce')
        # df['pages'] = pd.to_numeric(df['pages'], errors='coerce')
        # Convert low-cardinality columns to 'category'
        df.loc['type'] = df['type'].astype('category')
        logger.info(f"Loaded {len(df)} source records into DataFrame")
        return df
    except Exception as e:
        logger.error(f"Error creating DataFrame from sources data: {e}")
        return pd.DataFrame()


# --- Parallel Content Collection ---

# Worker function for process_file
@cython.boundscheck(False)
@cython.wraparound(False)
def process_single_file(file_info_tuple: Tuple[str, str, int, int]) -> Tuple[bool, List[Dict[str, Any]], str]:
    """
    Processes a single file: reads, finds '@@' lines, assigns sequence numbers.

    Args:
        file_info_tuple: Tuple containing (root_dir, filename, file_order, start_sequence).

    Returns:
        Tuple: (success_flag, list_of_line_dicts, file_path)
               line_dict keys: 'line', 'file', 'file_order', 'line_num', 'sequence'
    """
    cdef:
        str root_dir = file_info_tuple[0]
        str filename = file_info_tuple[1]
        cython.int file_order = file_info_tuple[2]
        cython.int start_sequence = file_info_tuple[3]
        str file_path
        list content = None
        str encoding
        list result_lines = []
        cython.int line_num
        str line
        cython.int sequence_counter = 0

    file_path = os.path.join(root_dir, filename)
    # logger.debug(f"Worker {os.getpid()} starting processing: {file_path}") # Debug logging

    try:
        content, encoding = read_file_with_multiple_encodings(file_path)

        if content is not None:
            for line_num, line in enumerate(content):
                # Strip whitespace efficiently
                line_stripped = line.strip()
                if line_stripped.startswith('@@'):
                    sequence_counter += 1
                    # Store only necessary info, raw line included
                    result_lines.append({
                        'line': line_stripped, # Store stripped line
                        'file_order': file_order,
                        'line_num': line_num,
                        'sequence': start_sequence + sequence_counter
                        # File path added later in the collector to reduce IPC data size
                    })

            # logger.debug(f"Worker {os.getpid()} finished {file_path}. Found {len(result_lines)} lines.")
            # Return success, results, and file_path for logging in the main process
            return True, result_lines, file_path
        else:
            logger.warning(f"Could not read or decode file {file_path}, skipping.")
            return False, [], file_path # Return failure, empty list, and file_path
    except MemoryError:
         logger.error(f"MemoryError processing file {file_path} in worker {os.getpid()}. Skipping.")
         # Explicitly trigger garbage collection in the worker before exiting
         gc.collect()
         return False, [], file_path
    except Exception as e:
        logger.error(f"Unexpected error processing file {file_path} in worker {os.getpid()}: {str(e)}")
        return False, [], file_path
    # finally:
        # Ensure memory is released (though Python GC should handle 'content')
        # del content # Explicit deletion
        # gc.collect() # Force collection in worker (use cautiously, can add overhead)

@cython.boundscheck(False)
@cython.wraparound(False)
def collect_content_lines_parallel(input_folder: str, num_workers: cython.int = 0) -> List[Dict[str, Any]]:
    """
    Walk through directories, collect '@@' lines using parallel processing with chunking.

    Args:
        input_folder: Path to the root directory containing .txt files.
        num_workers: Number of worker processes. If 0, calculated automatically.

    Returns:
        A list of dictionaries, each representing a found line and its metadata.
        Sorted by 'sequence'.
    """
    cdef:
        list file_infos = []
        cython.int file_order = 0
        cython.int total_files = 0
        str root, filename
        list dirs, files, txt_files # For os.walk results
        list all_lines = []
        cython.int processed_files = 0
        cython.int skipped_files = 0
        cython.int actual_workers
        cython.int chunk_size_files
        cython.int chunk_idx, num_file_chunks
        list file_chunk
        object pool # multiprocessing.Pool object
        list results
        bint success
        list lines
        str file_path_processed # Renamed from file_path to avoid clash
        dict line_dict


    logger.info(f"Starting parallel collection from folder: {input_folder}")

    # 1. Gather all file information first
    logger.info("Scanning directories for .txt files...")
    for root, dirs, files in os.walk(input_folder):
        # Sort files within directory for deterministic order (optional but good practice)
        txt_files = sorted([f for f in files if f.lower().endswith('.txt')])
        if txt_files:
            total_files += len(txt_files)
            for filename in txt_files:
                # Tuple: (root_dir, filename, file_order, starting_sequence_for_this_file)
                # Assign a large sequence block per file to ensure sorting works
                file_infos.append((root, filename, file_order, file_order * 1000000)) # Increased sequence gap
                file_order += 1

    if total_files == 0:
        logger.warning(f"No .txt files found in {input_folder}. Returning empty list.")
        return []

    logger.info(f"Found {total_files} .txt files to process.")

    # 2. Determine processing parameters (workers, file chunk size for pool submission)
    # Calculate based on number of *files* here
    _, actual_workers = calculate_processing_parameters(total_files, "files")
    if num_workers > 0: # Allow user override
        actual_workers = max(1, min(num_workers, actual_workers)) # Respect user choice but cap by calculation
    logger.info(f"Using {actual_workers} worker processes for file collection.")

    # Determine chunk size for submitting files to the pool
    chunk_size_files = max(1, min(FILE_PROCESSING_CHUNK_SIZE, (total_files + actual_workers - 1) // actual_workers))
    num_file_chunks = (len(file_infos) + chunk_size_files - 1) // chunk_size_files
    logger.info(f"Processing files in {num_file_chunks} chunks of up to {chunk_size_files} files each.")


    # 3. Process file chunks in parallel
    for chunk_idx in range(num_file_chunks):
        start_idx = chunk_idx * chunk_size_files
        end_idx = min(start_idx + chunk_size_files, total_files)
        file_chunk = file_infos[start_idx:end_idx]

        if not file_chunk: continue # Should not happen, but safety check

        logger.info(f"Processing file chunk {chunk_idx + 1}/{num_file_chunks} ({len(file_chunk)} files) using {actual_workers} workers...")

        try:
            # Create pool inside the loop to potentially help with memory leaks in long runs
            # Use maxtasksperchild to automatically restart workers after N tasks
            with multiprocessing.Pool(processes=actual_workers, maxtasksperchild=10) as pool:
                 # Use map_async for potentially better overlap, or imap_unordered for memory efficiency if order doesn't matter until final sort
                 # Let's stick with map for simplicity now, but imap_unordered is a good candidate if memory is tight
                results = pool.map(process_single_file, file_chunk, chunksize=1) # Small chunksize for map

            # Process results from the chunk
            for success, lines, file_path_processed in results:
                if success:
                    # Add file path to each line dictionary now
                    for line_dict in lines:
                        line_dict['file'] = file_path_processed
                    all_lines.extend(lines)
                    processed_files += 1
                else:
                    skipped_files += 1
                    # Logging for skipped file already happened in worker or process_single_file

            logger.info(f"Finished file chunk {chunk_idx + 1}. Processed: {processed_files}, Skipped: {skipped_files}, Lines collected so far: {len(all_lines)}")

        except (MemoryError, OSError, Exception) as e: # Catch broader errors during pool operation
            logger.error(f"Error during parallel processing of file chunk {chunk_idx + 1}: {type(e).__name__} - {e}")
            logger.warning("Attempting to continue with the next chunk (if any). Some files may be skipped.")
            # Count files in the failed chunk as skipped
            skipped_files += len(file_chunk)
            # Fallback: Could implement sequential processing for the failed chunk here if needed
            # for file_info in file_chunk:
            #     success, lines, file_path_processed = process_single_file(file_info)
            #     # ... (process results as above) ...

        finally:
            # Explicitly trigger garbage collection after each chunk processing
            # Helps release memory held by results list and potentially workers
            logger.debug("Triggering garbage collection after file chunk processing.")
            # Clear intermediate results list
            del results
            gc.collect()


    # 4. Final Sort and Summary
    if not all_lines:
        logger.warning("No '@@' lines collected from any file.")
        return []

    logger.info(f"Sorting {len(all_lines)} collected lines by sequence...")
    try:
        # Sort in-place to save memory
        all_lines.sort(key=lambda x: x['sequence'])
        logger.info("Sorting complete.")
    except MemoryError:
        logger.error("MemoryError during final sorting of collected lines. Data might be incomplete or unsorted.")
        # Cannot proceed reliably if sorting fails due to memory.
        # Could attempt to save unsorted data here if needed.
        return [] # Return empty or handle error appropriately

    logger.info(f"--- File Collection Summary ---")
    logger.info(f"- Total .txt files found: {total_files}")
    logger.info(f"- Files processed successfully: {processed_files}")
    logger.info(f"- Files skipped (read error, decode error, processing error): {skipped_files}")
    logger.info(f"- Total '@@' lines collected: {len(all_lines)}")
    logger.info(f"-------------------------------")

    return all_lines


# --- Content Parsing (from Collected Lines) ---

@cython.boundscheck(False)
@cython.wraparound(False)
def process_content_batch(batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Processes a batch of line dictionaries: extracts ID/content, cleans content.

    Args:
        batch_data: A list of line dictionaries from the collection step.

    Returns:
        A list of dictionaries ready for DataFrame creation.
        Keys: 'id', 'content', 'file', 'file_order', 'line_num', 'sequence'
    """
    cdef:
        list rows = []
        dict line_dict
        str line, content_raw, content_cleaned
        list parts
        str line_id # Use specific name

    # logger.debug(f"Worker {os.getpid()} processing batch of {len(batch_data)} lines.") # Debug logging
    for line_dict in batch_data:
        line = line_dict['line'] # Already stripped in collection phase
        # Efficiently split '@@id content'
        if line.startswith('@@') and ' ' in line:
             # Find the first space after '@@'
             space_index = line.find(' ', 2)
             if space_index != -1:
                 line_id = line[2:space_index]
                 content_raw = line[space_index+1:]
                 content_cleaned = clean_text(content_raw) # Apply cleaning

                 rows.append({
                     'id': line_id,
                     'content': content_cleaned,
                     'file': line_dict.get('file', 'unknown'), # Use .get for safety
                     'file_order': line_dict['file_order'],
                     'line_num': line_dict['line_num'],
                     'sequence': line_dict['sequence']
                 })
             else: # Line is like "@@id" with no space/content
                  logger.debug(f"Skipping line with '@@' but no content: {line[:100]}...")
        else: # Should not happen if collected correctly, but safety check
             logger.warning(f"Skipping unexpected line format in batch: {line[:100]}...")

    # logger.debug(f"Worker {os.getpid()} finished batch. Produced {len(rows)} rows.")
    # Explicitly delete input batch data in worker before returning
    # del batch_data # This might help worker memory, test impact
    # gc.collect() # Use cautiously
    return rows


def _save_temp_dataframe(df: pd.DataFrame, temp_file_path: str) -> bool:
    """Helper to save a dataframe chunk, returns success status."""
    try:
        # Use Parquet for intermediate files - often faster and more space efficient than CSV
        df.to_parquet(temp_file_path, index=False, engine='pyarrow', compression='snappy')
        # df.to_csv(temp_file_path, index=False) # CSV Alternative
        logger.debug(f"Saved temporary batch to {temp_file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save temporary batch to {temp_file_path}: {e}")
        return False

def _combine_temp_files(temp_files: List[str], final_columns: List[str], dtypes: Dict[str, Any]) -> pd.DataFrame:
    """Helper to combine temporary Parquet/CSV files into a single DataFrame."""
    cdef:
        list dfs_to_concat = []
        object df_chunk, combined_df
        str temp_file
        bint success = True

    logger.info(f"Combining {len(temp_files)} temporary batch files...")
    try:
        for temp_file in temp_files:
            try:
                # Read Parquet intermediate file
                df_chunk = pd.read_parquet(temp_file, columns=final_columns)
                # df_chunk = pd.read_csv(temp_file) # CSV Alternative
                dfs_to_concat.append(df_chunk)
            except FileNotFoundError:
                 logger.warning(f"Temporary file not found: {temp_file}. Skipping.")
                 success = False # Mark as potentially incomplete
            except Exception as e:
                 logger.error(f"Error reading temporary file {temp_file}: {e}. Skipping.")
                 success = False # Mark as potentially incomplete

        if not dfs_to_concat:
             logger.error("No temporary files could be read. Combined DataFrame will be empty.")
             return pd.DataFrame(columns=final_columns).astype(dtypes) # Return empty DF with correct structure

        # Concatenate all chunks
        combined_df = pd.concat(dfs_to_concat, ignore_index=True)

        # Re-apply desired dtypes (important after concat)
        for col, dtype in dtypes.items():
             if col in combined_df.columns:
                try:
                    combined_df.loc[col] = combined_df[col].astype(dtype)
                except Exception as e:
                     logger.warning(f"Could not apply dtype '{dtype}' to column '{col}' after combining: {e}")

        logger.info(f"Successfully combined temporary files into DataFrame with {len(combined_df)} rows.")
        if not success:
            logger.warning("Combined DataFrame may be incomplete due to errors reading temporary files.")

        return combined_df

    except MemoryError:
        logger.error("MemoryError during final concatenation of temporary files!")
        # Critical error, likely cannot proceed
        return pd.DataFrame() # Return empty DF
    except Exception as e:
        logger.error(f"Unexpected error during final concatenation: {e}")
        return pd.DataFrame() # Return empty DF
    finally:
        # Clean up temporary files regardless of success/failure
        logger.info("Cleaning up temporary batch files...")
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Could not remove temporary file {temp_file}: {e}")
        # Explicit GC after cleanup
        del dfs_to_concat # Clear list of dataframes
        gc.collect()

@cython.boundscheck(False)
@cython.wraparound(False)
def parse_content_lines_sequential(line_dicts: List[Dict[str, Any]], batch_size: cython.int) -> pd.DataFrame:
    """
    Parse content lines sequentially, saving intermediate results to disk to conserve memory.

    Args:
        line_dicts: The full list of collected line dictionaries.
        batch_size: How many lines to process in each sequential step.

    Returns:
        A Pandas DataFrame containing the parsed content.
    """
    cdef:
        cython.int i, batch_num, total_lines, batch_count
        list batch, rows, temp_files = []
        str temp_dir = "temp_batches"
        str temp_file_path
        object temp_df # Use object for DataFrame type
        dict dtypes = {'id': str, 'content': str, 'file': str, 'file_order': 'int32', 'line_num': 'int32', 'sequence': 'int64'}


    total_lines = len(line_dicts)
    if total_lines == 0:
        logger.warning("No content lines provided to sequential parser.")
        return pd.DataFrame()

    logger.info(f"Starting sequential parsing of {total_lines} lines with batch size {batch_size}.")

    # Create a temporary directory for batch files
    try:
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        else: # Clean up old temp files if directory exists
             for old_file in os.listdir(temp_dir):
                 if old_file.startswith("temp_batch_") and (old_file.endswith(".parquet") or old_file.endswith(".csv")):
                     try: os.remove(os.path.join(temp_dir, old_file))
                     except: pass
    except OSError as e:
        logger.error(f"Could not create or clean temporary directory {temp_dir}: {e}. Aborting sequential parse.")
        return pd.DataFrame()

    batch_count = (total_lines + batch_size - 1) // batch_size

    try:
        for i in range(0, total_lines, batch_size):
            batch_num = i // batch_size + 1
            batch = line_dicts[i : i + batch_size]

            logger.info(f"Processing sequential batch {batch_num}/{batch_count} ({len(batch)} lines)")

            # Process this batch in the current process
            rows = process_content_batch(batch)

            if rows:
                try:
                    # Create a temporary DataFrame
                    temp_df = pd.DataFrame(rows)
                    # Apply dtypes immediately
                    for col, dtype in dtypes.items():
                        if col in temp_df.columns:
                            try: temp_df.loc[col] = temp_df[col].astype(dtype)
                            except: pass # Ignore dtype errors for now

                    # Define temp file path
                    temp_file_path = os.path.join(temp_dir, f"temp_batch_{batch_num}.parquet")
                    # Save to Parquet (or CSV)
                    if _save_temp_dataframe(temp_df, temp_file_path):
                        temp_files.append(temp_file_path)

                except MemoryError:
                    logger.error(f"MemoryError creating or saving DataFrame for batch {batch_num}. Skipping batch.")
                    # Continue to next batch if possible
                except Exception as e:
                    logger.error(f"Error processing or saving batch {batch_num}: {e}. Skipping batch.")
                finally:
                    # Clear memory explicitly
                    del temp_df
                    del rows
                    del batch # Clear slice
                    gc.collect()
            else:
                logger.warning(f"Batch {batch_num} resulted in no valid rows.")
                del batch # Clear slice
                gc.collect()

        # Combine all temporary files
        if temp_files:
            combined_df = _combine_temp_files(temp_files, list(dtypes.keys()), dtypes)
            return combined_df
        else:
            logger.warning("No valid data batches were processed successfully.")
            return pd.DataFrame()

    except Exception as e:
        logger.error(f"Unexpected error during sequential processing loop: {str(e)}")
        return pd.DataFrame() # Return empty on major failure
    finally:
        # Ensure temp dir is cleaned up if possible, even on error
        try:
            if os.path.exists(temp_dir) and not temp_files: # Only remove if empty or fully processed
                 pass # Keep temp files if _combine failed midway for debugging? Or remove always?
            # Let's remove always for cleanliness
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.info(f"Removed temporary directory: {temp_dir}")
        except Exception as e:
             logger.warning(f"Could not remove temporary directory {temp_dir}: {e}")


@cython.boundscheck(False)
@cython.wraparound(False)
def parse_content_lines_parallel(line_dicts: List[Dict[str, Any]], batch_size: cython.int, num_workers: cython.int) -> pd.DataFrame:
    """
    Parse content lines into a DataFrame using parallel processing, with sequential fallback.

    Args:
        line_dicts: List of collected line dictionaries.
        batch_size: Size of batches submitted to workers.
        num_workers: Number of worker processes.

    Returns:
        A Pandas DataFrame containing the parsed content.
    """
    cdef:
        cython.int total_lines = len(line_dicts)
        double available_memory_gb = get_available_memory()
        list batches, all_rows = []
        cython.int chunk_size_batches # How many batches to submit to pool at once
        cython.int batch_chunk_idx, num_batch_chunks
        list batch_chunk, results
        object pool # multiprocessing.Pool object
        dict dtypes = {'id': str, 'content': str, 'file': str, 'file_order': 'int32', 'line_num': 'int32', 'sequence': 'int64'}
        object df # Use object for DataFrame type
        double mem_before, mem_after, mem_used_gb
        cython.int i # Loop variable
        # ---- Added flag to track successful parallel completion ----
        bint parallel_completed_successfully = False


    if total_lines == 0:
        logger.warning("No content lines provided to parallel parser.")
        return pd.DataFrame()

    # Memory check: If estimated needs are very high vs available memory, or too many lines, go sequential
    if total_lines > 500_000 and available_memory_gb < 8.0:
        logger.warning(f"Large dataset ({total_lines} lines) and limited memory ({available_memory_gb:.1f} GB). Switching to sequential parsing.")
        # Pass line_dicts directly, it hasn't been deleted yet
        return parse_content_lines_sequential(line_dicts, batch_size)

    logger.info(f"Starting parallel parsing of {total_lines} lines.")
    logger.info(f"Using batch size: {batch_size}, workers: {num_workers}")

    # Split lines into batches for workers
    batches = [line_dicts[i : i + batch_size] for i in range(0, total_lines, batch_size)]
    # --- REMOVED: del line_dicts --- We need it for fallback
    # gc.collect() # Keep gc.collect after creating potentially large 'batches' list
    logger.info(f"Created {len(batches)} batches.")


    # Determine chunk size for submitting batches to the pool
    chunk_size_batches = max(1, min(BATCH_PROCESSING_CHUNK_SIZE, (len(batches) + num_workers - 1) // num_workers))
    num_batch_chunks = (len(batches) + chunk_size_batches - 1) // chunk_size_batches
    logger.info(f"Processing batches in {num_batch_chunks} chunks of up to {chunk_size_batches} batches each.")

    try:
        mem_before = get_available_memory()

        for batch_chunk_idx in range(num_batch_chunks):
            start_idx = batch_chunk_idx * chunk_size_batches
            end_idx = min(start_idx + chunk_size_batches, len(batches))
            batch_chunk = batches[start_idx:end_idx]

            if not batch_chunk: continue

            logger.info(f"Processing batch chunk {batch_chunk_idx + 1}/{num_batch_chunks} ({len(batch_chunk)} batches)...")

            # Create pool inside the loop, use maxtasksperchild
            with multiprocessing.Pool(processes=num_workers, maxtasksperchild=20) as pool:
                # Use imap_unordered for better memory usage potential
                imap_chunksize = max(1, len(batch_chunk) // (num_workers * 4)) # Heuristic
                results_iterator = pool.imap_unordered(process_content_batch, batch_chunk, chunksize=imap_chunksize)

                # Consume results as they arrive
                for batch_rows in results_iterator:
                     if batch_rows: # Only extend if the batch produced rows
                         all_rows.extend(batch_rows)

            # Memory check and potential fallback after processing a chunk
            mem_after = get_available_memory()
            mem_used_gb = mem_before - mem_after
            logger.info(f"Finished batch chunk {batch_chunk_idx + 1}. Rows collected so far: {len(all_rows)}. Approx memory used by chunk: {mem_used_gb:.2f} GB. Available: {mem_after:.1f} GB.")

            if mem_after < max(1.0, available_memory_gb * 0.1): # If less than 1GB or 10% of initial available memory remains
                logger.warning(f"Low memory detected ({mem_after:.1f} GB). Falling back to sequential processing for remaining batches.")

                logger.info("Saving currently collected rows to temporary file before sequential fallback...")
                temp_fallback_df = pd.DataFrame() # Placeholder
                temp_fallback_file = None
                if all_rows:
                     try:
                         temp_fallback_df = pd.DataFrame(all_rows)
                         for col, dtype in dtypes.items():
                             if col in temp_fallback_df.columns:
                                 try: temp_fallback_df.loc[col] = temp_fallback_df[col].astype(dtype)
                                 except: pass
                         temp_fallback_file = "temp_fallback_partial.parquet"
                         _save_temp_dataframe(temp_fallback_df, temp_fallback_file)
                         all_rows = [] # Clear memory
                         gc.collect()
                     except Exception as save_err:
                         logger.error(f"Could not save partial results before fallback: {save_err}. Proceeding without saving.")
                         temp_fallback_file = None # Ensure it's None

                # Get the actual line dictionaries for the remaining batches
                remaining_line_dicts = []
                logger.info("Gathering line dictionaries for remaining batches...")
                try:
                    current_line_index = end_idx * batch_size # Estimate starting index in original line_dicts
                    for i in range(batch_chunk_idx + 1, num_batch_chunks):
                         r_start_idx = i * chunk_size_batches
                         r_end_idx = min(r_start_idx + chunk_size_batches, len(batches))
                         for batch in batches[r_start_idx:r_end_idx]:
                              remaining_line_dicts.extend(batch) # Extend directly from batch data

                    # ---- Alternatively, reslice original line_dicts ----
                    # This assumes batches were created sequentially from line_dicts
                    # first_remaining_line_idx = end_idx * batch_size # Index of first line NOT processed in parallel
                    # if first_remaining_line_idx < total_lines:
                    #      remaining_line_dicts = line_dicts[first_remaining_line_idx:]
                    # else:
                    #      remaining_line_dicts = [] # Should not happen if loop logic is correct
                    # logger.info(f"Gathered {len(remaining_line_dicts)} remaining lines for sequential processing.")

                except MemoryError:
                     logger.error("MemoryError gathering remaining line dictionaries for sequential fallback.")
                     if temp_fallback_file:
                         logger.warning("Returning only data processed before low memory condition.")
                         return _combine_temp_files([temp_fallback_file], list(dtypes.keys()), dtypes)
                     else:
                         return pd.DataFrame() # Nothing could be saved/processed
                except Exception as gather_err:
                     logger.error(f"Error gathering remaining lines: {gather_err}")
                     # Decide how to proceed - maybe return partial if saved?
                     if temp_fallback_file:
                         logger.warning("Returning only data processed before low memory condition due to error gathering remaining lines.")
                         return _combine_temp_files([temp_fallback_file], list(dtypes.keys()), dtypes)
                     else:
                         return pd.DataFrame()

                # Run sequential process on remaining lines
                sequential_df = parse_content_lines_sequential(remaining_line_dicts, batch_size)

                # Clean up remaining data structures
                del remaining_line_dicts
                gc.collect()

                # Combine partial parallel results (if saved) with sequential results
                if temp_fallback_file:
                     partial_df = _combine_temp_files([temp_fallback_file], list(dtypes.keys()), dtypes)
                     final_df = pd.concat([partial_df, sequential_df], ignore_index=True)
                     del partial_df # Clean up intermediate df
                else: # No partial results saved
                     final_df = sequential_df

                # Re-apply dtypes one last time after concat
                if not final_df.empty:
                    for col, dtype in dtypes.items():
                        if col in final_df.columns:
                            try: final_df.loc[col] = final_df[col].astype(dtype)
                            except: pass

                logger.info(f"Low memory fallback complete. Final DataFrame has {len(final_df)} rows.")
                # --- Important: Set flag to indicate parallel part didn't fully complete ---
                parallel_completed_successfully = False # Set to false as we bailed early
                # --- Clean up original line_dicts here as it's no longer needed ---
                del line_dicts
                gc.collect()
                return final_df # Exit outer loop and function, returning combined result

            # Update memory baseline for next chunk check
            mem_before = mem_after

            # Clean up memory after chunk processing
            del batch_chunk
            del results_iterator # Ensure iterator is closed/deleted
            gc.collect()

        # ---- If the loop completes without low-memory fallback ----
        parallel_completed_successfully = True

    except (MemoryError, OSError, Exception) as e:
        logger.error(f"Fatal error during parallel batch processing: {type(e).__name__} - {e}")
        logger.info("Attempting fallback to full sequential batch processing using original data...")

        # ---- Fallback logic ----
        # At this point, line_dicts should still exist because we didn't delete it yet.
        try:
            # Directly call sequential parse with the original line_dicts
            # No need for NameError check here anymore.
            logger.info(f"Falling back to process all {len(line_dicts)} lines sequentially.")
            sequential_fallback_df = parse_content_lines_sequential(line_dicts, batch_size)
            # --- Clean up original line_dicts after fallback attempt ---
            del line_dicts
            gc.collect()
            return sequential_fallback_df # Return result of sequential processing

        except Exception as fallback_err:
             logger.error(f"Sequential fallback also failed: {fallback_err}")
             # --- Clean up original line_dicts even if fallback fails ---
             try: del line_dicts
             except NameError: pass # Already deleted or never existed (edge case)
             gc.collect()
             # Try to return partially collected rows if any exist from before the fatal error
             if all_rows:
                  logger.warning("Attempting to return partially collected rows from before the fatal error.")
                  try:
                      df_partial = pd.DataFrame(all_rows)
                      for col, dtype in dtypes.items():
                           if col in df_partial.columns:
                               try: df_partial[col] = df_partial[col].astype(dtype)
                               except: pass
                      return df_partial
                  except Exception as partial_err:
                       logger.error(f"Could not create DataFrame from partial rows: {partial_err}")
                       return pd.DataFrame() # Give up, return empty
             else:
                  return pd.DataFrame() # Give up, return empty


    # ---- If parallel processing completed successfully (no low memory fallback, no fatal error) ----
    if parallel_completed_successfully:
        # --- Clean up original line_dicts NOW, after successful parallel run ---
        logger.info("Parallel processing successful. Deleting original line dictionaries list.")
        del line_dicts
        gc.collect()

        if not all_rows:
            logger.warning("Parallel processing completed successfully, but no valid data rows were generated.")
            return pd.DataFrame()

        logger.info(f"Parallel processing finished. Creating final DataFrame from {len(all_rows)} collected rows...")
        try:
            df = pd.DataFrame(all_rows)
            # Apply dtypes
            for col, dtype in dtypes.items():
                if col in df.columns:
                    try:
                        df.loc[col] = df[col].astype(dtype)
                    except Exception as e:
                        logger.warning(f"Could not apply dtype '{dtype}' to column '{col}' after parallel processing: {e}")

            logger.info(f"Successfully parsed {len(df)} valid content lines using parallel processing.")
            # --- Clean up all_rows after creating final DataFrame ---
            del all_rows
            gc.collect()
            return df
        except MemoryError:
            logger.error("MemoryError during final DataFrame creation after successful parallel processing.")
            del all_rows # Try to free memory
            gc.collect()
            return pd.DataFrame() # Return empty
        except Exception as e:
            logger.error(f"Error creating final DataFrame after parallel processing: {e}")
            del all_rows # Try to free memory
            gc.collect()
            return pd.DataFrame() # Return empty
    else:
         # This case should theoretically not be reached if fallbacks return correctly,
         # but as a safeguard:
         logger.warning("Parallel processing flag indicates incompletion, but no fallback path returned. Returning empty DataFrame.")
         try: del line_dicts
         except NameError: pass
         gc.collect()
         return pd.DataFrame()

# --- Merging and Formatting ---

@cython.boundscheck(False)
@cython.wraparound(False)
def merge_data(content_df: pd.DataFrame, sources_df: pd.DataFrame) -> pd.DataFrame:
    """Merges content and sources DataFrames with memory usage logging."""
    cdef:
        object merged_df # Use object for DataFrame type
        double initial_memory = content_df.memory_usage(deep=True).sum() / (1024**2) + \
                               sources_df.memory_usage(deep=True).sum() / (1024**2)

    if content_df.empty or sources_df.empty:
        logger.warning("One or both DataFrames are empty, cannot merge.")
        return pd.DataFrame()

    logger.info(f"Starting merge. Content DF: {len(content_df)} rows, Sources DF: {len(sources_df)} rows.")
    logger.info(f"Approximate memory usage before merge: {initial_memory:.1f} MB")

    try:
        # Ensure 'id' column exists and is string type in both
        if 'id' not in content_df.columns or 'id' not in sources_df.columns:
             logger.error("Missing 'id' column in one of the DataFrames for merging.")
             return pd.DataFrame()
        if content_df['id'].dtype != 'object': content_df['id'] = content_df['id'].astype(str)
        if sources_df['id'].dtype != 'object': sources_df['id'] = sources_df['id'].astype(str)

        merged_df = pd.merge(content_df, sources_df, on='id', how='inner') # Inner join common IDs

        final_memory = merged_df.memory_usage(deep=True).sum() / (1024**2)
        logger.info(f"Merge complete. Result: {len(merged_df)} rows.")
        logger.info(f"Approximate memory usage after merge: {final_memory:.1f} MB")

        # Optimize merged DataFrame memory
        # Convert low-cardinality columns inherited from sources_df back to category if needed
        if 'type' in merged_df.columns and merged_df['type'].dtype == 'object':
             merged_df.loc['type'] = merged_df['type'].astype('category')

        # Sort by original sequence to maintain order
        logger.info("Sorting merged data by sequence...")
        merged_df = merged_df.sort_values('sequence', ignore_index=True) # ignore_index resets index
        logger.info("Sorting complete.")

        return merged_df

    except MemoryError:
        logger.error("MemoryError during DataFrame merge!")
        gc.collect() # Attempt to free memory
        return pd.DataFrame() # Return empty on failure
    except Exception as e:
        logger.error(f"Error during DataFrame merge: {e}")
        return pd.DataFrame() # Return empty on failure

@cython.boundscheck(False)
@cython.wraparound(False)
def format_for_validation(merged_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Format merged data into the target validation structure, processing in batches.

    Args:
        merged_df: The merged and sorted DataFrame.

    Returns:
        A list of dictionaries in the validation format.
    """
    cdef:
        list validation_format = []
        list conversation
        cython.int start_idx, end_idx, total_rows, batch_size, batch_num = 0, total_batches
        object batch_df = None # Initialize batch_df to None
        dict row_dict # Not used currently
        object row # Variable for itertuples result
        cython.Py_ssize_t index # Not used with itertuples(index=False)
        list required_cols = ['content', 'source'] # Define required cols once

    total_rows = len(merged_df)
    if total_rows == 0:
        logger.warning("No data in merged DataFrame to format.")
        return []

    batch_size = 10000
    total_batches = (total_rows + batch_size - 1) // batch_size

    logger.info(f"Formatting {total_rows} rows for validation structure in {total_batches} batches...")

    try:
        for start_idx in range(0, total_rows, batch_size):
            batch_num = start_idx // batch_size + 1
            end_idx = min(start_idx + batch_size, total_rows)

            logger.info(f"Formatting validation batch {batch_num}/{total_batches}")

            try: # Add inner try/finally for batch_df cleanup
                batch_df = merged_df.iloc[start_idx:end_idx]

                # Ensure required columns ('content', 'source') exist before iterating
                if not all(col in batch_df.columns for col in required_cols):
                    logger.error(f"Missing required columns {required_cols} in batch {batch_num}. Skipping iteration for this batch.")
                    # --- Just continue to the finally block for cleanup ---
                    continue # Skips the 'for row in batch_df.itertuples()' part

                # Iterate using itertuples
                for row in batch_df.itertuples(index=False):
                    content_val = str(row.content) if pd.notna(row.content) else ""
                    source_val = str(row.source) if pd.notna(row.source) else "unknown"

                    conversation = [
                        {
                            "from": "human",
                            "value": content_val
                        },
                        {
                            "from": "gpt",
                            "value": "Tell the user what dialect this is and provide additional context and learn the dialect." # Fixed template
                        }
                    ]

                    validation_format.append({
                        "conversations": conversation,
                        "source": source_val,
                        "score": 0,  # Fixed score
                    })

            finally:
                # --- Always cleanup batch_df for the current iteration ---
                # This block executes whether the try block finished,
                # hit a 'continue', or raised an exception caught by the outer handler.
                if batch_df is not None:
                    # logger.debug(f"Cleaning up batch_df for batch {batch_num}") # Optional debug log
                    del batch_df
                    batch_df = None # Reset for next iteration or if error occurs
                    gc.collect()

        logger.info(f"Successfully formatted {len(validation_format)} records.")
        return validation_format

    except MemoryError:
         logger.error(f"MemoryError during validation formatting (around batch {batch_num}). Partial results may be returned.")
         # No explicit cleanup needed here, finally block within loop handles batch_df
         return validation_format
    except Exception as e:
        logger.error(f"Error during validation formatting (around batch {batch_num}): {e}", exc_info=True)
        # No explicit cleanup needed here, finally block within loop handles batch_df
        return validation_format

# --- Main Processing Orchestration ---

@cython.boundscheck(False)
@cython.wraparound(False)
def process_data(input_folder: str, sources_file: str, num_workers: cython.int = 0) -> List[Dict[str, Any]]:
    """
    Main data processing pipeline function.

    Args:
        input_folder: Path to the corpora directory.
        sources_file: Path to the sources.txt file.
        num_workers: Number of workers for parallel steps (0 for auto).

    Returns:
        A list of dictionaries in the final validation format, or empty list on failure.
    """
    cdef:
        object sources_df, content_df, merged_df # Pandas objects
        list all_line_dicts, result
        cython.int batch_size, optimal_workers

    logger.info("="*20 + " Starting Data Processing Pipeline " + "="*20)

    # 1. Parse Sources
    logger.info("--- Step 1: Parsing Sources File ---")
    sources_df = parse_sources_file(sources_file)
    if sources_df.empty:
        logger.error("Failed to load sources data. Aborting pipeline.")
        return []
    gc.collect() # Collect after loading sources

    # 2. Collect Content Lines
    logger.info("--- Step 2: Collecting Content Lines (Parallel) ---")
    # Determine workers specifically for collection if needed, or use overall num_workers
    # Let's use the passed/auto-calculated num_workers for consistency
    all_line_dicts = collect_content_lines_parallel(input_folder, num_workers)
    if not all_line_dicts:
        logger.error("Failed to collect any content lines. Aborting pipeline.")
        # Clean up sources DF before exiting
        del sources_df
        gc.collect()
        return []
    logger.info(f"Collected {len(all_line_dicts)} total '@@' lines.")
    gc.collect() # Collect after gathering all lines into memory

    # 3. Parse Content Lines into DataFrame
    logger.info("--- Step 3: Parsing Content Lines (Parallel/Sequential) ---")
    # Calculate optimal parameters based on the number of *lines* collected
    batch_size, optimal_workers = calculate_processing_parameters(len(all_line_dicts), "lines")
    if num_workers > 0: # Allow user override, capped by calculated optimum
        actual_parse_workers = max(1, min(num_workers, optimal_workers))
    else:
        actual_parse_workers = optimal_workers
    logger.info(f"Using {actual_parse_workers} workers and batch size {batch_size} for parsing.")

    content_df = parse_content_lines_parallel(all_line_dicts, batch_size, actual_parse_workers)
    # Crucially, free the large list of line dictionaries now
    logger.info("Deleting raw collected line data...")
    del all_line_dicts
    gc.collect()
    logger.info("Raw line data deleted.")

    if content_df.empty:
        logger.error("Failed to parse content lines into DataFrame. Aborting pipeline.")
        del sources_df
        gc.collect()
        return []
    logger.info(f"Parsed content into DataFrame with {len(content_df)} rows.")
    gc.collect() # Collect after creating content_df

    # 4. Merge DataFrames
    logger.info("--- Step 4: Merging Content and Sources ---")
    merged_df = merge_data(content_df, sources_df)
    # Free individual DataFrames after merge
    logger.info("Deleting individual content and sources DataFrames...")
    del content_df
    del sources_df
    gc.collect()
    logger.info("Individual DataFrames deleted.")

    if merged_df.empty:
        logger.error("Merging failed or resulted in empty DataFrame. Aborting pipeline.")
        return []
    logger.info(f"Merging complete. Final merged DataFrame has {len(merged_df)} rows.")
    gc.collect() # Collect after merge and sort

    # 5. Format for Validation
    logger.info("--- Step 5: Formatting Data for Validation ---")
    result = format_for_validation(merged_df)
    # Free merged DataFrame after formatting
    logger.info("Deleting merged DataFrame...")
    del merged_df
    gc.collect()
    logger.info("Merged DataFrame deleted.")

    if not result:
        logger.warning("Formatting resulted in empty list, but processing technically succeeded.")
    else:
         logger.info(f"Successfully formatted {len(result)} records for validation.")

    logger.info("="*20 + " Data Processing Pipeline Finished " + "="*20)
    return result


# --- Exporting ---

def export_to_parquet_and_excel(data: List[Dict[str, Any]], output_basename: str, chunk_size: cython.int = EXPORT_CHUNK_SIZE):
    """
    Export data to Parquet and optionally chunked Excel file.

    Args:
        data: List of processed data records (dictionaries).
        output_basename: Base name for the output files (e.g., 'output').
                         '.parquet' and '.xlsx' will be appended.
        chunk_size: Number of rows per chunk for writing.
    """
    cdef:
        cython.int total_rows = len(data)
        cython.int num_chunks, i, chunk_num, start_idx, end_idx
        list chunk_data
        object df_chunk # Pandas DataFrame object
        str parquet_file = f"{output_basename}.parquet"
        str excel_file = f"{output_basename}.xlsx"
        object pq_writer = None # PyArrow ParquetWriter
        object excel_writer = None # Pandas ExcelWriter
        object schema = None # Pyarrow schema

    if total_rows == 0:
        logger.warning("No data provided to export.")
        return

    logger.info(f"Starting export of {total_rows} records...")
    logger.info(f"Parquet output: {parquet_file}")
    logger.info(f"Excel output: {excel_file}")

    num_chunks = (total_rows + chunk_size - 1) // chunk_size

    try:
        # --- Parquet Export (Chunked using PyArrow) ---
        logger.info(f"Writing Parquet file in {num_chunks} chunks...")

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk_data = data[start_idx:end_idx]
            chunk_num = i + 1

            if not chunk_data: continue

            logger.debug(f"Preparing Parquet chunk {chunk_num}/{num_chunks} ({len(chunk_data)} rows)")
            # Keep chunk processing within its own try-except to handle chunk-specific errors
            try:
                df_chunk = pd.DataFrame(chunk_data)
                table = pa.Table.from_pandas(df_chunk, preserve_index=False)

                if schema is None:
                    schema = table.schema
                    # Initialize ParquetWriter here, only if data exists
                    pq_writer = pq.ParquetWriter(parquet_file, schema, compression='snappy')

                # Schema check (optional, can be removed if causing issues and data is uniform)
                if table.schema != schema:
                     logger.warning(f"Schema mismatch in Parquet chunk {chunk_num}. Casting to expected schema.")
                     try:
                         table = table.cast(schema, safe=False) # Cast to the writer's schema
                     except Exception as cast_err:
                         logger.error(f"Failed to cast table schema for chunk {chunk_num}: {cast_err}. Skipping chunk.")
                         # Clean up chunk-specific resources before skipping
                         del df_chunk
                         del table
                         gc.collect()
                         continue # Skip to the next chunk

                # Write the table (potentially casted)
                if pq_writer: # Ensure writer was initialized
                    pq_writer.write_table(table)
                    logger.info(f"Written Parquet chunk {chunk_num}/{num_chunks}")
                else:
                    # This should not happen if schema was derived, but safety check
                    logger.error(f"Parquet writer not initialized when trying to write chunk {chunk_num}. Skipping.")


            except Exception as chunk_err:
                 logger.error(f"Error processing or writing Parquet chunk {chunk_num}: {chunk_err}")
                 # Continue to the next chunk? Or re-raise to abort? Let's continue.
            finally:
                # Clean up chunk-specific resources immediately
                if 'df_chunk' in locals(): del df_chunk
                if 'table' in locals(): del table
                gc.collect()

        # Close the writer *after* the loop if it was initialized
        if pq_writer:
             logger.info("Closing Parquet writer...")
             pq_writer.close()
             logger.info(f"Parquet file '{parquet_file}' successfully written (or attempted).")
        else:
             # This could happen if the input `data` was empty or all chunks failed before schema inference
             logger.warning("Parquet writer was never initialized (likely no valid data chunks). Parquet file may not exist or be empty.")

    except MemoryError:
        # Error Handling: Keep logging, remove the faulty .closed check
        logger.error("MemoryError during export process.")
        # No need to explicitly close pq_writer here, finally block will handle it.
        gc.collect()
    except Exception as e:
        # Error Handling: Keep logging, remove the faulty .closed check
        logger.error(f"Unexpected error during export: {e}", exc_info=True) # Log traceback for unexpected errors
        # No need to explicitly close pq_writer here, finally block will handle it.
    finally:
         # Final Cleanup Attempt for Parquet Writer
         if pq_writer: # Check if the writer object exists
             logger.info("Ensuring Parquet writer is closed in finally block...")
             try:
                 pq_writer.close()
             except Exception as close_err:
                 # Log if closing itself fails, but don't crash the script
                 logger.warning(f"Error encountered while closing Parquet writer in finally block: {close_err}")
         logger.info("Export process finished.")

# --- Main Execution Block ---

def main():
    """Main entry point for the script."""
    cdef:
        str platform_type, input_folder, sources_txt, output_base
        double available_memory_gb
        cython.int num_workers = 0 # 0 means auto-calculate
        list processed_data
        object df_preview # Pandas DataFrame object

    # 1. Configure Environment
    configure_multiprocessing()
    platform_type = get_platform_info()
    available_memory_gb = get_available_memory()
    logger.info(f"--- System Information ---")
    logger.info(f"Platform: {platform_type}")
    logger.info(f"Python Version: {sys.version.split()[0]}")
    logger.info(f"Pandas Version: {pd.__version__}")
    logger.info(f"PyArrow Version: {pa.__version__}")
    logger.info(f"Available Memory: {available_memory_gb:.1f} GB")
    logger.info(f"CPU Count (Logical): {psutil.cpu_count(logical=True)}")
    logger.info(f"Multiprocessing Start Method: {multiprocessing.get_start_method(allow_none=True)}")
    logger.info(f"--------------------------")


    # 2. Define Paths
    # Use environment variables or command-line args for flexibility?
    input_folder = os.getenv('INPUT_FOLDER', 'corpora')
    sources_txt = os.getenv('SOURCES_FILE', 'sources.txt')
    output_base = os.getenv('OUTPUT_BASENAME', 'output') # Base name for output files

    logger.info(f"Input Folder: {input_folder}")
    logger.info(f"Sources File: {sources_txt}")
    logger.info(f"Output Base Name: {output_base}")

    # Check if input exists
    if not os.path.isdir(input_folder):
         logger.error(f"Input folder not found: {input_folder}")
         sys.exit(1)
    if not os.path.isfile(sources_txt):
         logger.error(f"Sources file not found: {sources_txt}")
         sys.exit(1)


    # 3. Determine Number of Workers (can be overridden by env var)
    try:
        env_workers = os.getenv('NUM_WORKERS')
        if env_workers:
            num_workers = int(env_workers)
            logger.info(f"Using NUM_WORKERS from environment variable: {num_workers}")
        else:
            # Auto-calculate based on overall heuristics (will be refined per step)
            _, num_workers = calculate_processing_parameters(100000, "items") # Use dummy item count for initial calc
            logger.info(f"Auto-calculated initial number of workers: {num_workers} (will be adjusted per step)")
    except ValueError:
        logger.warning(f"Invalid NUM_WORKERS environment variable value: {env_workers}. Using auto-calculation.")
        _, num_workers = calculate_processing_parameters(100000, "items")


    # 4. Run Processing Pipeline
    try:
        processed_data = process_data(input_folder, sources_txt, num_workers)

        # 5. Export Results
        if processed_data:
            logger.info(f"Processing completed. Exporting {len(processed_data)} records...")
            export_to_parquet_and_excel(processed_data, output_base, chunk_size=EXPORT_CHUNK_SIZE)

            # 6. Preview Output (Optional and Memory-Safe)
            logger.info("--- Output Preview (First 5 Rows) ---")
            try:
                output_parquet_file = f"{output_base}.parquet"
                # Read only specific columns and head for preview
                df_preview = pd.read_parquet(output_parquet_file).head(5)
                # Pretty print the preview
                print(df_preview.to_string())
                logger.info("---------------------------------------")

            except FileNotFoundError:
                 logger.warning(f"Output file {output_parquet_file} not found for preview.")
            except Exception as e:
                logger.warning(f"Could not preview output file: {e}")
        else:
            logger.warning("Processing pipeline returned no data to export.")

    except MemoryError:
        logger.critical("A MemoryError occurred during the main execution. The process might be unstable or results incomplete.")
        gc.collect() # Try to free some memory before exiting
        sys.exit(2) # Exit with error code
    except KeyboardInterrupt:
         logger.info("Process interrupted by user (KeyboardInterrupt). Exiting.")
         sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred in main execution: {e}", exc_info=True) # Log traceback
        sys.exit(3) # Exit with different error code

    logger.info("Script finished successfully.")

if __name__ == "__main__":
    # Crucial for multiprocessing, especially on Windows/macOS with 'spawn'
    # configure_multiprocessing() is called inside main() now to log settings first.
    main()
