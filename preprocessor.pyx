# cython: language_level=3
# distutils: language = c

import re
import pandas as pd
import logging
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional, Tuple
import os
import functools
import multiprocessing
from multiprocessing import Pool, cpu_count
import platform
import sys
import gc
import psutil
import math
cimport cython
from libc.stdlib cimport malloc, free

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Define constants
ENCODINGS = ['utf-8', 'latin-1', 'cp1252']

# Public function that users will call
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef str clean_text(str text_obj):
    """
    Clean text by applying various text cleaning operations.
    This is the public API that calls the cached implementation.
    
    Args:
        text_obj: The text to clean
        
    Returns:
        Cleaned text
    """
    # Implement actual text cleaning if needed
    # For now, just returning the input
    return text_obj

# Add platform detection
def get_platform_info():
    """Detect the platform and return information about it."""
    cdef str system = platform.system()
    if system == "Linux":
        # Check if running in WSL
        if "microsoft" in platform.uname().release.lower():
            return "WSL"
        return "Linux"
    elif system == "Darwin":
        return "MacOS"
    elif system == "Windows":
        return "Windows"
    return "Unknown"

# Get available system memory
def get_available_memory():
    """Get available system memory in GB."""
    cdef double available_gb, total_gb
    
    try:
        mem_info = psutil.virtual_memory()
        available_gb = mem_info.available / (1024 ** 3)
        total_gb = mem_info.total / (1024 ** 3)
        logger.info(f"System memory: {total_gb:.1f} GB total, {available_gb:.1f} GB available")
        return available_gb
    except Exception as e:
        logger.warning(f"Could not determine available memory: {e}")
        return 4.0  # Conservative default

# Configure multiprocessing based on platform and memory
def configure_multiprocessing():
    """Configure multiprocessing based on the platform."""
    cdef str platform_type = get_platform_info()
    
    # Set multiprocessing start method
    if platform_type in ["Windows", "WSL"]:
        # Windows and WSL need specific multiprocessing configuration
        multiprocessing.set_start_method('spawn', force=True)
        logger.info(f"Configured multiprocessing for {platform_type} using 'spawn' method")
    elif platform_type == "MacOS":
        # macOS should use 'fork' on older versions and 'spawn' on newer ones
        if sys.version_info >= (3, 8):
            multiprocessing.set_start_method('spawn', force=True)
            logger.info("Configured multiprocessing for MacOS using 'spawn' method")
        else:
            multiprocessing.set_start_method('fork', force=True)
            logger.info("Configured multiprocessing for MacOS using 'fork' method")
    else:
        # Linux and others default to 'fork'
        multiprocessing.set_start_method('fork', force=True)
        logger.info(f"Configured multiprocessing for {platform_type} using 'fork' method")

# Calculate optimal batch size and worker count based on available memory
@cython.cdivision(True)
cpdef tuple calculate_processing_parameters(int total_lines):
    """Calculate optimal batch size and worker count based on system resources."""
    cdef:
        double available_memory = get_available_memory()
        str platform_type = get_platform_info()
        int estimated_memory_per_line = 10  # 10 KB per line (adjust based on your data)
        double total_memory_needed_gb
        double max_memory_per_batch_gb
        int optimal_batch_size, physical_cpu_count, logical_cpu_count
        int base_workers, memory_based_workers, optimal_workers
    
    # Calculate memory requirements
    total_memory_needed_gb = (total_lines * estimated_memory_per_line) / (1024 * 1024)
    logger.info(f"Estimated memory needed for all data: {total_memory_needed_gb:.2f} GB")
    
    # Determine optimal batch size based on available memory (use at most 25% of available memory per batch)
    max_memory_per_batch_gb = available_memory * 0.25
    
    # Calculate optimal batch size (at least 100 lines per batch, at most 1000)
    optimal_batch_size = min(
        max(
            100,  # Minimum batch size
            int((max_memory_per_batch_gb * 1024 * 1024) / estimated_memory_per_line)
        ),
        1000  # Maximum batch size
    )
    
    # Determine optimal worker count based on system and memory
    physical_cpu_count = psutil.cpu_count(logical=False) or 2
    logical_cpu_count = psutil.cpu_count(logical=True) or 4
    
    # Base worker count on platform
    if platform_type == "WSL":
        # WSL is memory-constrained, use fewer workers
        base_workers = min(2, physical_cpu_count)
    elif platform_type == "Windows":
        # Windows has overhead for processes
        base_workers = min(4, logical_cpu_count - 1)
    else:
        # Linux and Mac can use more workers
        base_workers = min(logical_cpu_count - 1, physical_cpu_count * 2)
    
    # Adjust worker count based on available memory
    # Each worker might need about 500MB
    memory_based_workers = max(1, int(available_memory / 0.5))
    
    # Choose the minimum to be safe
    optimal_workers = min(base_workers, memory_based_workers)
    
    logger.info(f"Optimal processing parameters: batch_size={optimal_batch_size}, workers={optimal_workers}")
    return optimal_batch_size, optimal_workers

@cython.boundscheck(False)
@cython.wraparound(False)
def read_file_with_multiple_encodings(str file_path, list encodings=None):
    """Try to read a file with multiple encodings."""
    cdef:
        str encoding
        list lines
    
    if encodings is None:
        encodings = ENCODINGS
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                lines = f.readlines()
                return lines, encoding
        except UnicodeDecodeError:
            logger.debug(f"Failed to decode {file_path} with {encoding}")
            continue
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None, ""
    
    logger.warning(f"Could not decode {file_path} with any encoding")
    return None, ""

@cython.boundscheck(False)
@cython.wraparound(False)
def parse_sources_file(str sources_file):
    """Parse the sources file into a DataFrame."""
    cdef:
        list lines
        str line, encoding
        list parts
        dict record
        list sources_data = []
    
    lines, encoding = read_file_with_multiple_encodings(sources_file)
    if not lines:
        logger.error("Could not read sources file")
        return pd.DataFrame()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 6:
            record = {
                'id': parts[0],
                'year': parts[1],
                'type': parts[2],
                'pages': parts[3],
                'source': parts[4],
                'title': ' '.join(parts[5:])
            }
            sources_data.append(record)
    
    df = pd.DataFrame(sources_data)
    if df.empty:
        logger.warning("No valid data found in sources file")
    else:
        df = df.astype({'id': str})
        logger.info(f"Loaded {len(df)} source records")
    
    return df

@cython.boundscheck(False)
@cython.wraparound(False)
def process_file(tuple file_info):
    """Process a single file and return content lines."""
    cdef:
        str root_dir = file_info[0]
        str filename = file_info[1]
        int file_order = file_info[2]
        int start_sequence = file_info[3]
        str file_path
        list content
        str encoding
        list matching_lines = []
        list result_lines = []
        int line_num
        str line
        int sequence_counter = 0
    
    file_path = os.path.join(root_dir, filename)
    
    try:
        logger.info(f"Processing {file_path}...")
        content, encoding = read_file_with_multiple_encodings(file_path)
        
        if content:
            for line_num, line in enumerate(content):
                line = line.strip()
                if line.startswith('@@'):
                    sequence_counter += 1
                    matching_lines.append(line)
                    result_lines.append({
                        'line': line,
                        'file': file_path,
                        'file_order': file_order,
                        'line_num': line_num,
                        'sequence': start_sequence + sequence_counter
                    })
            
            logger.info(f"  - Found {len(matching_lines)} lines starting with @@ using {encoding} encoding")
            # Clean up to reduce memory usage
            del content
            del matching_lines
            gc.collect()
            return True, result_lines
        else:
            logger.warning(f"  ! Could not process {file_path}")
            return False, []
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return False, []

@cython.boundscheck(False)
@cython.wraparound(False)
def collect_content_lines_parallel(str input_folder, int num_workers=0):
    """Walk through directories and collect lines starting with '@@' using parallel processing."""
    cdef:
        list file_infos = []
        int file_order = 0
        int total_files = 0
        str root
        list all_lines = []
        int processed_files = 0
        int skipped_files = 0
        int chunk_size, chunk_idx
    
    if num_workers == 0:
        num_workers = max(1, cpu_count() - 1)  # Leave one CPU for system
    
    logger.info(f"Using {num_workers} worker processes")
    
    # Get all text files and prepare for parallel processing
    for root, dirs, files in sorted(os.walk(input_folder)):
        txt_files = sorted([f for f in files if f.endswith('.txt')])
        total_files += len(txt_files)
        
        for filename in txt_files:
            # Root dir, filename, file order, starting sequence number
            file_infos.append((root, filename, file_order, file_order * 10000))
            file_order += 1
    
    logger.info(f"Found {total_files} .txt files to process")
    
    # Process files in parallel
    
    # Process in smaller chunks to manage memory better
    chunk_size = min(20, len(file_infos))
    chunks = [file_infos[i:i+chunk_size] for i in range(0, len(file_infos), chunk_size)]
    
    for chunk_idx, chunk in enumerate(chunks):
        logger.info(f"Processing file chunk {chunk_idx+1}/{len(chunks)} ({len(chunk)} files)")
        
        try:
            with Pool(processes=num_workers) as pool:
                results = pool.map(process_file, chunk)
                
                for success, lines in results:
                    if success:
                        all_lines.extend(lines)
                        processed_files += 1
                    else:
                        skipped_files += 1
                        
            # Force garbage collection between chunks
            gc.collect()
                
        except Exception as e:
            logger.error(f"Error in parallel processing chunk {chunk_idx+1}: {str(e)}")
            logger.info("Falling back to sequential processing for this chunk")
            
            # Sequential fallback if parallel processing fails
            for file_info in chunk:
                success, lines = process_file(file_info)
                if success:
                    all_lines.extend(lines)
                    processed_files += 1
                else:
                    skipped_files += 1
    
    # Sort lines by sequence to ensure correct order
    logger.info(f"Sorting {len(all_lines)} collected lines...")
    all_lines.sort(key=lambda x: x['sequence'])
    
    logger.info(f"- Processed {processed_files}/{total_files} files")
    logger.info(f"- Skipped {skipped_files} files")
    logger.info(f"- Total lines collected: {len(all_lines)}")
    
    return all_lines

@cython.boundscheck(False)
@cython.wraparound(False)
def process_content_batch(list batch):
    """Process a batch of content lines."""
    cdef:
        list rows = []
        dict line_dict
        str line
        list parts
    
    for line_dict in batch:
        line = line_dict['line']
        parts = line[2:].split(' ', 1)  # Skip the '@@' prefix
        if len(parts) == 2:
            rows.append({
                'id': parts[0],
                'content': clean_text(parts[1]),  # Apply cleaning during processing
                'file': line_dict['file'],
                'file_order': line_dict['file_order'],
                'line_num': line_dict['line_num'],
                'sequence': line_dict['sequence']
            })
    
    # Help garbage collection
    del batch
    gc.collect()
    return rows

@cython.boundscheck(False)
@cython.wraparound(False)
def parse_content_lines_sequential(list line_dicts, int batch_size=1000):
    """Parse content lines sequentially to avoid memory issues."""
    cdef:
        int i, batch_num, batch_count
        list batch, rows, temp_files = []
        str temp_file
        object temp_df, combined_df
    
    if not line_dicts:
        logger.warning("No content lines to parse")
        return pd.DataFrame()
    
    logger.info(f"Processing {len(line_dicts)} lines sequentially with batch size {batch_size}")
    
    # Process batches sequentially and write to temporary CSV files
    batch_count = math.ceil(len(line_dicts) / batch_size)
    
    try:
        for i in range(0, len(line_dicts), batch_size):
            batch_num = i // batch_size + 1
            batch = line_dicts[i:i+batch_size]
            
            logger.info(f"Processing batch {batch_num}/{batch_count} ({len(batch)} lines)")
            
            # Process this batch
            rows = process_content_batch(batch)
            
            if rows:
                # Create a temporary DataFrame and save to CSV
                temp_df = pd.DataFrame(rows)
                temp_file = f"temp_batch_{batch_num}.csv"
                temp_df.to_csv(temp_file, index=False)
                temp_files.append(temp_file)
                
                # Clear memory
                del temp_df
                del rows
                gc.collect()
            
        # Combine all temporary files
        if temp_files:
            logger.info(f"Combining {len(temp_files)} temporary files...")
            combined_df = pd.concat([pd.read_csv(f) for f in temp_files], ignore_index=True)
            combined_df = combined_df.astype({'id': str})
            
            # Delete temporary files
            for file in temp_files:
                try:
                    os.remove(file)
                except Exception as e:
                    logger.warning(f"Could not remove temporary file {file}: {e}")
            
            logger.info(f"Parsed {len(combined_df)} valid content lines")
            return combined_df
        else:
            logger.warning("No valid data lines after parsing")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error in sequential processing: {str(e)}")
        
        # Try to clean up temporary files
        for file in temp_files:
            try:
                if os.path.exists(file):
                    os.remove(file)
            except:
                pass
                
        return pd.DataFrame()

@cython.boundscheck(False)
@cython.wraparound(False)
def parse_content_lines_parallel(list line_dicts, int batch_size=1000, int num_workers=0):
    """Parse content lines into a DataFrame, falling back to sequential if parallel fails."""
    cdef:
        int total_lines = len(line_dicts)
        double available_memory = get_available_memory()
        list batches, all_rows = []
        int chunk_size, chunk_idx
        list chunks
        double mem_before, mem_after
        object seq_df
    
    # Check estimated memory needs and fall back to sequential if needed
    # If we have large number of lines and limited memory, use sequential processing
    if total_lines > 100000 and available_memory < 8:
        logger.info(f"Large dataset ({total_lines} lines) with limited memory ({available_memory:.1f} GB), using sequential processing")
        return parse_content_lines_sequential(line_dicts, batch_size)
    
    if not line_dicts:
        logger.warning("No content lines to parse")
        return pd.DataFrame()
    
    if num_workers == 0:
        num_workers = max(1, cpu_count() - 1)  # Leave one CPU for system
    
    # Split lines into batches for parallel processing
    batches = [line_dicts[i:i+batch_size] for i in range(0, len(line_dicts), batch_size)]
    logger.info(f"Processing {len(batches)} batches with {num_workers} workers")
    
    # Try parallel processing with reduced worker count and memory monitoring
    try:
        # Monitor available memory
        mem_before = get_available_memory()
        
        # Gradually process in chunks to avoid memory exhaustion
        chunk_size = min(20, len(batches))
        chunks = [batches[i:i+chunk_size] for i in range(0, len(batches), chunk_size)]
        
        for chunk_idx, chunk in enumerate(chunks):
            logger.info(f"Processing batch chunk {chunk_idx+1}/{len(chunks)} ({len(chunk)} batches)")
            
            with Pool(processes=num_workers) as pool:
                results = pool.map(process_content_batch, chunk)
                for batch_rows in results:
                    all_rows.extend(batch_rows)
            
            # Force garbage collection between chunks
            gc.collect()
            
            # Check memory usage
            mem_after = get_available_memory()
            if mem_after < 1.0:  # Less than 1GB remaining
                logger.warning(f"Low memory detected ({mem_after:.1f} GB). Falling back to sequential processing for remaining batches.")
                
                # Process remaining batches sequentially
                remaining_batches = []
                for remaining_chunk in chunks[chunk_idx+1:]:
                    remaining_batches.extend(remaining_chunk)
                
                # Flatten line_dicts for remaining batches
                remaining_lines = []
                for remaining_batch in remaining_batches:
                    remaining_lines.extend(remaining_batch)
                
                # Process sequentially and merge results
                seq_df = parse_content_lines_sequential(remaining_lines, batch_size)
                if not seq_df.empty:
                    all_rows.extend(seq_df.to_dict('records'))
                break
    except Exception as e:
        logger.error(f"Error in parallel batch processing: {str(e)}")
        logger.info("Falling back to sequential batch processing")
        
        # Try sequential processing
        return parse_content_lines_sequential(line_dicts, batch_size)
    
    if not all_rows:
        logger.warning("No valid data lines after parsing")
        return pd.DataFrame()
    
    # Create DataFrame from processed rows
    logger.info(f"Creating final DataFrame from {len(all_rows)} rows")
    df = pd.DataFrame(all_rows)
    df = df.astype({'id': str})
    
    logger.info(f"Parsed {len(df)} valid content lines")
    return df

@cython.boundscheck(False)
@cython.wraparound(False)
def format_for_validation(object merged_df):
    """Format merged data for validation structure."""
    cdef:
        list validation_format = []
        list conversation
        int start_idx, end_idx, total_rows, batch_size
    
    # Process in batches to avoid memory issues
    batch_size = 5000
    total_rows = len(merged_df)
    
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = merged_df.iloc[start_idx:end_idx]
        
        logger.info(f"Formatting validation batch {start_idx//batch_size + 1}/{(total_rows + batch_size - 1)//batch_size}")
        
        for _, row in batch_df.iterrows():
            conversation = [
                {
                    "from": "human",
                    "value": f"{row['content']}"
                },
                {
                    "from": "gpt",
                    "value": "Tell the user what dialect this is and provide additional context and learn the dialect."
                }
            ]
            
            validation_format.append({
                "conversations": conversation,
                "source": row['source'],
                "score": 0,  # Fixed score as per original code
            })
        
        # Help garbage collection
        del batch_df
        gc.collect()
    
    return validation_format

@cython.boundscheck(False)
@cython.wraparound(False)
def process_data(str input_folder, str sources_file, int num_workers=0):
    """Main data processing function with parallel processing."""
    cdef:
        object sources_df, content_df, merged_df
        list all_line_dicts, result
        int batch_size, optimal_workers
    
    # Parse sources file
    sources_df = parse_sources_file(sources_file)
    if sources_df.empty:
        return []
    
    # Collect content lines with sequence information using parallel processing
    all_line_dicts = collect_content_lines_parallel(input_folder, num_workers)
    if not all_line_dicts:
        return []
    
    # Calculate optimal batch size and worker count based on data size
    batch_size, optimal_workers = calculate_processing_parameters(len(all_line_dicts))
    if num_workers == 0:
        num_workers = optimal_workers
    
    # Parse content lines using parallel processing with optimized parameters
    content_df = parse_content_lines_parallel(all_line_dicts, batch_size=batch_size, num_workers=num_workers)
    
    # Help garbage collection
    del all_line_dicts
    gc.collect()
    
    if content_df.empty:
        return []
    
    # Merge content with sources
    logger.info("Merging content with sources...")
    merged_df = pd.merge(content_df, sources_df, on='id')
    logger.info(f"Merged {len(merged_df)} records")
    
    # Help garbage collection
    del content_df
    gc.collect()
    
    # Sort by sequence to maintain original order
    merged_df = merged_df.sort_values('sequence')
    
    # Format for validation
    logger.info("Formatting for validation...")
    result = format_for_validation(merged_df)
    
    # Final cleanup
    del merged_df
    gc.collect()
    
    return result

@cython.boundscheck(False)
@cython.wraparound(False)
def export_to_parquet(list data, str output_file):
    """Export data to a Parquet file."""
    cdef:
        int chunk_size = 10000
        int total_chunks, i, chunk_num
        list temp_files = []
        list chunk
        object df_chunk, existing_df, combined_df
        str temp_file, excel_file
    
    # Process in chunks to avoid memory issues with large datasets
    total_chunks = (len(data) + chunk_size - 1) // chunk_size
    
    if total_chunks > 1:
        logger.info(f"Exporting data in {total_chunks} chunks...")
        
        # Create temporary CSV files for each chunk
        for i in range(0, len(data), chunk_size):
            chunk_num = i // chunk_size + 1
            chunk = data[i:i+chunk_size]
            
            logger.info(f"Processing export chunk {chunk_num}/{total_chunks}")
            df_chunk = pd.DataFrame(chunk)
            
            temp_file = f"temp_export_{chunk_num}.csv"
            df_chunk.to_csv(temp_file, index=False)
            temp_files.append(temp_file)
            
            # Help garbage collection
            del df_chunk
            gc.collect()
        
        # Combine all temporary CSV files and export to Parquet
        logger.info("Combining chunks and exporting to Parquet...")
        
        # Read and combine CSVs in chunks
        with pd.ExcelWriter(output_file.replace('.parquet', '.xlsx'), engine='openpyxl') as writer:
            for i, temp_file in enumerate(temp_files):
                df_chunk = pd.read_csv(temp_file)
                
                if i == 0:
                    df_chunk.to_parquet(output_file, index=False)
                else:
                    # Read existing parquet, append new data, write back
                    existing_df = pd.read_parquet(output_file)
                    combined_df = pd.concat([existing_df, df_chunk], ignore_index=True)
                    combined_df.to_parquet(output_file, index=False)
                
                # Also write to Excel (optional, since Parquet is more efficient)
                df_chunk.to_excel(writer, sheet_name=f"Chunk_{i+1}", index=False)
                
                # Delete temp file
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Could not remove temporary file {temp_file}: {e}")
                
                # Help garbage collection
                del df_chunk
                gc.collect()
    else:
        # Small enough to process in one go
        df_output = pd.DataFrame(data)
        df_output.to_parquet(output_file, index=False)
        
        # Also save as Excel for easier viewing (optional)
        excel_file = output_file.replace('.parquet', '.xlsx')
        df_output.to_excel(excel_file, index=False)
        logger.info(f"Also exported to Excel: {excel_file}")
    
    logger.info(f"Exported {len(data)} records to {output_file}")

def main():
    """Main entry point."""
    cdef:
        str platform_type, input_folder, sources_txt, output_parquet_file
        double available_memory
        int num_workers
        list processed_data
        object df_preview
    
    # Configure multiprocessing based on platform
    configure_multiprocessing()
    
    # Log platform information
    platform_type = get_platform_info()
    logger.info(f"Running on platform: {platform_type}")
    
    # Define input folder and sources file
    input_folder = 'corpora'
    sources_txt = 'sources.txt'
    output_parquet_file = 'output.parquet'
    
    # Determine optimal number of workers based on platform and system resources
    available_memory = get_available_memory()
    
    # Base worker count on platform and available memory
    if platform_type == "WSL":
        # WSL is often memory-constrained
        num_workers = max(1, min(2, int(available_memory / 2)))
    elif platform_type == "Windows":
        # Windows has process creation overhead
        num_workers = max(1, min(4, int(available_memory / 1.5)))
    else:
        # Linux and Mac can use more workers
        num_workers = max(1, min(cpu_count() - 1, int(available_memory / 1)))
    
    logger.info(f"Using {num_workers} worker processes (based on {available_memory:.1f}GB available memory)")
    
    # Process data with memory-optimized processing
    try:
        processed_data = process_data(input_folder, sources_txt, num_workers)
        
        # Export processed data
        if processed_data:
            export_to_parquet(processed_data, output_parquet_file)
            
            # Preview the output (read just the first few rows to be memory efficient)
            try:
                df_preview = pd.read_parquet(output_parquet_file).head(5)
                logger.info(f"Successfully processed {len(processed_data)} entries")
                
                # Optional: Preview first record
                if not df_preview.empty:
                    logger.info("First record sample:")
                    print(df_preview)
            except Exception as e:
                logger.warning(f"Could not preview output: {e}")
        else:
            logger.error("No data to export")
    except Exception as e:
        logger.error(f"Error in main processing: {e}")

if __name__ == "__main__":
    main()