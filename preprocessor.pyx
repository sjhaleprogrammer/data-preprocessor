# File: text_processor.pyx

import re
import pandas as pd
import logging
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional, Tuple
import os
import functools
import multiprocessing
from multiprocessing import Pool, cpu_count

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Define constants
ENCODINGS = ['utf-8', 'latin-1', 'cp1252']




# Public function that users will call
def clean_text(object text_obj):
    """
    Clean text by applying various text cleaning operations.
    This is the public API that calls the cached implementation.
    
    Args:
        text_obj: The text to clean
        
    Returns:
        Cleaned text
    """
    return text_obj





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

def parse_sources_file(str sources_file):
    """Parse the sources file into a DataFrame."""
    cdef:
        list lines
        str line
        list parts
        dict record
        list sources_data = []
    
    lines, _ = read_file_with_multiple_encodings(sources_file)
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
            return True, result_lines
        else:
            logger.warning(f"  ! Could not process {file_path}")
            return False, []
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return False, []

def collect_content_lines_parallel(str input_folder, int num_workers=0):
    """Walk through directories and collect lines starting with '@@' using parallel processing."""
    if num_workers is 0:
        num_workers = max(1, cpu_count() - 1)  # Leave one CPU for system
    
    logger.info(f"Using {num_workers} worker processes")
    
    # Get all text files and prepare for parallel processing
    cdef:
        list file_infos = []
        int file_order = 0
        int total_files = 0
        str root
    
    for root, dirs, files in sorted(os.walk(input_folder)):
        txt_files = sorted([f for f in files if f.endswith('.txt')])
        total_files += len(txt_files)
        
        for filename in txt_files:
            # Root dir, filename, file order, starting sequence number
            file_infos.append((root, filename, file_order, file_order * 10000))
            file_order += 1
    
    logger.info(f"Found {total_files} .txt files to process")
    
    # Process files in parallel
    all_lines = []
    processed_files = 0
    skipped_files = 0
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_file, file_infos)
        
        for success, lines in results:
            if success:
                all_lines.extend(lines)
                processed_files += 1
            else:
                skipped_files += 1
    
    # Sort lines by sequence to ensure correct order
    all_lines.sort(key=lambda x: x['sequence'])
    
    logger.info(f"- Processed {processed_files}/{total_files} files")
    logger.info(f"- Skipped {skipped_files} files")
    logger.info(f"- Total lines collected: {len(all_lines)}")
    
    return all_lines

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
    
    return rows

def parse_content_lines_parallel(list line_dicts, int batch_size=1000, int num_workers=0):
    """Parse content lines into a DataFrame using parallel processing."""
    if not line_dicts:
        logger.warning("No content lines to parse")
        return pd.DataFrame()
    
    if num_workers is 0:
        num_workers = max(1, cpu_count() - 1)  # Leave one CPU for system
    
    # Split lines into batches for parallel processing
    batches = [line_dicts[i:i+batch_size] for i in range(0, len(line_dicts), batch_size)]
    logger.info(f"Processing {len(batches)} batches with {num_workers} workers")
    
    # Process batches in parallel
    all_rows = []
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_content_batch, batches)
        for batch_rows in results:
            all_rows.extend(batch_rows)
    
    if not all_rows:
        logger.warning("No valid data lines after parsing")
        return pd.DataFrame()
    
    # Create DataFrame from processed rows
    df = pd.DataFrame(all_rows)
    df = df.astype({'id': str})
    
    logger.info(f"Parsed {len(df)} valid content lines")
    return df

def format_for_validation(object merged_df):
    """Format merged data for validation structure."""
    cdef:
        list validation_format = []
        list conversation
    
    for _, row in merged_df.iterrows():
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
            "metadata": {
                "file": row['file'],
                "sequence": row['sequence']
            }
        })
    
    return validation_format

def process_data(str input_folder, str sources_file, int num_workers=0):
    """Main data processing function with parallel processing."""
    # Parse sources file
    sources_df = parse_sources_file(sources_file)
    if sources_df.empty:
        return []
    
    # Collect content lines with sequence information using parallel processing
    all_line_dicts = collect_content_lines_parallel(input_folder, num_workers)
    if not all_line_dicts:
        return []
    
    # Parse content lines using parallel processing
    content_df = parse_content_lines_parallel(all_line_dicts, num_workers=num_workers)
    if content_df.empty:
        return []
    
    # Merge content with sources
    merged_df = pd.merge(content_df, sources_df, on='id')
    logger.info(f"Merged {len(merged_df)} records")
    
    # Sort by sequence to maintain original order
    merged_df = merged_df.sort_values('sequence')
    
    # Format for validation
    return format_for_validation(merged_df)

def export_to_parquet(list data, str output_file):
    """Export data to a Parquet file."""
    df_output = pd.DataFrame(data)
    df_output.to_parquet(output_file, index=False)
    logger.info(f"Exported {len(df_output)} records to {output_file}")

def main():
    """Main entry point."""
    # Define input folder and sources file
    input_folder = 'corpora'
    sources_txt = 'sources.txt'
    output_parquet_file = 'output.parquet'
    
    # Determine optimal number of workers
    num_workers = max(1, cpu_count() - 1)  # Leave one CPU for system
    logger.info(f"Using {num_workers} worker processes")
    
    # Process data with parallel processing
    processed_data = process_data(input_folder, sources_txt, num_workers)
    
    # Export processed data
    if processed_data:
        export_to_parquet(processed_data, output_parquet_file)
        
        # Preview the output
        df = pd.read_parquet(output_parquet_file)
        logger.info(f"Successfully processed {len(df)} entries")
        
        # Optional: Preview first record
        if not df.empty:
            logger.info("First record sample:")
            logger.info(df.iloc[0][0])
    else:
        logger.error("No data to export")

if __name__ == "__main__":
    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)
    main()