import os
import pandas as pd
import logging
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Define constants
ENCODINGS = ['utf-8', 'latin-1', 'cp1252']

def clean_text(text: str) -> str:
    """Clean text by removing markers and fixing formatting issues."""
    # Remove multiple '@' characters
    text = text.replace('@@', '')
    # Replace HTML entities with spaces
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    # Remove unnecessary spaces around 's and n't
    text = text.replace(" '", "'").replace(" n't", "n't").replace("@ @ @ @ @ @ @ @ @ @","").replace("\\","")
    return text

def read_file_with_multiple_encodings(file_path: str, encodings: List[str] = ENCODINGS) -> Tuple[Optional[List[str]], str]:
    """
    Try to read a file with multiple encodings.
    
    Args:
        file_path: Path to the file to read
        encodings: List of encodings to try
        
    Returns:
        Tuple containing:
            - List of lines from the file or None if all encodings failed
            - Encoding used (if successful)
    """
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.readlines(), encoding
        except UnicodeDecodeError:
            logger.debug(f"Failed to decode {file_path} with {encoding}")
            continue
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None, ""
    
    logger.warning(f"Could not decode {file_path} with any encoding")
    return None, ""

def parse_sources_file(sources_file: str) -> pd.DataFrame:
    """
    Parse the sources file into a DataFrame.
    
    Args:
        sources_file: Path to the sources file
        
    Returns:
        DataFrame containing source information
    """
    lines, _ = read_file_with_multiple_encodings(sources_file)
    if not lines:
        logger.error("Could not read sources file")
        return pd.DataFrame()
    
    sources_data = []
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

def collect_content_lines(input_folder: str) -> List[Dict[str, Any]]:
    """
    Walk through all subdirectories and collect lines starting with '@@'.
    
    Args:
        input_folder: Path to the folder to search
        
    Returns:
        List of dictionaries containing line content and metadata
    """
    all_lines = []
    processed_files = 0
    skipped_files = 0
    total_files = 0
    sequence_counter = 0
    
    # Sort directories and files for consistent processing order
    for root, dirs, files in sorted(os.walk(input_folder)):
        # Sort files for consistent processing
        txt_files = sorted([f for f in files if f.endswith('.txt')])
        total_files += len(txt_files)
        logger.info(f"Found {len(txt_files)} .txt files in {root}")
        
        for filename in txt_files:
            file_path = os.path.join(root, filename)
            logger.info(f"Processing {file_path}...")
            
            content, encoding = read_file_with_multiple_encodings(file_path)
            if content:
                # Process each line while tracking its order
                matching_lines = []
                for line_num, line in enumerate(content):
                    line = line.strip()
                    if line.startswith('@@'):
                        sequence_counter += 1
                        matching_lines.append(line)
                        all_lines.append({
                            'line': line,
                            'file': file_path,
                            'file_order': processed_files,
                            'line_num': line_num,
                            'sequence': sequence_counter
                        })
                
                logger.info(f"  - Found {len(matching_lines)} lines starting with @@ using {encoding} encoding")
                processed_files += 1
            else:
                logger.warning(f"  ! SKIPPED: Could not process {file_path}")
                skipped_files += 1
    
    logger.info(f"- Processed {processed_files}/{total_files} files")
    logger.info(f"- Skipped {skipped_files} files")
    logger.info(f"- Total lines collected: {len(all_lines)}")
    
    return all_lines

def parse_content_lines(line_dicts: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Parse content lines into a DataFrame.
    
    Args:
        line_dicts: List of dictionaries with line content and metadata
        
    Returns:
        DataFrame with id, content, and ordering information
    """
    if not line_dicts:
        logger.warning("No content lines to parse")
        return pd.DataFrame()
    
    rows = []
    for line_dict in line_dicts:
        line = line_dict['line']
        parts = line[2:].split(' ', 1)  # Skip the '@@' prefix
        if len(parts) == 2:
            rows.append({
                'id': parts[0],
                'content': parts[1],
                'file': line_dict['file'],
                'file_order': line_dict['file_order'],
                'line_num': line_dict['line_num'],
                'sequence': line_dict['sequence']
            })
        else:
            logger.warning(f"Skipping invalid line: {line}")
    
    if not rows:
        logger.warning("No valid data lines after parsing")
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    df['content'] = df['content'].apply(clean_text)
    df = df.astype({'id': str})
    
    logger.info(f"Parsed {len(df)} valid content lines")
    return df

def format_for_validation(merged_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Format merged data for validation structure.
    
    Args:
        merged_df: DataFrame with merged content and source information
        
    Returns:
        List of dictionaries in validation format
    """
    validation_format = []
    
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

def process_data(input_folder: str, sources_file: str) -> List[Dict[str, Any]]:
    """
    Main data processing function.
    
    Args:
        input_folder: Path to the input folder
        sources_file: Path to the sources file
        
    Returns:
        List of dictionaries in validation format
    """
    # Parse sources file
    sources_df = parse_sources_file(sources_file)
    if sources_df.empty:
        return []
    
    # Collect content lines with sequence information
    all_line_dicts = collect_content_lines(input_folder)
    if not all_line_dicts:
        return []
    
    # Parse content lines
    content_df = parse_content_lines(all_line_dicts)
    if content_df.empty:
        return []
    
    # Merge content with sources
    merged_df = pd.merge(content_df, sources_df, on='id')
    logger.info(f"Merged {len(merged_df)} records")
    
    # Sort by sequence to maintain original order
    merged_df = merged_df.sort_values('sequence')
    
    # Format for validation
    return format_for_validation(merged_df)

def export_to_parquet(data: List[Dict[str, Any]], output_file: str) -> None:
    """
    Export data to a Parquet file.
    
    Args:
        data: List of dictionaries to export
        output_file: Path to the output file
    """
    df_output = pd.DataFrame(data)
    df_output.to_parquet(output_file, index=False)
    logger.info(f"Exported {len(df_output)} records to {output_file}")

def main():
    """Main entry point."""
    # Define input folder and sources file
    input_folder = 'corpora'
    sources_txt = 'sources.txt'
    output_parquet_file = 'output.parquet'
    
    # Process data
    processed_data = process_data(input_folder, sources_txt)
    
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
        
        # Optional: Compare with validation data if available
        validation_file = 'validation.parquet'
        if os.path.exists(validation_file):
            validation_df = pd.read_parquet(validation_file)
            logger.info(f"Validation data has {len(validation_df)} records")
    else:
        logger.error("No data to export")

if __name__ == "__main__":
    main()