import os
import pandas as pd
import logging
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional, Tuple
import re
import functools

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# Define constants
ENCODINGS = ['utf-8', 'latin-1', 'cp1252']

# Precompile regular expressions for performance
HTML_TAG_PATTERN = re.compile(r'<[^>]+>')
URL_PATTERN = re.compile(r'https?://[^\s<>"]+|www\.[^\s<>"]+\.[^\s<>"]+(?:/[^\s<>"]*)?')
DOMAIN_PATTERN = re.compile(r'(?:https?://)?(?:www\.)?([^/\s<>"]+\.[^/\s<>"]+)')
CITATION_PATTERN = re.compile(r'\*\*\d+;\d+;TOOLONG')
REFERENCE_PATTERN = re.compile(r'\[\d+\]')
POST_NUMBER_PATTERN = re.compile(r'#\d+\s*')
TIMESTAMP_PATTERN = re.compile(r'\d+\s+(?:days?|hours?|minutes?|seconds?)\s+ago')
ATTRIBUTION_PATTERN = re.compile(r'(?:Quote|Originally Posted by).*?(?=<p>|\n|$)')
SIGNATURE_PATTERN = re.compile(r'(?:^|\n)(?:--|Regards,|Last edited by)[\s\w]+?(?=\n|$)')
USERNAME_PATTERN = re.compile(r'^\s*[A-Za-z0-9_]+\s*$', flags=re.MULTILINE)
WHITESPACE_PATTERN = re.compile(r'\s+')

# Define contraction patterns
CONTRACTION_PATTERNS = [
    (re.compile(r'\s+n\'t'), "n't"),
    (re.compile(r'\s+\'s'), "'s"),
    (re.compile(r'\s+\'m'), "'m"),
    (re.compile(r'\s+\'re'), "'re"),
    (re.compile(r'\s+\'ve'), "'ve"),
    (re.compile(r'\s+\'ll'), "'ll"),
    (re.compile(r'\s+\'d'), "'d"),
]

# Define punctuation patterns
PUNCTUATION_PATTERNS = [
    (re.compile(r'\s+\,'), ","),
    (re.compile(r'\s+\.'), "."),
    (re.compile(r'\s+\:'), ":"),
    (re.compile(r'\s+\;'), ";"),
    (re.compile(r'\s+\?'), "?"),
    (re.compile(r'\s+\!'), "!"),
    (re.compile(r'\s+\)'), ")"),
    (re.compile(r'\(\s+'), "("),
]

# HTML entities mapping
HTML_ENTITIES = {
    '&lt;': '<',
    '&gt;': '>',
    '&amp;': '&',
    '&quot;': '"',
    '&apos;': "'",
    '&nbsp;': ' ',
}

# Use lru_cache for functions that might be called with the same inputs
@functools.lru_cache(maxsize=1024)
def fix_encoded_html(text):
    """
    Fix encoded HTML entities and remove HTML tags.
    
    Args:
        text: The text to clean
        
    Returns:
        Text with decoded and removed HTML
    """
    # Skip if the text doesn't contain any encoded HTML entities
    if not ('&lt;' in text or '&gt;' in text or '&amp;' in text):
        return text
        
    # Replace HTML entities with their actual characters
    for entity, char in HTML_ENTITIES.items():
        text = text.replace(entity, char)
    
    # Now use BeautifulSoup to remove the HTML tags
    return clean_html(text)

def clean_html(text):
    """
    Remove HTML tags using BeautifulSoup or regex fallback.
    
    Args:
        text: The text containing HTML tags
    
    Returns:
        Text with HTML tags removed
    """
    # Skip processing if the text looks like a filename or path
    if len(text) < 255 and ('\\' in text or '/' in text or '.' in text) and ' ' not in text:
        return text
        
    # Check if there are any HTML-like tags in the text before using BeautifulSoup
    if '<' in text and '>' in text:
        try:
            # Use BeautifulSoup to parse and remove HTML tags
            soup = BeautifulSoup(text, 'html.parser')
            return soup.get_text(' ', strip=True)
        except Exception as e:
            logger.warning(f"BeautifulSoup error: {str(e)}")
            # Fall back to a simple regex-based approach
            return HTML_TAG_PATTERN.sub('', text)
    else:
        # If no HTML-like content, return as is
        return text

def fix_contractions(text):
    """Fix separated contractions like 'was n't' to 'wasn't'."""
    for pattern, replacement in CONTRACTION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text

def fix_punctuation(text):
    """Fix separated punctuations like 'WORD ,' to 'WORD,'."""
    for pattern, replacement in PUNCTUATION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text

def handle_numerical_citations(text):
    """Handle numerical citations and special reference patterns."""
    # Replace citation patterns
    text = CITATION_PATTERN.sub('[citation]', text)
    text = REFERENCE_PATTERN.sub('[reference]', text)
    return text

def handle_forum_content(text):
    """Remove forum/blog-specific elements."""
    text = POST_NUMBER_PATTERN.sub('', text)
    text = TIMESTAMP_PATTERN.sub('', text)
    text = ATTRIBUTION_PATTERN.sub('', text)
    text = SIGNATURE_PATTERN.sub('', text)
    text = USERNAME_PATTERN.sub('', text)
    return text

def truncate_urls(text):
    """Truncate long URLs to just the domain name."""
    def replace_url(match):
        url = match.group(0)
        domain_match = DOMAIN_PATTERN.search(url)
        if domain_match:
            domain = domain_match.group(1)
            return f"www.{domain}"
        return url
    
    return URL_PATTERN.sub(replace_url, text)

def clean_text(text):
    """
    Clean text by applying all cleaning functions in sequence.
    
    Args:
        text: The text to clean
    
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        logger.warning(f"Non-string input to clean_text: {type(text)}")
        return str(text)
        
    try:
        # Replace Unicode replacement character
        text = text.replace('\ufffd', '<unk>')
        
        # Apply cleaning steps in a logical sequence
        text = fix_encoded_html(text)
        text = clean_html(text)
        text = handle_numerical_citations(text)
        text = handle_forum_content(text)
        text = fix_contractions(text)
        text = fix_punctuation(text)
        text = truncate_urls(text)
        
        # Normalize whitespace
        text = text.strip()
        text = WHITESPACE_PATTERN.sub(' ', text)
        
        return text
    except Exception as e:
        logger.error(f"Error during text cleaning: {str(e)}")
        # Fallback: return ASCII characters only
        return ''.join(c if c.isascii() else '<unk>' for c in str(text))

def read_file_with_multiple_encodings(file_path: str, encodings: List[str] = ENCODINGS) -> Tuple[Optional[List[str]], str]:
    """Try to read a file with multiple encodings."""
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
    """Parse the sources file into a DataFrame."""
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
    """Walk through directories and collect lines starting with '@@'."""
    all_lines = []
    processed_files = 0
    skipped_files = 0
    total_files = 0
    sequence_counter = 0
    
    # Process files in sorted order for consistency
    for root, dirs, files in sorted(os.walk(input_folder)):
        txt_files = sorted([f for f in files if f.endswith('.txt')])
        total_files += len(txt_files)
        logger.info(f"Found {len(txt_files)} .txt files in {root}")
        
        for filename in txt_files:
            file_path = os.path.join(root, filename)
            logger.info(f"Processing {file_path}...")
            
            content, encoding = read_file_with_multiple_encodings(file_path)
            if content:
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
    """Parse content lines into a DataFrame."""
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
    
    # Create DataFrame first, then apply cleaning function using pandas
    df = pd.DataFrame(rows)
    logger.info("Cleaning text content...")
    df['content'] = df['content'].apply(clean_text)
    df = df.astype({'id': str})
    
    logger.info(f"Parsed {len(df)} valid content lines")
    return df

def format_for_validation(merged_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Format merged data for validation structure."""
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
    """Main data processing function."""
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
        
       
    else:
        logger.error("No data to export")

if __name__ == "__main__":
    main()