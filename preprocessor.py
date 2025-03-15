import os
import pandas as pd
import random
from bs4 import BeautifulSoup

def clean_text(text):
    # Remove multiple '@' characters
    text = text.replace('@@', '')
    # Replace HTML entities with spaces
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    # Remove unnecessary spaces around 's and n't
    text = text.replace(" '", "'").replace(" n't", "n't")
    return text

def process_data(input_folder, sources_file):
    # Load data from sources file with error handling for encoding issues
    sources_data = []
    try:
        # Try UTF-8 first
        with open(sources_file, 'r', encoding='utf-8') as f:
            for line in f:
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
    except UnicodeDecodeError:
        # If UTF-8 fails, try latin-1 (which can read any byte sequence)
        with open(sources_file, 'r', encoding='latin-1') as f:
            for line in f:
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
    
    sources_df = pd.DataFrame(sources_data)
    if sources_df.empty:
        print("No valid data found in sources file")
        return []
    
    sources_df = sources_df.astype({'id': str})
    
    # Initialize empty list for all lines
    all_lines = []
    
    # Walk through all subdirectories and process each .txt file
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith('.txt'):
                file_path = os.path.join(root, filename)
                # Try different encodings for each file
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            lines = [line.strip() for line in f if line.strip().startswith('@@')]
                            all_lines.extend(lines)
                        # If successful, break out of the encoding loop
                        break
                    except UnicodeDecodeError:
                        # Try the next encoding
                        continue
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")
                        break
    
    # Create DataFrame from all lines
    if not all_lines:
        print("No valid data found in any files")
        return []
    
    valid_lines = []
    for line in all_lines:
        parts = line[2:].split(' ', 1)
        if len(parts) == 2:
            valid_lines.append(parts)
        else:
            print(f"Skipping invalid line: {line}")
    
    if not valid_lines:
        print("No valid data lines after parsing")
        return []
    
    df_input = pd.DataFrame(valid_lines, columns=['id', 'content'])
    df_input['content'] = df_input['content'].apply(clean_text)
    
    # Merge based on ID match
    merged_df = pd.merge(df_input.astype({'id': str}), sources_df, on='id')
    
    # Format data to match validation structure
    validation_format = []
    for index, row in merged_df.iterrows():
        conversation = [
            {
                "from": "human",
                "value": f"{row['content']}"
            },
            {
                "from": "gpt",
                "value": "Tell the user what dialect this is and provide additional context."
            }
        ]
        
        # Extract source from the sources data
        source = row['source']
        
        # Generate a random score between 4.5 and 5.5 to match validation data
        score = random.uniform(4.5, 5.5)
        
        validation_format.append({
            "conversations": conversation,
            "source": source,
            "score": score
        })
    
    return validation_format

def export_to_parquet(data, output_file):
    # Convert list of dictionaries to DataFrame
    df_output = pd.DataFrame(data)
    
    # Save to Parquet
    df_output.to_parquet(output_file, index=False)

if __name__ == "__main__":
    # Define input folder and sources file
    input_folder = 'corpora'  # Main folder containing potentially multiple subfolders with .txt files
    sources_txt = 'sources.txt'
    
    # Process data
    processed_data = process_data(input_folder, sources_txt)
    
    # Export processed data to Parquet file if we have data
    if processed_data:
        output_parquet_file = 'output.parquet'
        export_to_parquet(processed_data, output_parquet_file)
        
        # Preview the output
        df = pd.read_parquet('output.parquet')
        print(f"Successfully processed {len(df)} entries")
        print(df.head(10), '\n')

        print(f"Validation:")
        df = pd.read_parquet('validation.parquet')
        
        print(df.head(10))
    else:
        print("No data to export")