import pandas as pd
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

def process_data(input_file, sources_file):
    # Load data from input corpus file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.startswith('@@')]

    # Create DataFrame from lines
    df_input = pd.DataFrame([line[2:].split(' ', 1) for line in lines], columns=['id', 'content'])
    df_input['content'] = df_input['content'].apply(clean_text)

    # Load data from sources file
    with open(sources_file, 'r') as f:
        sources_data = []
        for line in f:
            parts = line.strip().split()
            # Extract the first 5 fields
            if len(parts) >= 6:
                record = {
                    'id': parts[0],
                    'year': parts[1],
                    'type': parts[2],
                    'pages': parts[3],
                    'source': parts[4],
                    'title': ' '.join(parts[5:])  # Join remaining parts as title
                }
                sources_data.append(record)
        sources_data = pd.DataFrame(sources_data)

    # Merge based on ID match (assuming IDs are integers or can be cast to integer)
    df_sources = sources_data.astype({'id': str})
    merged_df = pd.merge(df_input.astype({'id': str}), df_sources, on='id')

    # Prepare output structure
    result_list = []
    for index, row in merged_df.iterrows():
        # Human entry
        human_dict = {
            "from": "human",
            "value": f"{row['content']}"
        }
        result_list.append(human_dict)

        # GPT entry
        gpt_dict = {
            "from": "gpt",
            "value": "Tell the user what dialect this is and provide additional context."
        }
        result_list.append(gpt_dict)

    return result_list


def export_to_parquet(data, output_file):
    # Convert list of dictionaries to DataFrame
    df_output = pd.DataFrame(data)

    # Save to Parquet
    df_output.to_parquet(output_file, index=False)


if __name__ == "__main__":

    # Define input files
    input_corpus = 'text_blog_01.txt'
    sources_txt = 'sources.txt'

    # Process data
    processed_data = process_data(input_corpus, sources_txt)

    #print(processed_data[0])
    #print(processed_data[1])

    # Export processed data to Parquet file
    output_parquet_file = 'output.parquet'
    export_to_parquet(processed_data, output_parquet_file)
    
    df = pd.read_parquet('output.parquet')

    df.head(10)