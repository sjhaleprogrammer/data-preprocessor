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
    df_input['from'] = 'human'
    df_input['content'] = df_input['content'].apply(clean_text)

    # Load sources data
    sources_data = pd.read_csv(sources_file, sep='\s+', header=None, names=['id', 'year', 'magazine', 'pages', 'title'], engine='python')

    # Merge based on ID match (assuming IDs are integers or can be cast to integer)
    df_sources = sources_data.astype({'id': str})
    merged_df = pd.merge(df_input.astype({'id': str}), df_sources, on='id')

    # Prepare output structure
    result_list = []
    for index, row in merged_df.iterrows():
        result_dict = {
            "from": row['from'],
            "value": f"***{row['content']}*** - Source: {row['title']}"
        }
        result_list.append(result_dict)

    return result_list