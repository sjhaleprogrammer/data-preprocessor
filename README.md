# Text Corpus Processor

A memory-optimized tool for processing and analyzing large text corpora with parallel processing capabilities.

## Description

This tool processes text files from a corpus, extracts content lines that begin with "@@", matches them with source information, and outputs the processed data in both Parquet and Excel formats. The implementation is optimized for performance and memory efficiency through Cython compilation and intelligent resource management.

## Features

- Parallel processing with automatic worker count optimization
- Memory-optimized batch processing for large datasets
- Platform-specific optimizations (Windows, WSL, macOS, Linux)
- Robust file encoding detection
- Automatic fallback to sequential processing when needed
- Export to both Parquet and Excel formats
- Comprehensive logging

## Requirements

- Python 3.6+
- Cython
- C compiler (recommended gcc)

## Dependencies

```
numpy
pandas
beautifulsoup4
pyarrow
cython
psutil
openpyxl
setuptools
```

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/text-corpus-processor.git
   cd text-corpus-processor
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Compile the Cython extensions:
   ```
   python setup.py build_ext --inplace
   ```

## Usage

1. Organize your data:
   - Place your text corpus files in a directory named `corpora`
   - Create a `sources.txt` file with metadata about your sources

2. Run the processor:
   ```
   python main.py
   ```

3. The processed data will be saved to:
   - `output.parquet` (optimized for further processing)
   - `output.xlsx` (for easy viewing and sharing)

## Input Format

### Text Files
The program scans all `.txt` files in the `corpora` directory and its subdirectories, looking for lines that start with `@@`. Each matching line should follow this format:
```
@@id content
```

### Sources File
The `sources.txt` file should contain metadata about your sources, with each line following this format:
```
id year type pages source title
```

## Performance Tuning

The program automatically optimizes resource usage based on:
- Available system memory
- Number of CPU cores
- Operating system platform
- Dataset size

No manual configuration is typically needed, but you can modify the code to adjust:
- Batch sizes
- Worker counts
- Memory usage limits

## Troubleshooting

If you encounter memory errors:
- Ensure you have enough available RAM (at least 4GB recommended)
- The program will automatically fall back to sequential processing if parallel processing fails
- For very large datasets, consider processing in smaller chunks by manually splitting the input

## License

MIT License
