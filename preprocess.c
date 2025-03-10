#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/stat.h>
#include <sys/types.h>

// Function to create directory if it doesn't exist
void create_directory(const char *path) {
    #ifdef _WIN32
    mkdir(path);
    #else
    mkdir(path, 0755); // For Unix-like systems
    #endif
}

// Function to process text with enhanced cleanup:
// 1. Remove HTML tags
// 2. Remove sequences of @ characters and any spaces around them
// 3. Normalize spacing around periods and commas
// 4. Remove dashes with spaces (-- -- -- patterns)
void process_text(char *input, char *output, size_t output_size) {
    int in_tag = 0;
    size_t output_index = 0;
    int consecutive_at_signs = 0;
    int consecutive_dashes = 0;
    int skip_next_spaces = 0;
    
    for (size_t i = 0; input[i] != '\0' && output_index < output_size - 2; i++) {
        // Check for HTML tags
        if (input[i] == '<') {
            in_tag = 1;
            continue;
        }
        
        if (input[i] == '>') {
            in_tag = 0;
            continue;
        }
        
        if (!in_tag) {
            // Handle @ sequences and spaces around them
            if (input[i] == '@') {
                consecutive_at_signs++;
                skip_next_spaces = 1; // Skip spaces that follow @ signs
                continue;
            } else if (consecutive_at_signs > 0 && isspace(input[i])) {
                // Skip spaces after @ sequences
                consecutive_at_signs = 0;
                continue;
            } else {
                consecutive_at_signs = 0;
                skip_next_spaces = 0;
            }
            
            // Handle dash sequences with spaces (-- -- --)
            if (input[i] == '-') {
                consecutive_dashes++;
                // Skip this dash
                continue;
            } else if (consecutive_dashes > 0 && isspace(input[i])) {
                // Skip spaces after dash sequences
                continue;
            } else {
                consecutive_dashes = 0;
            }
            
            // Handle spaces around punctuation
            if (input[i] == ' ') {
                // Check if next non-space character is a period or comma
                size_t look_ahead = i + 1;
                while (input[look_ahead] == ' ' && input[look_ahead] != '\0') {
                    look_ahead++;
                }
                
                if ((input[look_ahead] == '.' || input[look_ahead] == ',') && look_ahead > i) {
                    // Space(s) before period or comma - skip them
                    continue;
                }
                
                // Check if we just processed a period or comma and are now seeing spaces
                if (output_index > 0 && (output[output_index-1] == '.' || output[output_index-1] == ',')) {
                    // If it's a comma, skip the space after it
                    if (output[output_index-1] == ',') {
                        continue;
                    }
                }
                
                // Skip multiple consecutive spaces, keep just one
                if (output_index > 0 && output[output_index-1] == ' ') {
                    continue;
                }
            }
            
            // Normal processing for other characters
            output[output_index++] = input[i];
        }
    }
    
    output[output_index] = '\0';
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }
    
    FILE *input_file = fopen(argv[1], "r");
    if (input_file == NULL) {
        fprintf(stderr, "Error: Cannot open input file %s\n", argv[1]);
        return 1;
    }
    
    FILE *output_file = fopen(argv[2], "w");
    if (output_file == NULL) {
        fprintf(stderr, "Error: Cannot create output file %s\n", argv[2]);
        fclose(input_file);
        return 1;
    }
    
    char buffer[4096];
    char processed_buffer[8192];
    
    // Process the entire file
    while (fgets(buffer, sizeof(buffer), input_file) != NULL) {
        process_text(buffer, processed_buffer, sizeof(processed_buffer));
        fprintf(output_file, "%s", processed_buffer);
    }
    
    fclose(input_file);
    fclose(output_file);
    printf("Processing complete. Output saved to %s\n", argv[2]);
    
    return 0;
}