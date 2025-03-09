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
// 2. Add newline after single periods (not ellipses)
// 3. Remove sequences of @ characters and any spaces around them
// 4. Normalize spacing around periods and commas
// 5. Remove dashes with spaces (-- -- -- patterns)
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
                    // If it's a period and we're adding a newline, skip this space
                    if (output[output_index-1] == '.' && 
                        (input[i-1] == '.' && (input[i-2] != '.' && input[i] != '.'))) {
                        continue;
                    }
                    
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
            
            // Check if this is a single period, not part of an ellipsis
            if (input[i] == '.' && 
                (input[i+1] != '.' && input[i-1] != '.')) {
                
                // Check if the next character is already a newline
                if (input[i+1] != '\n' && input[i+1] != '\r') {
                    output[output_index++] = '\n';
                }
            }
        }
    }
    
    output[output_index] = '\0';
}

// Function to process a file and split text by @@NUMBER pattern into separate files
void split_by_delimiter(FILE *input_file, const char *output_dir) {
    char buffer[4096];  // Increased buffer size for longer lines
    char processed_buffer[8192];  // Double size to accommodate added newlines
    FILE *current_output = NULL;
    char current_filename[256] = "";
    
    // Read the file line by line
    while (fgets(buffer, sizeof(buffer), input_file) != NULL) {
        // Check if the line contains the delimiter pattern
        char *delimiter = strstr(buffer, "@@");
        
        if (delimiter != NULL && isdigit(*(delimiter + 2))) {
            // Close previous output file if open
            if (current_output != NULL) {
                fclose(current_output);
                current_output = NULL;
            }
            
            // Extract the number following @@
            char *number_start = delimiter + 2;
            char *number_end = number_start;
            
            while (isdigit(*number_end)) {
                number_end++;
            }
            
            // Create a temporary string to hold just the number
            size_t number_len = number_end - number_start;
            char number[256];
            if (number_len >= sizeof(number)) {
                number_len = sizeof(number) - 1;
            }
            strncpy(number, number_start, number_len);
            number[number_len] = '\0';
            
            // Create the new filename
            snprintf(current_filename, sizeof(current_filename), 
                     "%s/%s.txt", output_dir, number);
            
            // Open the new output file
            current_output = fopen(current_filename, "w");
            if (current_output == NULL) {
                fprintf(stderr, "Error: Cannot create file %s\n", current_filename);
                return;
            }
            
            printf("Creating file: %s\n", current_filename);
            
            // Process and write the content after the delimiter to the file
            char *content_after_delimiter = number_end;
            process_text(content_after_delimiter, processed_buffer, sizeof(processed_buffer));
            fprintf(current_output, "%s", processed_buffer);
        } else if (current_output != NULL) {
            // Process the line and write to the current output file
            process_text(buffer, processed_buffer, sizeof(processed_buffer));
            fprintf(current_output, "%s", processed_buffer);
        }
    }
    
    // Close the last output file if open
    if (current_output != NULL) {
        fclose(current_output);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return 1;
    }
    
    // Create the output directory
    const char *output_dir = "processed";
    create_directory(output_dir);
    
    FILE *input_file = fopen(argv[1], "r");
    if (input_file == NULL) {
        fprintf(stderr, "Error: Cannot open input file %s\n", argv[1]);
        return 1;
    }
    
    split_by_delimiter(input_file, output_dir);
    
    fclose(input_file);
    printf("Processing complete. Files are in the '%s' directory.\n", output_dir);
    
    return 0;
}