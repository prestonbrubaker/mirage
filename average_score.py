import os

def calculate_average_scores(input_file, output_file):
    scores = {}  # Dictionary to hold total scores
    appearances = {}  # Dictionary to hold total appearances
    
    # Read through each line in the file to calculate scores and appearances
    with open(input_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            left_photo, right_photo, outcome = parts[0], parts[1], int(parts[2])
            
            # Increment appearances
            appearances[left_photo] = appearances.get(left_photo, 0) + 1
            appearances[right_photo] = appearances.get(right_photo, 0) + 1
            
            # Increment score based on outcome
            if outcome == 0:
                scores[left_photo] = scores.get(left_photo, 0) + 1
            else:  # outcome == 1
                scores[right_photo] = scores.get(right_photo, 0) + 1
                
    # Calculate average score for each photo
    average_scores = {photo: score / appearances[photo] for photo, score in scores.items()}
    
    # For any photo that did not score but appeared, it should have an average score of 0
    for photo in appearances:
        if photo not in average_scores:
            average_scores[photo] = 0.0
    
    # Write the average scores to the output file
    with open(output_file, 'w') as file:
        for photo, average_score in sorted(average_scores.items()):
            file.write(f"{photo} {average_score}\n")

# Paths to the input and output files
input_file_path = "ticker_tape.txt"
output_file_path = "average_scores.txt"

# Calculate and write the average scores
calculate_average_scores(input_file_path, output_file_path)

# Indicate the path to the output file for downloading
output_file_path
