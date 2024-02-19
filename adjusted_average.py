import os

# Define a dictionary to hold scores and occurrences
scores = {}

def adjust_and_calculate_scores():
    with open("ticker_tape.txt", "r") as file:
        for line in file:
            parts = line.strip().split()
            left_photo, right_photo, outcome = parts[0], parts[1], parts[2]
            
            # Ensure each photo is initialized in the dictionary
            if left_photo not in scores:
                scores[left_photo] = {'wins': 0, 'losses': 0, 'appearances': 0}
            if right_photo not in scores:
                scores[right_photo] = {'wins': 0, 'losses': 0, 'appearances': 0}
            
            # Increment wins for the winner and losses for the loser
            if outcome == '0':
                scores[left_photo]['wins'] += 1
                scores[right_photo]['losses'] += 1
            else:
                scores[right_photo]['wins'] += 1
                scores[left_photo]['losses'] += 1
            
            # Increment appearances for both photos
            scores[left_photo]['appearances'] += 1
            scores[right_photo]['appearances'] += 1

    # Adjust scores and calculate the adjusted average score
    for photo in scores:
        # Add a free win and adjust occurrences
        scores[photo]['wins'] += 1
        adjusted_occurrences = scores[photo]['appearances'] + 2  # Add 2 to adjust occurrences
        # Calculate the adjusted score
        adjusted_score = scores[photo]['wins'] / adjusted_occurrences
        scores[photo]['adjusted_score'] = adjusted_score

adjust_and_calculate_scores()

# Write the adjusted scores to "scores.txt"
with open("scores.txt", "w") as outfile:
    for photo, data in sorted(scores.items(), key=lambda x: x[1]['adjusted_score'], reverse=True):
        outfile.write(f"{photo} {data['adjusted_score']:.3f}\n")

print("Adjusted scores have been calculated and written to scores.txt.")
