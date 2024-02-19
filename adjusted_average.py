import os

# Define a dictionary to hold scores and occurrences
scores = {}

def adjust_and_calculate_scores():
    with open("ticker_tape.txt", "r") as file:
        for line in file:
            parts = line.strip().split()
            left_photo, right_photo, outcome = parts[0], parts[1], parts[2]
            
            # Initialize entries in scores dictionary if not already present
            if left_photo not in scores:
                scores[left_photo] = {'wins': 0, 'losses': 0, 'appearances': 0}
            if right_photo not in scores:
                scores[right_photo] = {'wins': 0, 'losses': 0, 'appearances': 0}
            
            # Increment wins and losses
            if outcome == '0':
                scores[left_photo]['wins'] += 1
                scores[right_photo]['losses'] += 1
            else:
                scores[right_photo]['wins'] += 1
                scores[left_photo]['losses'] += 1
            
            # Increment appearances
            scores[left_photo]['appearances'] += 1
            scores[right_photo]['appearances'] += 1

    # Calculate adjusted average score for each photo
    for photo in scores:
        total_scores = scores[photo]['wins'] + scores[photo]['losses']
        adjusted_score = scores[photo]['wins'] / total_scores
        scores[photo]['adjusted_score'] = adjusted_score

adjust_and_calculate_scores()

# Write the outcomes to "scores.txt"
with open("scores.txt", "w") as outfile:
    for photo, data in sorted(scores.items(), key=lambda x: x[1]['adjusted_score'], reverse=True):
        outfile.write(f"{photo} {data['adjusted_score']:.3f}\n")

print("Scores have been calculated and written to scores.txt.")
