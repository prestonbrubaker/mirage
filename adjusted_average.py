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
                scores[left_photo] = {'wins': 0, 'losses': 1, 'appearances': 0}  # Start with an extra loss for adjustment
            if right_photo not in scores:
                scores[right_photo] = {'wins': 0, 'losses': 1, 'appearances': 0}  # Start with an extra loss for adjustment
            
            # Increment wins and losses according to the outcome
            if outcome == '0':
                scores[left_photo]['wins'] += 1
            else:
                scores[right_photo]['wins'] += 1
            
            # Increment appearances
            scores[left_photo]['appearances'] += 1
            scores[right_photo]['appearances'] += 1
            
            # Adjust losses according to appearances
            scores[left_photo]['losses'] += 1
            scores[right_photo]['losses'] += 1

    # Calculate adjusted average score for each photo
    for photo in scores:
        # Now considering the initial extra loss, adjust the score calculation
        total_scores = scores[photo]['wins'] + scores[photo]['losses'] - scores[photo]['appearances']  # Subtract appearances since we added an extra loss initially
        adjusted_score = scores[photo]['wins'] / total_scores if total_scores > 0 else 0
        scores[photo]['adjusted_score'] = adjusted_score

adjust_and_calculate_scores()

# Correctly write the adjusted outcomes to "scores.txt"
with open("scores.txt", "w") as outfile:
    for photo, data in sorted(scores.items(), key=lambda x: x[1]['adjusted_score'], reverse=True):
        # Ensure adjusted scores are properly calculated with the added loss
        adjusted_score = (data['wins'] + 1) / (data['appearances'] + 1)  # +1 for the additional loss occurrence
        outfile.write(f"{photo} {adjusted_score:.3f}\n")

print("Scores have been calculated and written to scores.txt.")
