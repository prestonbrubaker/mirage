import os
import shutil
import random

# Paths to the folders
photos_path = "photos"
test_photos_path = "test_photos"

# Ensure the test_photos directory exists
os.makedirs(test_photos_path, exist_ok=True)

# Get all photo filenames in the photos directory
all_photos = os.listdir(photos_path)

# Select 1000 random photos (or all if there are fewer than 1000)
selected_photos = random.sample(all_photos, min(1000, len(all_photos)))

# Move the selected photos to test_photos
for photo in selected_photos:
    shutil.move(os.path.join(photos_path, photo), os.path.join(test_photos_path, photo))

# Now, handle the scores
scores_path = "scores.txt"
test_scores_path = "test_scores.txt"

# Initialize a set for faster lookup
selected_photos_set = set(selected_photos)

# Read scores.txt, write relevant lines to test_scores.txt, and others back to scores.txt
with open(scores_path, "r") as scores_file, \
     open(test_scores_path, "w") as test_scores_file, \
     open("temp_scores.txt", "w") as temp_scores_file:
    for line in scores_file:
        photo_name = line.split()[0]
        if photo_name in selected_photos_set:
            test_scores_file.write(line)
        else:
            temp_scores_file.write(line)

# Replace the original scores.txt with the updated one
os.replace("temp_scores.txt", scores_path)

print("Process completed: Photos moved and scores updated.")
