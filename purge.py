import os

# Path to the directory containing photos
photos_path = "photos"
# Path to the text file
text_file_path = "ticker_tape.txt"

def remove_unreferenced_photos(photos_path, text_file_path):
    # Step 1: Extract referenced photo filenames from the text file
    referenced_photos = set()
    with open(text_file_path, "r") as file:
        for line in file:
            parts = line.split()
            # Assuming the first two parts of each line are photo filenames
            referenced_photos.add(parts[0])
            referenced_photos.add(parts[1])

    # Step 2: List all photo files in the photos directory
    all_photos = set(os.listdir(photos_path))

    # Step 3: Determine which photos are never referenced
    unreferenced_photos = all_photos - referenced_photos

    # Step 4: Remove unreferenced photo files
    for photo in unreferenced_photos:
        os.remove(os.path.join(photos_path, photo))
        print(f"Removed {photo}")

# Execute the function
remove_unreferenced_photos(photos_path, text_file_path)
