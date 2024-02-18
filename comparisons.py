import tkinter as tk
from PIL import Image, ImageTk
import os
import random

def choose_photo(choice):
    """Append choice to ticker_tape.txt and close the window."""
    with open("ticker_tape.txt", "a") as file:
        file.write(f"{left_photo_name} {right_photo_name} {choice}\n")
    root.destroy()

def on_key_press(event):
    """Handle key press events."""
    if event.char == "m":  # Right photo chosen
        choose_photo(1)
    elif event.char == "n":  # Left photo chosen
        choose_photo(0)
while True:
    # Initialize tkinter window
    root = tk.Tk()
    root.title("Choose a Photo")
    
    # Load and display two random photos from the "photos" directory
    photos_path = "photos"
    photos = [f for f in os.listdir(photos_path) if os.path.isfile(os.path.join(photos_path, f))]
    selected_photos = random.sample(photos, 2)
    
    left_photo_name, right_photo_name = selected_photos
    
    left_photo_path = os.path.join(photos_path, left_photo_name)
    right_photo_path = os.path.join(photos_path, right_photo_name)
    
    left_image = Image.open(left_photo_path)
    right_image = Image.open(right_photo_path)
    
    left_photo = ImageTk.PhotoImage(left_image)
    right_photo = ImageTk.PhotoImage(right_image)
    
    left_label = tk.Label(root, image=left_photo)
    left_label.pack(side="left")
    
    right_label = tk.Label(root, image=right_photo)
    right_label.pack(side="right")
    
    # Bind key press events
    root.bind("<KeyPress>", on_key_press)
    
    # Start the GUI event loop
    root.mainloop()
