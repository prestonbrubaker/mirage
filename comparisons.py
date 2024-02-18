import tkinter as tk
from PIL import Image, ImageTk
import os
import random
import time

# Program start time
start_time = time.time()

# Iteration counter
it_C = 0

def update_photos():
    global left_photo_label, right_photo_label, left_photo_name, right_photo_name, info_label
    selected_photos = random.sample(photos, 2)
    left_photo_name, right_photo_name = selected_photos
    
    left_photo_path = os.path.join(photos_path, left_photo_name)
    right_photo_path = os.path.join(photos_path, right_photo_name)
    
    left_image = Image.open(left_photo_path)
    right_image = Image.open(right_photo_path)
    
    left_photo = ImageTk.PhotoImage(left_image)
    right_photo = ImageTk.PhotoImage(right_image)
    
    left_photo_label.configure(image=left_photo)
    left_photo_label.image = left_photo  # Keep a reference!
    
    right_photo_label.configure(image=right_photo)
    right_photo_label.image = right_photo  # Keep a reference!
    
    # Update info label with current iteration and elapsed time
    elapsed_time = time.time() - start_time
    info_label.config(text=f"Decisions made: {it_C}, Elapsed time: {elapsed_time:.1f} seconds")

def choose_photo(choice):
    global it_C
    """Append choice to ticker_tape.txt, update photos, and info label."""
    time_ep = time.time()
    with open("ticker_tape.txt", "a") as file:
        file.write(f"{left_photo_name} {right_photo_name} {choice} {time_ep} {it_C}\n")
    it_C += 1
    update_photos()

def on_key_press(event):
    """Handle key press events."""
    if event.char == "m":  # Right photo chosen
        choose_photo(1)
    elif event.char == "n":  # Left photo chosen
        choose_photo(0)

# Initialize tkinter window
root = tk.Tk()
root.title("Choose a Photo: 'n' for left, 'm' for right")
root.geometry("800x600")
root.configure(bg='grey')

# Load photos from the "photos" directory
photos_path = "photos"
photos = [f for f in os.listdir(photos_path) if os.path.isfile(os.path.join(photos_path, f))]

# Initialize info label and pack it first to be at the top
info_label = tk.Label(root, text="Decisions made: 0, Elapsed time: 0.0 seconds", bg='grey')
info_label.pack(side="top", pady=(10, 20))

# Create a container frame for photo frames
container_frame = tk.Frame(root, bg='grey')
container_frame.pack(expand=True, fill='both')

# Create frames for each photo within the container frame
left_frame = tk.Frame(container_frame, bg='light blue')
right_frame = tk.Frame(container_frame, bg='light blue')

# Pack the photo frames with a specific gap
left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
right_frame.pack(side="left", fill="both", expand=True, padx=(5, 0))

# Initialize photo labels within the frames
left_photo_label = tk.Label(left_frame, bg='grey')
left_photo_label.pack(fill="both", expand=True)

right_photo_label = tk.Label(right_frame, bg='grey')
right_photo_label.pack(fill="both", expand=True)

# Bind key press events
root.bind("<KeyPress>", on_key_press)

# First update to display photos and initialize info
update_photos()

# Start the GUI event loop
root.mainloop()
