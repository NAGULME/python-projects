# Make sure to run this code in an environment where Tkinter and Pillow are installed. You can install Pillow using:
# step1: pip install pillow

#ensure that your environment has access to requests for downloading the image
# step2:pip install requests



import tkinter as tk
from tkinter import ttk
from translate import Translator
from gtts import gTTS
import pygame
from io import BytesIO
from PIL import Image, ImageTk
import requests

# Initialize pygame mixer
pygame.mixer.init()

languages = {
    "English": "en",
    "Telugu": "te",
    "Hindi": "hi",
    "Tamil": "ta",
    # Add more languages
}

# Store the translation audio in a variable
translation_audio = None

# Function to handle translation
def translate_text():
    global translation_audio
    user_input = input_entry.get()
    selected_lang_name = language_var.get()
    if selected_lang_name in languages:
        selected_lang_code = languages[selected_lang_name]
        translator = Translator(to_lang=selected_lang_code)
        translation = translator.translate(user_input)
        result_label.config(text=translation, background="light blue")
        # Use gTTS to convert translation to speech and save to an in-memory file
        tts = gTTS(text=translation, lang=selected_lang_code)
        translation_audio = BytesIO()
        tts.write_to_fp(translation_audio)
        translation_audio.seek(0)
    else:
        result_label.config(
            text="Please select a valid language", background="light blue"
        )

# Function to play the audio
def play_audio():
    global translation_audio
    if translation_audio:
        translation_audio.seek(0)
        pygame.mixer.music.load(translation_audio)
        pygame.mixer.music.play()

# Creating the main application window
app = tk.Tk()
app.title("Text Translator with Audio")
app.geometry("800x600")

# Load the background image after initializing the root window
response = requests.get("https://images.rawpixel.com/image_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvdjEwMTYtYy0wOF8xLWtzaDZtemEzLmpwZw.jpg")
bg_image = Image.open(BytesIO(response.content))
bg_image = bg_image.resize((800, 600), Image.LANCZOS)  # Resize image to fit the window
bg_photo = ImageTk.PhotoImage(bg_image)

# Create a canvas and add the background image
canvas = tk.Canvas(app, width=800, height=600)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=bg_photo, anchor="nw")

# Create and configure widgets
input_label = ttk.Label(app, text="Enter the text:", background="light blue")
input_entry = ttk.Entry(app, width=30)
language_label = ttk.Label(app, text="Select a language:", background="light blue")
language_var = tk.StringVar()
language_dropdown = ttk.Combobox(app, textvariable=language_var, values=list(languages.keys()))
translate_button = ttk.Button(app, text="Translate", command=translate_text)
play_button = ttk.Button(app, text="Play audio", command=play_audio)
result_label = ttk.Label(app, text="Translation will appear here", wraplength=280, background="light blue")

# Arrange widgets on the canvas
canvas.create_window(100, 50, anchor="nw", window=input_label)
canvas.create_window(250, 50, anchor="nw", window=input_entry)
canvas.create_window(100, 100, anchor="nw", window=language_label)
canvas.create_window(250, 100, anchor="nw", window=language_dropdown)
canvas.create_window(150, 150, anchor="nw", window=translate_button)
canvas.create_window(250, 150, anchor="nw", window=play_button)
canvas.create_window(100, 200, anchor="nw", window=result_label)

# Start the tkinter main loop
app.mainloop()
