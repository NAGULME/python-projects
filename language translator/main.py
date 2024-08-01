# Step 1: pip install pygame
# (Pygame is used for creating video games and multimedia applications)
# Step 2: pip install gTTS
# (Google Text-to-Speech converter library)
# Step 3: pip install tkinter
# (Tkinter is used for creating graphical user interfaces)
# Step 4: pip install googletrans==4.0.0-rc1
# (Google Translator API for language translation)
# Step 5: pip install --upgrade pip
# (To update pip, the Python package manager)
# Step 6: pip install translate
# (Additional library for translation)

from translate import Translator
from requests.exceptions import RequestException

# List of languages with their corresponding codes
LANGUAGES = {
    1: "en",  # English
    2: "te",  # Telugu (Andhra Pradesh)
    3: "hi",  # Hindi
    4: "es",  # Spanish
    5: "pt",  # Portuguese
    6: "de",  # German
    7: "it",  # Italian
    8: "ja",  # Japanese
    9: "ko",  # Korean
    10: "ru", # Russian
    11: "ar", # Arabic
    12: "nl", # Dutch
    13: "el", # Greek
    14: "tr", # Turkish
    15: "sv", # Swedish
    16: "pl", # Polish
    17: "vi", # Vietnamese
    18: "ta", # Tamil (Indian language)
    19: "bn", # Bengali (Indian language)
    20: "ml", # Malayalam (Indian language)
    21: "pa", # Punjabi (Indian language)
    22: "kn", # Kannada (Indian language)
    23: "ur"  # Urdu (Indian language)
    # Add more languages as needed
}

# Function to get user input for the text to translate
def get_user_input():
    return input("Enter the text to translate (or 'exit' to quit): ")

# Function to display available languages
def display_languages():
    print("Available languages:")
    for index, language in LANGUAGES.items():
        print(f'{index}) {language.capitalize()}')

# Function to get the target language from the user
def get_target_language():
    while True:
        try:
            selected_lang = int(input("Select a language (1-23): "))
            if selected_lang not in LANGUAGES:
                raise ValueError("Invalid option selected.")
            return LANGUAGES[selected_lang]
        except ValueError:
            print("Invalid input. Please enter a valid numeric option.")

# Function to translate the text to the target language
def translate_text(user_input, target_language):
    try:
        translator = Translator(to_lang=target_language)
        translation = translator.translate(user_input)
        print(f"Translated text ({target_language}): {translation}")
    except RequestException as e:
        print(f"Translation failed. Request error: {str(e)}")
    except Exception as e:
        print(f"Translation failed with error: {str(e)}")

# Main function to run the translation program
def main():
    while True:
        user_input = get_user_input()
        if user_input.lower() == "exit":
            print("Exiting the translator.")
            break
        display_languages()
        target_language = get_target_language()
        translate_text(user_input, target_language)

if __name__ == "__main__":
    main()
