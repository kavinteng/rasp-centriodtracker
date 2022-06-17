import gdown
import os

if os.path.isdir('camcount') == False:
    url = 'https://drive.google.com/drive/folders/15raFceARgdxrvEN8sfVKrRrmkEPNhroo'
    gdown.download_folder(url)

if os.path.isdir('gender_age_model') == False:
    url = "https://drive.google.com/drive/u/1/folders/1n7WSJV0CdGY8vaPukLxZDXT3TjzEZ-Hp"
    gdown.download_folder(url)