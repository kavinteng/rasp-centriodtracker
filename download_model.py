import gdown
import os

if os.path.isdir('camcount') == False:
    url = 'https://drive.google.com/drive/folders/15raFceARgdxrvEN8sfVKrRrmkEPNhroo'
    gdown.download_folder(url)