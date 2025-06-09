import os
import shutil


count = 17
for file in os.listdir("C:/Users/87790/Downloads/"):
    if file.endswith(".jpeg"):
        count += 1
        shutil.copy(f"C:/Users/87790/Downloads/{file}", f"C:/Users/87790/PycharmProjects/SDScratch/data/liutao/{count}.jpeg")


