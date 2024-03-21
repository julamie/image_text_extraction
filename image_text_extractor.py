import pytesseract  # OCR library
import cv2          # image processing
import os
import random
import json
import time
from collections import defaultdict # manipulate dicts
import sys          # system information          
import subprocess   # run command on PC
import difflib      # find similar words

class Image_Folder:
    def __init__(self, directory: str):
        self.directory = directory                # path to the image folder
        self.image_paths = os.listdir(directory)  # list of all image paths in the directory
        # only use jpg, jpeg and png files
        self.image_paths = [filename for filename in self.image_paths if filename.lower().endswith(('.jpg', '.jpeg', ".png"))]

    def get_random_image_path(self) -> str:
        # choose a number from 0 to the number of images in the directory
        num_elements = random.randint(0, len(self.image_paths))
        # pick the image in list at previously chosen index
        curr_image_path = f"{self.directory}/{self.image_paths[num_elements]}"

        return curr_image_path

    def open_image(self, path: str):
        img = cv2.imread(path)

        return img

class Image_Detection:
    def __init__(self, image_folder: Image_Folder, image_path: str):
        self.image_folder = image_folder
        self.image_path = image_path
        self.image = None
        self.image_preprocessed_path = None
        self.image_preprocessed = None
        self.config = r"--psm 11" # psm 11 = Sparse text. Find as much text as possible in no particular order.
        self.min_confidence = 80
        self.data = None

    def get_image_data(self):
        self.image = self.image_folder.open_image(self.image_path)
        self.data = pytesseract.image_to_data(self.image, output_type=pytesseract.Output.DICT, config=self.config, lang="eng+ger")

        return self.data

    def get_text_as_list(self):
        text_list = []

        # add all words in data["text"] with a high enough confidence and store them in a list
        for i in range(0, len(self.data["text"])):
            word = self.data["text"][i]
            conf = float(self.data["conf"][i]) # confidence

            if conf > self.min_confidence:
                word = clean_word(word)
                if word:
                    # add the word only if there is anything left after cleaning
                    text_list.append(word)
        return text_list


    def add_bounding_boxes(self):
        # for every entry add one bounding box of a word in the picture
        for i in range(0, len(self.data["text"])):
            # upper left point of box
            x = self.data["left"][i]
            y = self.data["top"][i]

            w = self.data["width"][i]
            h = self.data["height"][i]

            text = self.data["text"][i]
            conf = float(self.data["conf"][i]) # confidence

            # if the guess is confident enough, add the bounding box
            if conf > self.min_confidence:
                cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(self.image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

        # open a resizable window
        cv2.namedWindow(f"Annotated image {self.image_path}", cv2.WINDOW_NORMAL)
        cv2.imshow(f"Annotated image {self.image_path}", self.image)
        cv2.waitKey(0)

    def extract_text_from_image(self, show_boxes=False):
        self.data = self.get_image_data()

        # use pytesseract to extract the text from the provided image
        word_list = self.get_text_as_list()
        extracted_text = " ".join(word_list) + "\n"

        # show image with all bounding boxes of words
        if show_boxes:
            self.add_bounding_boxes()

        return extracted_text

class Word_Dict:
    def __init__(self, image_folder: Image_Folder, export_file_name: str):
        self.image_paths = image_folder.image_paths

        # use the extracted words json, if it already exists
        try:
            with open("extracted_words.json", "r") as file:
                self.extracted_words = json.load(file)
        except:
            self.extracted_words = {}
        self.word_to_image_assignment = {}


def clean_word(word: str):
    # special case for "|", the program assumes | are I's, so switch them manually
    if word == "|": return "i"

    # changes word to lowercase, removes special chars
    word = ''.join(letter.lower() for letter in word if letter.isalnum())
    
    return word

def extract_words_of_all_images(image_folder: Image_Folder, word_dict: Word_Dict, verbose=False):
    # extract the words out of all images from the directory
    for index, img_path in enumerate(image_folder.image_paths):

        # skip images which had already been processed
        if img_path in word_dict.extracted_words:
            if verbose:
                print(f"[{index + 1}/{len(image_folder.image_paths)}] {img_path} has already been processed.")
            continue

        full_path = f"{directory}/{img_path}"

        # extract the words of an image and save them in word_dict.extracted_words
        detector = Image_Detection(image_folder=image_folder, image_path=full_path)
        _ = detector.extract_text_from_image(show_boxes=False)

        word_dict.extracted_words[img_path] = detector.get_text_as_list()

        # print progress
        if verbose:
            print(f"[{index + 1}/{len(image_folder.image_paths)}] {img_path} has been processed.")

    return word_dict.extracted_words

def create_word_image_match():
    # reverse the dictionary, for every word, there now exists a list with all images who contain the specific word
    word_image_match = defaultdict(set)
    for key, value in extract_words.items():
        # if no words were detected, add image path to empty key
        if extract_words[key] == []:
            word_image_match[""].add(key)
        for word in value:
            word_image_match[word].add(key)
    
    # convert the result sets to lists
    for key, value in word_image_match.items():
        word_image_match[key] = list(value)

    word_dict.word_to_image_assignment = word_image_match
    return word_image_match

def save_results(extracted_words_path: str, word_to_image_match_path: str):
    # save the extracted words in a json file, to skip redundant processing in the future
    with open(extracted_words_path, "w") as file:
        json.dump(extract_words, file, sort_keys=True, indent=4)
    with open(word_to_image_match_path, "w") as file:
        json.dump(word_dict.word_to_image_assignment, file, sort_keys=True, indent=4)

def request_images_by_word(word_dict: Word_Dict, image_folder: Image_Folder):
    possible_keywords = sorted(word_dict.word_to_image_assignment.keys())
    try:
        print("-" * 30)
        print("You can now type in your keyword to find the corresponding images. If you want to exit the program, please press Ctrl+C")
        while True:
            # ask user for a keyword
            keyword = clean_word(input("What is your keyword? "))
            results = word_dict.word_to_image_assignment[keyword]

            # if there are no results found, return images with similar keywords if possible
            if not results:
                print("No exact matches found.")

                # find similar keywords out of all keywords
                similar_keywords = difflib.get_close_matches(keyword, possible_keywords)
                if similar_keywords:
                    print(f"Here are similar keywords that you could use: {similar_keywords}")
                continue

            # print full path to image as a result
            results = [image_folder.directory + "/" + img_path for img_path in results]
            print(results)
            print("-" * 30)

            # open respective image viewer application
            imageViewerFromCommandLine = {'linux':'xdg-open',
                                    'win32':'explorer',
                                    'darwin':'open'}[sys.platform]
            for result in results:
                subprocess.run([imageViewerFromCommandLine, os.path.abspath(result)])
    except KeyboardInterrupt:
        print("\nExiting program...goodbye")

if __name__ == "__main__":
    directory = "Images"
    extracted_words_path = "extracted_words.json"
    word_to_image_match_path = "word_image_match.json"

    image_folder = Image_Folder(directory=directory)
    word_dict = Word_Dict(image_folder, extracted_words_path)

    print("Welcome to my image extraction tool. Put your images you want to process in the Images folder.")
    print("Depending on the number and size of images, this image processing can take a while.")
    print("If the images were already processed, the program can skip those to achieve faster execution time.")
    user_input = input("Press v for verbose execution or enter for normal execution... ")
    if user_input == "v": verbose_execution = True
    else: verbose_execution = False

    start_time = time.time()    # measure execution time
    extract_words = extract_words_of_all_images(image_folder, word_dict, verbose=verbose_execution)
    word_image_match = create_word_image_match()
    save_results(extracted_words_path, word_to_image_match_path)

    if verbose_execution: print(f"Processing of the images took{time.time() - start_time: .4f} seconds") # print execution time
    
    request_images_by_word(word_dict, image_folder)
