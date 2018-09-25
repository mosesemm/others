from pytesseract import image_to_string 
from PIL import Image


def parse_certifcate(file):
    try:

        #we parsing Umalusi certifcate

        image_text = image_to_string(Image.open(file))
        #print(image_text)

        #subjects are between "subject passed and Aggregate"
        image_text = image_text.split('Subjects passed')[1]
        image_text = image_text.split('Aggregate')[0]

        #valid subject lines will always be more than 4 characters since, they include subject names and scores
        subject_lines = [get_subject_and_symbol(subject_line.split(" ")) for subject_line in image_text.split('\n') if len(subject_line) > 6]

        return {'subjects': subject_lines}

    except Exception as e:
        print(e)

def get_subject_and_symbol(tokens):
    if len(tokens) <= 0:
        return None
    #assumming last part of the line, is range of marks
    return { 'subject': (" ".join(tokens[:-2])), 
                'symbol' : (tokens[len(tokens) -2])}

#print(parse_certifcate(Image.open('/home/moses7/school/human_computer_interaction/practicals/IMG_quick.jpg')))

#" ".join( test[:-1])
#https://stackoverflow.com/questions/10467623/uploading-file-via-base64
    