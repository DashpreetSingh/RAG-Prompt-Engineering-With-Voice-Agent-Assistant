import string
import docx
from definitions.definition import DATA


class Preprocess:
    def __init__(self):
        self.data = DATA

    def getText(self):
        fullText = []
        doc = docx.Document(self.data)
        for para in doc.paragraphs:
            fullText.append(para.text)
        text =  '\n'.join(fullText)
        return text

    def punctuationRemove(self):
        punction_text = []
        text = self.getText()
        # steps for consistency and for accuracy
        #lower case
        lower_case = text.lower()
        #punctuation removal
        removePun = lower_case.translate(str.maketrans('', '',string.punctuation))
        punction_text.append(removePun)
        return punction_text


# run = Preprocess()
# run.punctuationRemove()


