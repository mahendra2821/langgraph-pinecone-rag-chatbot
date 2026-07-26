import fitz


def load_pdf(path):

    document = fitz.open(path)

    text = ""

    for page_number, page in enumerate(document):

        page_text = page.get_text()

        text += page_text


    return text