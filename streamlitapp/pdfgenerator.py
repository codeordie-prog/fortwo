from fpdf import FPDF
import re


def clean_text(text:str)-> str:

    text = re.sub(r'[“”]','"',text)
    text = re.sub(r'[‘’]',"'",text)

    return text


def wrap_text(pdf:FPDF, text:str, max_width:float, code:bool):

    if code:
        pdf.set_font("Courier",size=10)

    else:
        pdf.set_font("Times",size=10)

    
    #split text

    lines = text.split("\n")

    for line in lines:

        words = line.split(' ')
        current_line = ""

        for word in words:

            #check if the line exceeds the width prior to adding a new word
            if pdf.get_string_width(current_line+word)<max_width:
                current_line+=f"{word}"

            else:

                if current_line:
                    pdf.cell(0,5,txt=current_line.strip(),ln=True)

                current_line = f"{word}"
        
        if current_line:
            pdf.cell(0,5,txt=current_line.strip(),ln=True)


def generate_pdf(text:str):
    pdf = FPDF(orientation="landscape",format="A4")
    pdf.add_page()
    pdf.set_author("Fortytwo bot")
    
    margin = 10
    pdf.set_left_margin(margin)
    pdf.set_right_margin(margin)

    #calculate max width

    max_width = pdf.w - 2 * margin

    in_code_block = False
    code_block = ""

    for line in text.split("\n"):

        line = line.strip()

        if line.startswith("```"):
            in_code_block = not in_code_block

            if in_code_block:
                code_block="" #reset code block

            else:
                wrap_text(pdf,code_block,max_width,code=True)

        if in_code_block:
            code_block+=line+"\n"

        else:
            wrap_text(pdf,line,max_width,code=False)

    return pdf.output(dest="S").encode('latin1')
