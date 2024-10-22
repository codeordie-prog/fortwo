from fpdf import FPDF
import re

def clean_text(text: str) -> str:
    # Remove or replace unsupported characters
    # You can customize this function to handle different characters
    return ''.join(c for c in text if ord(c) < 256)  # Keep only ASCII characters

def wrap_text(pdf: FPDF, text: str, max_width: float, code: bool = False) -> None:
    """Wraps text to fit the specified width. Format as code if specified."""
    # Set the font for code or regular text
    if code:
        pdf.set_font("Courier", size=10)  # Use a monospaced font for code
    else:
        pdf.set_font("Times", size=10)  # Use the standard font for regular text

    # Split text into lines for proper handling
    lines = text.split('\n')

    for line in lines:
        # Split each line into words
        words = line.split(' ')
        current_line = ''
        
        for word in words:
            # Check if adding the next word exceeds the max width
            if pdf.get_string_width(current_line + word) < max_width:
                current_line += f"{word} "
            else:
                # If current line is not empty, print it
                if current_line:
                    pdf.cell(0, 5, txt=current_line.strip(), ln=True)
                current_line = f"{word} "

        # Print any remaining text in the current line
        if current_line:
            pdf.cell(0, 5, txt=current_line.strip(), ln=True)

def generate_pdf(content: str):
    pdf = FPDF(orientation="landscape", format="A4")
    pdf.add_page()
    pdf.set_author("42 Chatbot")
    
    # Set margins
    margin = 10
    pdf.set_left_margin(margin)
    pdf.set_right_margin(margin)
    
    # Calculate maximum width for the text
    max_width = pdf.w - 2 * margin  # Total width - left and right margins
    
    # Wrap and add each line to the PDF
    in_code_block = False
    code_block = ""
    
    for line in content.split("\n"):
        line = line.strip()
        
        # Check for start of code block
        if line.startswith("```"):
            in_code_block = not in_code_block  # Toggle code block state
            if in_code_block:  # Starting a code block
                code_block = ""  # Reset code block
            else:  # Ending a code block
                wrap_text(pdf, code_block, max_width, code=True)  # Print the code block
                continue  # Skip the code block marker
        
        # If we are in a code block, accumulate lines
        if in_code_block:
            code_block += line + "\n"  # Keep the newline for each line in the code
        else:
            wrap_text(pdf, line, max_width)

    # Return the PDF as UTF-8
    return pdf.output(dest='S')
