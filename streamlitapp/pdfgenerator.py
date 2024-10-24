from fpdf import FPDF
import re,io
import fitz

def clean_text(text: str) -> str:
    # Remove or replace unsupported characters
    # You can customize this function to handle different characters
    return ''.join(c for c in text if ord(c) < 256)  # Keep only ASCII characters


def wrap_text(pdf: FPDF, text: str, max_width: float, code: bool = False) -> None:
    """Wraps text to fit the specified width. Format as code if specified."""
    # Set the font for code or regular text
    if code:
        pdf.set_font("Arial", size=10)  # Use a monospaced font for code
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

# Generate the first draft of the PDF
def generate_pdf(content: str):
    pdf = FPDF(orientation="landscape", format="A4")
    pdf.add_page()
    pdf.set_author("42 Chatbot")

    # Set margins
    margin = 10
    pdf.set_left_margin(margin)
    pdf.set_right_margin(margin)
    
    # Calculate maximum width for the text
    max_width = pdf.w - 2 * margin  # Total width minus left and right margins
    
    # Wrap and add each line to the PDF
    in_code_block = False
    code_block = ""
    
    for line in content.split("\n"):
        line = line.strip()
        
        # Check for start/end of code block
        if line.startswith("```") or line.startswith("``"):
            in_code_block = not in_code_block  # Toggle code block state
            
            # If we've just exited a code block, render the accumulated block
            if not in_code_block:
                if code_block:
                    pdf.ln(5)  # Add some space before the code block
                    wrap_text(pdf, code_block.strip(), max_width, code=True)  # Print the accumulated code block in Courier font
                    pdf.ln(5)  # Add space after the code block
                    code_block = ""  # Reset the code block after printing
            continue  # Skip the code block marker
        
        # If we are in a code block, accumulate lines
        if in_code_block:
            code_block += line + "\n"  # Keep the newline for each line in the code
        else:
            # Otherwise, wrap normal text in Times font
            wrap_text(pdf, line, max_width)

    # Return the PDF as bytes
    return bytes(pdf.output(dest='S'))

#clean up the pdf to appear nice
def edit_the_generated_pdf(pdfbytesObj: bytes):
    try:

        pdf_stream = io.BytesIO(pdfbytesObj)

        doc = fitz.open(stream=pdf_stream, filetype = "pdf")

        cleaned_text = []

        for page_num in range(len(doc)):

            page = doc.load_page(page_num)

            text = page.get_text("text")

            text = text.replace("*","")

            lines = text.split("\n")


            for line in lines:

                if line.startswith("#"):
                    line = line.replace("#","")
                    line = f"**{line}**"

                cleaned_text.append(line)

        final_cleaned_text = "\n\n".join(cleaned_text)

        return generate_pdf(content=final_cleaned_text)

    except Exception:
        pass