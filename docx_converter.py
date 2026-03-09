import pandas as pd
from docx import Document
from docx.shared import Pt, Inches, Cm
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from docx.enum.section import WD_ORIENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

def add_page_number(run):
    fldChar1 = OxmlElement('w:fldChar')
    fldChar1.set(qn('w:fldCharType'), 'begin')

    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = "PAGE"

    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'separate')

    fldChar3 = OxmlElement('w:fldChar')
    fldChar3.set(qn('w:fldCharType'), 'end')

    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)
    run._r.append(fldChar3)

def generate_docx(df: pd.DataFrame, settings: dict, output_path: str):
    doc = Document()

    # Orientation
    section = doc.sections[0]
    if settings.get('orientation', 'portrait') == 'landscape':
        new_width, new_height = section.page_height, section.page_width
        section.orientation = WD_ORIENT.LANDSCAPE
        section.page_width = new_width
        section.page_height = new_height

    # Title
    title = settings.get('title', '')
    if title:
        heading = doc.add_heading(title, 0)
        heading.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

    font_size_pt = int(settings.get('font_size', 10))

    # Table
    # Rows + 1 for header
    table = doc.add_table(rows=1, cols=len(df.columns))
    table.style = 'Table Grid'

    # Headers
    hdr_cells = table.rows[0].cells
    for i, column_name in enumerate(df.columns):
        hdr_cells[i].text = str(column_name)
        for paragraph in hdr_cells[i].paragraphs:
            for run in paragraph.runs:
                run.font.bold = True
                run.font.size = Pt(font_size_pt)

    # Spacing mapping
    spacing_val = settings.get('spacing', 'normal')
    if spacing_val == 'tight':
        space_after = Pt(2)
    elif spacing_val == 'loose':
        space_after = Pt(12)
    else:
        space_after = Pt(6) # normal

    # Data
    for index, row in df.iterrows():
        row_cells = table.add_row().cells
        for i, val in enumerate(row):
            cell = row_cells[i]
            cell.text = str(val) if pd.notnull(val) else ""

            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = space_after
                for run in paragraph.runs:
                    run.font.size = Pt(font_size_pt)

    # Page numbers
    if settings.get('page_numbers'):
        footer = section.footer
        p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
        p.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER
        run = p.add_run()
        add_page_number(run)

    doc.save(output_path)
    return output_path
