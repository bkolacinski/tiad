import pandas as pd
from weasyprint import HTML, CSS
import os

def generate_pdf(df: pd.DataFrame, settings: dict, output_path: str):
    # CSS generation based on settings
    orientation = settings.get('orientation', 'portrait')
    font_size = settings.get('font_size', '12')
    title = settings.get('title', '')
    spacing_val = settings.get('spacing', 'normal')
    page_numbers = settings.get('page_numbers', False)

    # Przeliczenie marginesów wg odstępów
    if spacing_val == 'tight':
        padding = '4px 8px'
    elif spacing_val == 'loose':
        padding = '16px 24px'
    else:
        padding = '8px 12px'

    # Podstawowy CSS dokumentu
    css_content = f"""
    @page {{
        size: A4 {orientation};
        margin: 2cm;
    """

    if page_numbers:
        css_content += """
        @bottom-center {
            content: "Strona " counter(page) " z " counter(pages);
            font-size: 10pt;
            color: #555;
        }
        """

    css_content += f"""
    }}
    body {{
        font-family: Arial, sans-serif;
        font-size: {font_size}pt;
        color: #333;
    }}
    h1 {{
        text-align: center;
        font-size: {int(font_size) + 6}pt;
        margin-bottom: 20px;
        color: #000;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 20px;
    }}
    th, td {{
        border: 1px solid #ccc;
        padding: {padding};
        text-align: left;
    }}
    th {{
        background-color: #f4f4f4;
        font-weight: bold;
    }}
    /* Zebra striping for esthetics */
    tr:nth-child(even) {{
        background-color: #f9f9f9;
    }}
    """

    # HTML generation
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{title}</title>
    </head>
    <body>
        {"<h1>" + title + "</h1>" if title else ""}
        {df.to_html(index=False, border=0, classes='pdf-table', escape=True)}
    </body>
    </html>
    """

    # Generate PDF with WeasyPrint
    HTML(string=html_template).write_pdf(output_path, stylesheets=[CSS(string=css_content)])

    return output_path
