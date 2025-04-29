import json
import sys

def extract_markdown(ipynb_file):
    """Extracts all Markdown content from an .ipynb file.

    Args:
        ipynb_file (str): The path to the .ipynb file.

    Returns:
        list: A list of strings, where each string is a Markdown cell's content.
              Returns an empty list if the file is not found or an error occurs.
    """
    markdown_cells = []
    try:
        with open(ipynb_file, 'r', encoding='utf-8') as f:
            notebook_data = json.load(f)
            for cell in notebook_data.get('cells', []):
                if cell['cell_type'] == 'markdown':
                    markdown_cells.append(cell['source'])
    except FileNotFoundError:
        print(f"Error: File not found at '{ipynb_file}'")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{ipynb_file}'")
    except Exception as e:
        print(f"An error occurred: {e}")
    return markdown_cells

def save_markdown(markdown_content, output_file="extracted_markdown.md"):
    """Saves a list of Markdown strings to a single .md file.

    Args:
        markdown_content (list): A list of Markdown strings.
        output_file (str, optional): The name of the output .md file.
                                      Defaults to "extracted_markdown.md".
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, content in enumerate(markdown_content):
                if isinstance(content, list):
                    f.write("".join(content))
                else:
                    f.write(content)
                f.write("\n\n")
        print(f"Markdown content saved to '{output_file}'")
    except Exception as e:
        print(f"An error occurred while saving to file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_markdown.py <path_to_your_notebook.ipynb>")
        sys.exit(1)

    notebook_path = sys.argv[1]
    markdown_content = extract_markdown(notebook_path)

    if markdown_content:
        save_markdown(markdown_content)