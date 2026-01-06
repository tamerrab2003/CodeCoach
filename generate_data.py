import os
import json
import ollama
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import multiprocessing
import zipfile
import zlib
import base64
import re
from urllib.parse import unquote

# Import optional dependencies for document processing
try:
    import docx
except ImportError:
    docx = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from lxml import etree as ET
except Exception:
    ET = None
try:
    from PIL import Image
except Exception:
    Image = None
try:
    import pytesseract
except Exception:
    pytesseract = None

# Configuration
WORKSPACE_DIR = "./my_workspace"
OUTPUT_FILE = "data/custom_logic.jsonl"
MODEL_NAME = "deepseek-r1:1.5b"
SAMPLES_PER_FILE = 5
CPU_COUNT = os.cpu_count() or multiprocessing.cpu_count() or 4
NUM_WORKERS = 6

file_lock = threading.Lock()

def detect_file_type(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ('.xslt', '.xsl'):
        return 'XSLT'
    if ext in ('.xml',):
        return 'XML'
    if ext in ('.html', '.htm', '.htmlx', '.xhtml'):
        return 'HTML'
    if ext == '.pdf':
        return 'PDF'
    if ext == '.docx':
        return 'DOCX'
    if ext in ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff'):
        return 'IMAGE'
    if ext in ('.js', '.jsm', '.ts', '.tsx', '.jsx'):
        return 'JAVASCRIPT'
    if ext in ('.py',):
        return 'PYTHON'
    if ext in ('.go',):
        return 'GO'
    if ext in ('.css',):
        return 'CSS'
    return 'CODE'

def extract_text_from_file(file_path):
    """Extract text content from various file formats, including XSLT for DataPower."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == '.docx':
            if docx is None:
                return None
            doc = docx.Document(file_path)
            return '\n'.join([para.text for para in doc.paragraphs])

        elif ext == '.pdf':
            if PdfReader is None:
                return None
            reader = PdfReader(file_path)
            texts = []
            for page in reader.pages:
                try:
                    t = page.extract_text()
                    if t:
                        texts.append(t)
                except Exception:
                    continue
            return '\n'.join(texts)

        elif ext in ('.html', '.htm'):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            if BeautifulSoup:
                soup = BeautifulSoup(content, 'html.parser')
                return soup.get_text(separator='\n')
            return content

        elif ext in ('.xslt', '.xsl'):
            # Prefer structured extraction to spotlight templates and matches
            with open(file_path, "r", encoding="utf-8") as f:
                xml_txt = f.read()
            if ET:
                try:
                    root = ET.fromstring(xml_txt.encode('utf-8'))
                    ns = {'xsl': 'http://www.w3.org/1999/XSL/Transform'}
                    templates = root.findall('.//xsl:template', namespaces=ns)
                    lines = []
                    for tpl in templates:
                        match = tpl.get('match', '')
                        name = tpl.get('name', '')
                        mode = tpl.get('mode', '')
                        select_nodes = tpl.findall('.//xsl:value-of', namespaces=ns)
                        selects = [n.get('select', '') for n in select_nodes if n.get('select')]
                        lines.append(f"template name={name} match={match} mode={mode} selects={selects}")
                    if lines:
                        header = "XSLT Templates Summary:\n" + "\n".join(lines) + "\n\nRaw XML:\n"
                        return header + xml_txt
                except Exception:
                    # fallback to raw text
                    return xml_txt
            return xml_txt
        elif ext in ('.drawio', '.dtmp'):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if it's a compressed mxfile
                if '<mxfile' in content:
                    # Simple regex to find diagram data if compressed
                    # Often structure is <diagram ...>BASE64</diagram>
                    # But if it's plain XML, we just return it.
                    if 'compressed="false"' in content:
                        return content
                    
                    # Try to extract base64 payload from <diagram> tag
                    match = re.search(r'<diagram[^>]*>(.*?)</diagram>', content, re.DOTALL)
                    if match:
                        b64_data = match.group(1).strip()
                        try:
                            # Draw.io compression: Base64 -> Inflate (no header) -> URL Decode
                            # Actually usually: Deflate -> Base64.
                            # Standard draw.io: Raw Deflate (no zlib header)
                            compressed = base64.b64decode(b64_data)
                            try:
                                xml_data = zlib.decompress(compressed, -15) # -15 for raw deflate
                            except:
                                xml_data = zlib.decompress(compressed)
                            
                            return unquote(xml_data.decode('utf-8'))
                        except Exception:
                            # Fallback to returning raw content if decompression fails
                            pass
                return content
            except Exception:
                return None

        elif ext == '.vsdx':
            # Visio is a ZIP of XMLs. Extract text from page XMLs.
            try:
                text_content = []
                with zipfile.ZipFile(file_path, 'r') as z:
                    for name in z.namelist():
                        if name.startswith('visio/pages/page') and name.endswith('.xml'):
                            xml_data = z.read(name).decode('utf-8', errors='ignore')
                            # Simple XML tag stripping to get text
                            # Visio text is often in <v:text> or <a:t>
                            # We'll just strip all tags for simplicity
                            text = re.sub(r'<[^>]+>', ' ', xml_data)
                            text = re.sub(r'\s+', ' ', text).strip()
                            if text:
                                text_content.append(f"--- Page {name} ---\n{text}")
                return "\n".join(text_content)
            except Exception:
                return f"[Error reading VSDX file: {os.path.basename(file_path)}]"

        elif ext in ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.tiff'):
            if Image is not None and pytesseract is not None:
                try:
                    img = Image.open(file_path)
                    text = pytesseract.image_to_string(img)
                    if text and len(text.strip()) > 0:
                        return text
                except Exception:
                    pass
            return f"Image file: {os.path.basename(file_path)}"

        else:
            # Default to text
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()

    except Exception:
        return None

def process_single_sample(relative_name, content, file_type, pbar):
    sample = generate_reasoning_sample(relative_name, content, file_type)
    with file_lock:
        if sample:
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(sample) + "\n")
        pbar.update(1)

def generate_reasoning_sample(filename, content, file_type):
    header = f"File Type: {file_type}\n"
    ext = os.path.splitext(filename)[1].lower()
    if file_type == 'XSLT':
        header += "Note: DataPower-style XSLT. Focus on templates, match patterns, modes, variables, and transformation flow.\n"
    if file_type == 'JAVASCRIPT' and ext in ('.tsx', '.jsx'):
        header += "Note: React JSX/TSX component. Consider props/state, event flow, and component structure.\n"
    if file_type == 'JAVASCRIPT' and ext == '.ts':
        header += "Note: TypeScript file. Consider types, interfaces, generics, and compile-time constraints.\n"
    if file_type == 'DIAGRAM':
        header += "Note: Diagram file (Visio/Draw.io). The content is XML representing shapes and connections. Focus on the flow, hierarchy, and labels.\n"
    prompt = f"""
    {header}
    Analyze the following file: {filename}
    
    File Content:
    ```
    {content[:10000]}
    ```
    
    Task:
    1. Formulate a challenging technical question or request about the logic, structure, or purpose of this content.
    2. Provide a detailed "Chain of Thought" (CoT) reasoning process that explains how the content works or answers the question.
    3. Provide the final concise answer/output.
    
    Output Format:
    You must output a valid JSON object with exactly these keys: "input", "reasoning", "output".
    Do not add any markdown formatting (like ```json) around the response. Just the raw JSON string.
    
    Example structure:
    {{
        "input": "What happens if I call factorial(-1)?",
        "reasoning": "The function checks if n < 0. If so, it raises a ValueError. So for input -1...",
        "output": "It raises a ValueError: 'Factorial is not defined for negative numbers'."
    }}
    """
    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            format='json',
            options={
                "keep_alive": "10m",
                "num_ctx": 8192,
                "temperature": 0.7
            }
        )
        return json.loads(response['response'])
    except Exception:
        return None

def warmup_model():
    try:
        _ = ollama.generate(
            model=MODEL_NAME,
            prompt="Warm up. Return a JSON object with keys input, reasoning, output.",
            format='json',
            options={"keep_alive": "10m"}
        )
    except Exception:
        pass

def main():
    if not os.path.exists(WORKSPACE_DIR):
        print(f"Error: Directory {WORKSPACE_DIR} not found.")
        return

    files = []
    for root, dirs, filenames in os.walk(WORKSPACE_DIR):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for filename in filenames:
            if filename.startswith('.'):
                continue
            files.append(os.path.join(root, filename))

    if not files:
        print("No files found in workspace.")
        return

    print(f"Found {len(files)} files in {WORKSPACE_DIR}. Preparing to scan...")
    
    file_items = []
    for file_path in files:
        content = extract_text_from_file(file_path)
        if content and len(content.strip()) > 0:
            relative_name = os.path.relpath(file_path, WORKSPACE_DIR)
            file_type = detect_file_type(file_path)
            file_items.append((relative_name, content, file_type))
    
    print(f"Successfully loaded content from {len(file_items)} files.")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        pass

    total_tasks = len(file_items) * SAMPLES_PER_FILE
    print(f"Starting generation with {NUM_WORKERS} workers. Total samples to generate: {total_tasks}")

    warmup_model()

    with tqdm(total=total_tasks, desc="Generating Samples") as pbar:
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            for relative_name, content, file_type in file_items:
                for _ in range(SAMPLES_PER_FILE):
                    futures.append(
                        executor.submit(process_single_sample, relative_name, content, file_type, pbar)
                    )
            for _ in as_completed(futures):
                pass

    with open(OUTPUT_FILE, "r") as f:
        count = sum(1 for _ in f)
    print(f"\nSuccess! Generated {count} samples in {OUTPUT_FILE}")
    print("You can now train the model using:")
    print(f"python train.py --dataset custom --data_path {OUTPUT_FILE} ...")

if __name__ == "__main__":
    main()
