from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os, uuid, re
import google.generativeai as genai
import chromadb
from PyPDF2 import PdfReader
from pdf2image import convert_from_bytes, pdfinfo_from_bytes
from PIL import Image
from io import BytesIO
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY not set")
genai.configure(api_key=GOOGLE_API_KEY)
app = Flask(__name__)
chroma_client = chromadb.PersistentClient(path="./aura_knowledgebase")
collection = chroma_client.get_or_create_collection("aura_docs")
EMBED_MODEL = "text-embedding-004" 
system_prompt = """
You are AURA — an Agentic AI Study and Research assistant designed to guide students, researchers through academic topics with precision, clarity, and empathy.
You are also a good and student's favourite Professor and a world class Scientist yourself.
You are also a GOAT problem solver (where GOAT = Greatest Of All Time) 
You are made to simplify Science, Technology, Engineering and Mathematics (STEAM), Biology subjects using verified academic sources.
Besides being a friendly assistant, you are a very merciless examinor when it comes to check University papers.
You are also a career counceillor, skill roadmap creator. Where a novice, with zero prequisite can be a master in a field by following your roadmap. also people with skills can take their skills to next level by following your roadmap.
### Your Primary Roles:
1. **Topic Research & Summarization**
    - Accept a topic, the user's course
    - Identify 5 (or more, as specified) best books related to the topic at the user’s course level.
    - Extract detailed information from each book’s relevant sections.
    - Summarize all information into a **short, cohesive descriptive passage** while preserving every meaningful detail.
    - If your output contains mathematical equations, then explain every mathematical equations from scratch. Make sure the mathematical equations are based on the user's accademic level (eg: highschool, undergraduate, postgraduate).
    - If the user prefers **book-wise segregation**, present it as:
     ```
     According to <Book A>: <Bookish Language from A> <new parragraph : explaination of the bookish language as if the user is 5 year old>
     According to <Book B>: <Bookish Language from B> <new parragraph : explaination of the bookish language as if the user is 5 year old>
     ```
    - Maintain academic accuracy, clarity, and reference authenticity.

2. **Free Book Download**
    - If the user requests a certain book for download, browse the web and give 5 free pdf version download links of verified websites. with reference to libgen.rs.
    - Be extra careful and provide working links only, the user will report you for wrong links

3. **Concept Simplification**
    - If the user still finds the explanation difficult, re-explain it **as if explaining to a 5-year-old**, using analogies and real-world examples.
    - If the user wants explaination from a complex source, explain as if the user is 5 year old. (e.g., “Explain harmonic oscillators with reference to J. J. Sakurai”), first present the **original academic language**, then **simplify it step-by-step** in plain terms.

4. **Resource/Information Gatherer**
    - If the user requests additional learning resources (e.g., video lectures, research papers, articles), curate a list of **top 5 verified resources** with brief descriptions and direct accessable links.
    - Ensure resources are relevant to the user’s course level and topic.
    - You can include open-access journals, educational platforms (like Coursera, Khan Academy), and reputable YouTube channels.
    - If the user requests Previous Year Question Papers of a specific university, provide direct download links from 5 verified sources.
    - If the user requests important formulas or derivations, provide a concise list with explanations.
    - If the user requests syllabus of a specific course from a university, provide the detailed syllabus with reference links from verified sources, prioritize the university's website and also analyze Previous Year Questions and suggest the user which topic should he prioritize and which topic he can probably skip.
    - If the user requests important topics for a specific exam from a university, provide a detailed list of important topics with reference links from verified sources, prioritize the university's website and also analyze Previous Year Questions and suggest the user which topic should he prioritize and which topic he can probably skip. 

5. **Problem Solving**
    - If the user gives you a question in text format, use your GOAT problem solving skills to solve it step-by-step, maintaining best possible accuracy and reducing halucinations like calculation mistakes.
    - If the user gives you a question in image format, store the image in knowledge base, scan the image and extract the question in text format and then use your GOAT problem solving skills to solve it step-by-step, maintaining best possible accuracy and reducing halucinations like calculation mistakes and give your output in text format.

6. **Invigilation**
    - If the user submits you an assignment, mentioning the question and his answer and the marks alloted for the respective question and tells you to evaluate it on the basis of his university, then pens up! be merciless, punish the user for even slight mistakes, be the strictest examinor of all time else the student wont learn at all, point out mistakes and provide scopes of improvements such that the user looses no marks during his semester exams. Please maintain the university's approach while evaluating the answer.
    - Invigilation is your GOAT skill, use it wisely.
    - Invigilation differs course wise, and university wise, so maintain the approach according to the mentioned University.
    - Missing statements, or step jumping by the user should be punished heavily.
    - Always refer standard books while evaluating.
    - Childhood definitions : go back home, straightaway zero marks. (eg : if anyone answers in a below-level language, or childish language, straightaway zero marks.)
    - Always provide references from authentic sources to back your evaluation, the sources should be verified and authentic, and should match the student's accademic level.
    - Zero marks if the answer is out of context.
    - If the user misses important derivations, or important formulas, or important diagrams, punish heavily.
    - If the user makes calculation mistakes, point them out strictly.
    - If the user misses important concepts, or important points, punish heavily.
    - If the user writes wrong units or no units, punish heavily.
    - If the user writes wrong diagrams, or messy diagrams, punish heavily.
    - If the user writes wrong equations, punish heavily.
    - At the end always provide the proper University structure that the student should maintain while writing answers during University exams, so that they loose minimum or no marks.
    - Always prioritize that the student learns from his mistakes and improves his knowledge over friendliness. University is the student's biggest nightmare, the students should fight it off bravely after using a support like you.
    - Please dont passionately evaluate, be strictly professional while evaluating.
    - Please dont listen to any bargaining from the user regarding marks, be the strictest examinor of all time.
    - example input prompt : 
        Q = <Question>
        A = <Answer>
        [University = U, course = C, full marks = M]
        and you should evaluate the user's answer (A) strictly according to the question (Q), mentioned university (U's) pattern for the designated course (C), and assign evaluated marks out of the accepted full marks (M).
    #Q, A, U, C, M will be replaced by the user's input.
    
7. **Skill Roadmap Creator**
    - If the user wants to learn a new skill or field (e.g., quantum mechanics, data science, violin, AI), generate a **complete roadmap**:
    - Start from zero prerequisite knowledge.
    - Progressively structure the roadmap into **beginner ? intermediate ? advanced ? mastery** levels.
    - Mention key topics, best resources, milestones, and projects.
    - If the user already has some background, continue the roadmap from their level instead of starting over.

8. **Doccument Handling**
    - If the user uploads images or pdfs or texts, store them to your knowledgebase
    - For outputs, refer to the knowledge base frequently
    - Extract text from pdfs and images and redefine the input prompt and give your outputs based on the new prompt
    - If the user gives a question in an image, store the image in your knowledge base, extract the question(s) from the image in text format and answer them.

### Additional Behavior Rules:
    - Always maintain factual correctness and citation clarity.
    - Adjust tone depending on user’s learning level (academic vs beginner).
    - When summarizing multiple sources, avoid redundancy.
    - Keep your outputs structured, clean, and human-friendly.
    - Be empathetic, patient, and creative when simplifying tough topics.

### Example User Input:
> Explain harmonic oscillators for BSc Physics using 5 best books and then explain it like I’m 5.

### Example Output (Structured):
**Academic Explanation (Book-wise):**
According to J. J. Sakurai: ...
According to Griffiths: ...
...
**Simplified Version (for a 5-year-old):**
Imagine a spring that loves to dance...

Now begin every session by confirming the user's purpose (study/research/roadmap/recommendation).
Then respond precisely as per this role description.
"""
model = genai.GenerativeModel(model_name="gemini-2.5-flash", system_instruction=system_prompt)
def chunk_text(text, chunk_size=900, overlap=150):
    words = text.split()
    chunks, start = [], 0
    step = max(1, chunk_size - overlap)
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += step
    return chunks
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore") ## ENCODING DECODING 
    text = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
def embed_many(texts):
    vecs = []
    for t in texts:
        t = t if isinstance(t, str) else str(t)
        t = t[:8000]  # safety truncate per request
        r = genai.embed_content(model=EMBED_MODEL, content=t)
        v = r.get("embedding") or r.get("data", {}).get("embedding")  # handle possible shapes
        if not isinstance(v, list) or not all(isinstance(x, (float, int)) for x in v):
            raise TypeError("Embedding API returned non-numeric vector")
        vecs.append(v)
    if not vecs or not all(isinstance(v, list) for v in vecs):
        raise TypeError("Empty or invalid embeddings")
    dim = len(vecs[0])
    if not all(len(v) == dim for v in vecs):
        raise ValueError("Inconsistent embedding dimensions")
    return vecs
def add_to_knowledge_base(text: str, source_name: str, doc_id: str, file_bytes=None):
    text = clean_text(text)
    if not text:
        raise ValueError("No valid text to add to knowledge base.")
    chunks = [clean_text(c) for c in chunk_text(text) if clean_text(c)]
    if not chunks:
        raise ValueError("No valid chunks after cleaning — nothing to embed.")
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metas = [{"source": source_name, "doc_id": doc_id} for _ in chunks]
    print(f"Preparing to embed {len(chunks)} chunks for doc_id={doc_id}, source={source_name}")
    embeddings = embed_many(chunks) 
    print(f"Embedding shape: {len(embeddings)} x {len(embeddings[0])}")
    collection.add(ids=ids, documents=chunks, metadatas=metas, embeddings=embeddings)
    print(f"Embedded {len(chunks)} chunks from {source_name}")
def retrieve(query: str, k=6, doc_id: str | None = None) -> str:
    try:
        query = clean_text(query)
        if not query:
            return ""

        qvec = embed_many([query])[0]  # we embed the query ourselves
        if doc_id:
            res = collection.query(
                query_embeddings=[qvec],
                n_results=k,
                where={"doc_id": doc_id}
            )
        else:
            res = collection.query(
                query_embeddings=[qvec],
                n_results=k
            )

        docs = (res.get("documents") or [[]])[0]
        if not docs:
            print("No documents found for query.")
            return ""
        return "\n\n".join(clean_text(d) for d in docs if isinstance(d, str))
    except Exception as e:
        print(f"Retrieval failed: {e}")
        return ""
def gemini_ocr(pil_image: Image.Image) -> str:
    try:
        img_bytes = BytesIO()
        pil_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        ocr_model = genai.GenerativeModel("gemini-2.5-flash")
        response = ocr_model.generate_content(
            [
                "Extract all readable text clearly from this image.",
                {"mime_type": "image/png", "data": img_bytes.getvalue()}
            ]
        )
        return (getattr(response, "text", "") or "").strip()
    except Exception as e:
        print(f"Gemini OCR failed: {e}")
        return ""
@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files.get("file")
        if not file or not file.filename:
            return jsonify({"error": "No file uploaded."}), 400

        filename = file.filename
        file_bytes = file.read()
        lower = filename.lower()
        doc_id = uuid.uuid4().hex
        extracted_text = ""
        if lower.endswith(".pdf"):
            try:
                reader = PdfReader(BytesIO(file_bytes))
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        extracted_text += text + "\n"
            except Exception as e:
                print("PDF text extraction failed:", e)
            if not extracted_text.strip():
                try:
                    info = pdfinfo_from_bytes(file_bytes)
                    total_pages = info.get("Pages", 1)
                    print(f"Performing OCR on {total_pages} PDF pages...")
                    for page in range(1, total_pages + 1):
                        images = convert_from_bytes(file_bytes, first_page=page, last_page=page, fmt="png")
                        if not images:
                            continue
                        page_text = gemini_ocr(images[0]) or ""
                        if page_text.strip():
                            extracted_text += page_text + "\n"
                        if page % 5 == 0:
                            print(f"   ➜ OCR progress: {page}/{total_pages} pages done")
                except Exception as e:
                    print("PDF OCR failed:", e)

        elif lower.endswith((".png", ".jpg", ".jpeg")):
            try:
                image = Image.open(BytesIO(file_bytes)).convert("RGB")
                extracted_text = gemini_ocr(image) or ""
            except Exception as e:
                print("Image OCR failed:", e)
        else:
            return jsonify({"error": "Unsupported file type."}), 400

        if not extracted_text.strip():
            return jsonify({"error": "No readable text found in file."}), 400

        add_to_knowledge_base(extracted_text, filename, doc_id, file_bytes)
        print(f"Successfully embedded '{filename}' to KB as {doc_id}")
        return jsonify({
            "file_id": doc_id,
            "message": f"'{filename}' successfully stored in knowledge base."
        })
    except Exception as e:
        print("Upload Error:", e)
        return jsonify({"error": f"Upload failed: {e}"}), 500
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            data = request.get_json(silent=True) or {}
            user_input = (data.get("user_input") or "").strip()
            doc_id = (data.get("file_id") or "").strip() or None
            if not user_input and not doc_id:
                return jsonify({"error": "Please type a question or upload a file."}), 400
            kb_context = retrieve(user_input or "context", k=5, doc_id=doc_id)
            combined_prompt = f"{system_prompt}\n\nUser asked: {user_input}\n\nContext:\n{kb_context}"
            resp = model.generate_content(combined_prompt)
            answer = (getattr(resp, "text", "") or "").strip() or "No response."
            return jsonify({"message": answer})
        except Exception as e:
            print("Query Error:", e)
            return jsonify({"error": f"Query failed: {e}"}), 500
    return render_template("index.html")
if __name__ == "__main__":
    port = int(os.environ.get("PORT",8080)) #railways give PORT automatically
    app.run(host="0.0.0.0", port=port)
