from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM

# Resume data
resume_chunks = [
    "Education : MBA Data Science and Artificial Intelligence, Indian Institute of Technology, Mandi (Expected 2026). BS Data Science and Applications, Indian Institute of Technology, Madras, CGPA: 8.35 (2024)., BS Physics, Math, Computer Science, Bangalore University, CGPA: 7.04 (2021).",
    "Courses : Data Structures and Algorithms, AI Search Methods, Multiagent Systems, Mathematics and Statistics for Data Science (Basic+Advanced), Data Mining, Machine Learning Techniques, Deep Learning, Large Language Models, Computer Vision, Software Engineering, Managerial Economics, Game Theory, Business Analytics, Decision Making, Data Visualization.",
    "Technical Skills: Python, Java, JavaScript, Haskell, Scala, Prompt Engineering, Flask, VueJS, PyTorch, Scikit-Learn, NumPy, Pandas, SQL, NLP, HuggingFace, LangChain, LLM Fine-tuning, MS Office, Power BI.",
    "Soft Skills: Problem-Solving Skills, First Principles Thinking.",
    "Work Experience : Senior Technical Assistant – B at DRDO, Ministry of Defence, Government of India, Mysore (March 2024 – July 2024),Worked in the Food Engineering and Packaging Technology Division at DFRL (RnD).,Cracked Group ‘B’ DRDO Defence Exam (CEPTAM 10).",
    "Internship : Data Scientist Intern, GITAA Pvt. Ltd. Built an AI-powered PDF chatbot enabling interactive queries on PDF content via Llama3.2-vision models.",
    "Internship : Streamlined PDF processing with vector-based retrieval and applied conversational AI for context-driven responses using LangChain and HuggingFace Embeddings via LLMs like Llama/Phi/Mistral models.",
    "Internship : Developed a sports commentary generator using NLP techniques, template-based narratives, fine-tuned GPT-2, and SportsBERT for predictive text.",
    "Internship : Integrated contextual analysis to dynamically create real-time match insights based on game scenarios.",
    "Projects : Implemented transformer-based models, including encoder-decoder layers with multi-head attention, for Tamil-English translation and explored BERT’s MLM and GPT’s CLM objectives.",
    "Projects : Developed a U-Net-based computer vision model to segment images and identify flood-affected areas, enhancing disaster management with an IoU score of 65%.",
    "Projects : Developed a data-driven strategy to enhance sales and profitability for a handmade fashion jewelry business.",
    "Projects : Engineered a project management tool using Vue3 and Flask RESTful frameworks. Earned an S grade (91%) evaluated by an industry professional.",
    "Certifications : Quantum Computing Cohort 2 (ongoing), Center for Outreach and Digital Education (CODE), IIT Madras., Emotional Intelligence, NPTEL+., Theory of Computation, NPTEL+.",
    "Awards : Recipient of the Ram Shriram Merit Scholarship at IIT Madras., Awarded the Verizon Scholarship at IIT Madras for maintaining a CGPA above 8., Ranked 7th out of 990 students in Machine Learning Practice Capstone Project."
]

# Load the fine-tuned model
tokenizer = AutoTokenizer.from_pretrained("chatbot/model")
model = AutoModelForCausalLM.from_pretrained("chatbot/model")


# Step 1: Embedding using HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 2: Create the vectorstore using FAISS
vectorstore = FAISS.from_texts([doc.replace("\n", " ") for doc in resume_chunks], embeddings)

def retrieve_relevant_chunks(query, k=3):
    """Retrieve top-k relevant chunks based on the query."""
    try:
        docs = vectorstore.similarity_search_with_relevance_scores(query, k=k)
        return [doc[0].page_content for doc in docs]
    except Exception as e:
        print(f"Error in retrieval: {str(e)}")
        return []


def generate_response(query):
    """Generate factual responses strictly from resume content."""
    relevant_chunks = retrieve_relevant_chunks(query, k=2)
    if not relevant_chunks:
        return {"response": "I don't have that information in my resume."}

    context = "\n".join(relevant_chunks)

    prompt = f"""Provide a direct answer using ONLY the information given in the context. If the specific information is not present in the context, respond with "That information is not in my resume."

Context: {context}

Question: {query}
Answer: Let me answer based on my resume - """

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(
            **inputs,
            max_new_tokens=65,
            min_new_tokens=10,
            no_repeat_ngram_size=3,
            temperature=0.2,  # Reduced for more conservative responses
            top_p=0.75,  # More restrictive sampling
            top_k=30,
            do_sample=False  # Deterministic generation
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("Answer: Let me answer based on my resume - ")[-1].strip()
        return {"response": answer}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"response": "An error occurred."}
