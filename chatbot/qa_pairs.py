import json

# resume_data = "Education: MBA Data Science and Artificial Intelligence, Indian Institute of Technology, Mandi (Expected 2026). BS Data Science and Applications, Indian Institute of Technology, Madras, CGPA: 8.35 (2024). BS Physics, Math, Computer Science, Bangalore University, CGPA: 7.04 (2021). Skills: Data Structures and Algorithms, AI Search Methods, Multiagent Systems, Mathematics and Statistics for Data Science (Basic+Advanced), Data Mining, Machine Learning Techniques, Deep Learning, Large Language Models, Computer Vision, Software Engineering, Managerial Economics, Game Theory, Business Analytics, Decision Making, Data Visualization. SKILLS: Python, Java, JavaScript, Haskell, Scala, Prompt Engineering, Flask, VueJS, PyTorch, Scikit-Learn, NumPy, Pandas, SQL, NLP, HuggingFace, Langchain, LLM Finetuning, MS Office, Power BI, Problem-Solving Skills, First Principles Thinking. Work Experience: Senior Technical Assistant – B at DRDO, Ministry of Defence, Government of India, Mysore (March 2024 – July 2024). Worked in the Food Engineering and Packaging Technology Division at DFRL (RnD). Cracked Group ‘B’ DRDO Defence Exam (CEPTAM 10). Internships: Data Scientist Intern, GITAA Pvt. Ltd. Built an AI-powered PDF chatbot enabling interactive queries on PDF content via Llama3.2-vision models. Streamlined PDF processing with Vector-based Retrieval and applied conversational AI for context-driven responses using LangChain and HuggingFace Embeddings via LLMs like Llama/Phi/Mistral models. Developed a sports commentary generator using NLP techniques including template-based narratives, Fine-tuned GPT-2, and SportsBERT model for embeddings for predictive text. Integrated contextual analysis to dynamically create real-time match insights based on game scenarios. Projects: Implemented various transformer-based models and components, including encoder and decoder layers with multi-head attention, for tasks such as Tamil-English translation, and explored BERT’s masked language modelling (MLM) and GPT’s causal language modelling (CLM) objectives. Developed a U-Net-based computer vision model to accurately segment images and identify flood-affected areas, enhancing disaster management by providing critical information. Achieved an IoU score of 65%. Developed a strategy and crafted a business proposal using data analytics to enhance sales and profitability for a local handmade fashion jewelry business. Engineered a project management tool via Vue3 and Flask RESTful frameworks. This project was part of a course in my undergrad, the project component resulted in S grade with 91% which was evaluated by an industry professional via viva. Certifications: Quantum Computing Cohort 2 (ongoing) from Center for Outreach and Digital Education (CODE), IIT Madras, 6-month program in Quantum Computing and Quantum Machine Learning. Emotional Intelligence from NPTEL+. Theory of Computation from NPTEL+. Achievements: Recipient of the Ram Shriram Merit Scholarship at IIT Madras. Awarded the Verizon Scholarship at IIT Madras for maintaining a CGPA above 8. Ranked 7th out of 990 students in Machine Learning Practice Capstone Project."
resume_data = {
    "Education": """
    MBA Data Science and Artificial Intelligence, Indian Institute of Technology, Mandi (Expected 2026).
    BS Data Science and Applications, Indian Institute of Technology, Madras, CGPA: 8.35 (2024).
    BS Physics, Math, Computer Science, Bangalore University, CGPA: 7.04 (2021).
    """,

    "Skills": """
    Data Structures and Algorithms, AI Search Methods, Multiagent Systems, Mathematics and Statistics for Data Science (Basic+Advanced) Data Mining, Machine Learning Techniques, Deep Learning, Large Language Models, Computer Vision Software Engineering, Managerial Economics, Game Theory, Business Analytics, Decision Making, Data Visualization SKILLS Python, Java, JavaScript, Haskell, Scala, Prompt Engineering, Flask, VueJS PyTorch, Scikit-Learn, NumPy, Pandas, SQL, NLP, HuggingFace, Langchain, LLM Finetuning MS Office, Power BI, Problem-Solving Skills, First Principles Thinking
    """,

    "Work Experience": """
    Senior Technical Assistant – B at DRDO, Ministry of Defence, Government of India, Mysore (March 2024 – July 2024).
    Worked in the Food Engineering and Packaging Technology Division at DFRL (RnD).
    Cracked Group ‘B’ DRDO Defence Exam (CEPTAM 10).
    """,

    "Internships": """
    Data Scientist Intern, GITAA Pvt. Ltd.
    Built an AI-powered PDF chatbot with enabling interactive queries on PDF content via Llama3.2-vision models. Streamlined PDF processing with Vector-based Retrieval and applied conversational AI for context-driven responses, using LangChain and HuggingFace Embeddings via LLMs like Llama/Phi/Mistral models
    Developed a sports commentary generator using NLP techniques including template-based narratives, Fine-tuned GPT-2, and SportsBERT model for embeddings for predictive text. Integrated contextual analysis to dynamically create real-time match insights based on game scenarios.
    """,

    "Projects": """
    Implemented various transformer-based models and components, including encoder and decoder layers with multi-head attention, for tasks such as Tamil-English translation, and explored BERT’s masked language modelling (MLM) and GPT’s causal language modelling (CLM) objectives. 
    Developed a U-Net-based computer vision model to accurately segment images and identify flood-affected areas, enhancing disaster management by providing critical information. Achieved an IoU score of 65%.
    Developed a strategy and crafted a business proposal using data analytics to enhance sales and profitability for a local handmade fashion jewelry business.
    Engineered a project management tool via Vue3 and Flask RESTful frameworks. This project was part of a course in my undergrad, the project component resulted in S grade with 91% which was evaluated by an industry professional via viva.
    """,

    "Certifications": """
    Quantum Computing Cohort 2 (ongoing) from Center for Outreach and Digital Education (CODE), IIT Madras, 6-month program in Quantum Computing and Quantum Machine Learning.
    Emotional Intelligence from NPTEL+.
    Theory of Computation from NPTEL+.
    """,

    "Achievements": """
    Recipient of the Ram Shriram Merit Scholarship at IIT Madras.
    Awarded the Verizon Scholarship at IIT Madras for maintaining a CGPA above 8.
    Ranked 7th out of 990 students in Machine Learning Practice Capstone Project.
    """
}

qa_pairs = [
    {"question": "What is your educational background?", "answer": resume_data["Education"]},
    {"question": "What technical skills do you have?", "answer": resume_data["Skills"]},
    {"question": "What work experience do you have?", "answer": resume_data["Work Experience"]},
    {"question": "What internships have you completed?", "answer": resume_data["Internships"]},
    {"question": "What projects have you worked on?", "answer": resume_data["Projects"]},
    {"question": "What certifications do you have?", "answer": resume_data["Certifications"]},
    {"question": "What are your achievements?", "answer": resume_data["Achievements"]},
{"question": "What inspired you to choose a career in data science and AI?",
     "answer": "My fascination with problem-solving and technology motivated me to pursue data science and AI. My educational journey, starting with Physics and Math, laid a strong foundation in analytical thinking, which I later expanded with hands-on experience in machine learning and AI applications."},
    {"question": "How do you manage to balance technical and managerial skills during your MBA?",
     "answer": "I focus on leveraging both skill sets in tandem by applying technical skills to solve real-world business problems while understanding the strategic implications through my MBA coursework."},
    {"question": "Can you share a memorable learning experience from your time at IIT Madras?",
     "answer": "Participating in the Machine Learning Practice Capstone Project was a highlight. It allowed me to apply theoretical concepts to practical problems and finish in the top 1%, which was both challenging and rewarding."},
    {"question": "How do you approach building AI systems for real-world applications?",
     "answer": "My approach starts with understanding the problem’s context, followed by data collection, preprocessing, and exploratory analysis. I then design models iteratively, incorporating feedback and refining for performance. For example, while building the PDF chatbot, I optimized retrieval systems and embeddings for contextual responses."},
    {"question": "What were the key challenges in developing the sports commentary generator project?",
     "answer": "The main challenges were ensuring natural language fluency and integrating real-time match scenarios. Fine-tuning GPT-2 with SportsBERT embeddings helped create coherent and contextually relevant narratives."},
    {"question": "How did you implement flood segmentation using U-Net, and what was the biggest hurdle?",
     "answer": "The project involved training a U-Net model on satellite imagery to identify flood-affected regions. A significant challenge was obtaining labeled data, which I addressed by using transfer learning and data augmentation techniques."},
    {"question": "What role does LangChain play in your chatbot project?",
     "answer": "LangChain was instrumental in orchestrating LLMs for contextual processing and retrieval-based query handling. It helped seamlessly integrate embeddings and conversational logic for enhanced user interaction."},
    {"question": "What was the most significant contribution you made during your time at DRDO?",
     "answer": "At DRDO, I contributed to the Food Engineering and Packaging Technology Division by optimizing R&D processes. It was a valuable experience in applying engineering principles to defense needs."},
    {"question": "How has your internship at GITAA Pvt. Ltd. shaped your skills?",
     "answer": "It deepened my understanding of deploying advanced AI models in practical scenarios. I also learned the importance of balancing innovation with efficiency, especially when integrating LLMs into production."},
    {"question": "What lessons did you learn from working on the handmade jewelry business analytics project?",
     "answer": "The project taught me how data analytics can drive business strategy. By identifying key sales trends and customer preferences, I proposed actionable steps that boosted profitability."},
    {"question": "How has the Emotional Intelligence certification impacted your professional growth?",
     "answer": "It enhanced my ability to work effectively in teams, resolve conflicts, and understand others’ perspectives—key skills for both leadership and collaboration."},
    {"question": "Why did you choose to pursue Quantum Computing, and how do you plan to apply it?",
     "answer": "Quantum computing intrigues me for its potential to solve problems beyond the scope of classical systems. I aim to explore its applications in optimization and machine learning."},
    {"question": "How do you stay updated with the latest trends in AI and machine learning?",
     "answer": "I regularly participate in courses, workshops, and hackathons. Additionally, I follow research publications and apply new techniques to my projects whenever possible."},
    {"question": "What does receiving the Ram Shriram Merit Scholarship mean to you?",
     "answer": "It’s a recognition of my hard work and academic commitment. It motivates me to continue excelling in my field and take on challenges with confidence."},
    {"question": "Which of your achievements are you most proud of and why?",
     "answer": "Ranking 7th out of 990 in the Machine Learning Capstone Project stands out. It validated my technical expertise and ability to work under pressure."},
    {"question": "How do you apply first principles thinking to solve problems?",
     "answer": "I break down problems into fundamental components, analyze them from the ground up, and build solutions without relying on preconceived notions. This method helped me significantly in my projects, such as the Kanban board application."},
    {"question": "If you had to explain your career path to someone without a technical background, how would you do it?",
     "answer": "I’d say I combine the logic of a detective, the creativity of an artist, and the precision of a scientist to turn raw data into meaningful insights that solve real-world problems."},
    {"question": "What motivates you to keep learning and working on challenging projects?",
     "answer": "The excitement of solving complex problems and the potential impact of my work on society drive me. Seeing ideas come to life keeps me inspired."},
    {"question": "What’s your vision for the future of AI in businesses?",
     "answer": "AI will be integral to decision-making, enabling businesses to innovate faster, personalize customer experiences, and solve global challenges like climate change and healthcare."},
    {"question": "What advice would you give to someone starting in AI and data science?",
     "answer": "Focus on building a strong foundation in math and coding, and never stop experimenting with projects. Practical experience and curiosity are key to mastering this field."},
{"question": "Why did you choose IIT Mandi for your MBA in Data Science and AI?",
     "answer": "IIT Mandi's interdisciplinary approach and focus on AI innovation align perfectly with my career goals. Its emphasis on cutting-edge research and practical applications in data science made it an ideal choice."},
    {"question": "What are some of the unique skills you’ve developed during your internships?",
     "answer": "During my internships, I enhanced my expertise in LangChain, LLM fine-tuning, and real-time AI applications. I also developed a deeper understanding of conversational AI and predictive analytics in sports and business contexts."},
    {"question": "How did your dual degrees from IIT Madras and Bangalore University shape your technical and analytical abilities?",
     "answer": "My undergraduate studies at Bangalore University laid a foundation in math, physics, and computer science, while IIT Madras honed my data science expertise, combining theory with hands-on projects."},
    {"question": "What lessons have you learned from managing academic and professional responsibilities simultaneously?",
     "answer": "Balancing academics and work taught me time management, adaptability, and the importance of prioritization. These skills have been instrumental in maintaining consistent performance across projects and coursework."},
    {"question": "What impact do you believe your work on flood segmentation has on disaster management?",
     "answer": "By accurately identifying flood-affected areas with a U-Net model, my work contributes to quicker disaster response and resource allocation, potentially saving lives and minimizing damage."},
    {"question": "How do you approach collaboration on interdisciplinary projects?",
     "answer": "I leverage my technical skills to complement domain-specific expertise from team members. Clear communication and aligning goals have been key to success in projects like the PDF chatbot and sports commentary generator."},
    {"question": "What drives your passion for applying AI to real-world challenges?",
     "answer": "The potential to create meaningful impact motivates me. Whether it's enhancing disaster management or optimizing business operations, I find immense satisfaction in solving practical problems with AI."},
    {
        "question": "What motivated you to pursue an MBA in Data Science and AI, and how does it complement your technical background?",
        "answer": resume_data["Education"]
    },
    {
        "question": "Can you share some of the most impactful technical projects you’ve worked on and how they demonstrate your skills in AI, machine learning, and software development?",
        "answer": resume_data["Projects"]
    },
    {
        "question": "What are the most valuable skills you've acquired during your educational journey, and how do they make you a strong candidate for data science roles?",
        "answer": resume_data["Skills"]
    },
    {
        "question": "How did your experience at DRDO contribute to your understanding of technical problem-solving, and what challenges did you overcome?",
        "answer": resume_data["Work Experience"]
    },
    {
        "question": "How did your internship at GITAA Pvt. Ltd. enhance your knowledge of AI/ML applications, and which project are you most proud of?",
        "answer": resume_data["Internships"]
    },
    {
        "question": "What role do you believe your work on the Kanban Board Application project played in developing your full-stack development capabilities?",
        "answer": resume_data["Projects"]
    },
    {
        "question": "How do your certifications in Quantum Computing and Emotional Intelligence add value to your profile as a data scientist?",
        "answer": resume_data["Certifications"]
    },
    {
        "question": "Which achievement has had the most significant impact on your career, and how did it shape your aspirations?",
        "answer": resume_data["Achievements"]
    },
    {
        "question": "What is your approach to problem-solving in data-driven projects, and how do you apply first principles thinking in your work?",
        "answer": resume_data["Skills"]
    },
    {
        "question": "What is the most exciting project you've worked on that blends AI, machine learning, and business impact?",
        "answer": resume_data["Projects"]
    },
    {
        "question": "What challenges did you face while working on the flooded area detection project using computer vision, and how did you overcome them?",
        "answer": resume_data["Projects"]
    },
    {
        "question": "How do your academic and professional experiences prepare you for a future in AI-driven decision-making and strategic leadership?",
        "answer": resume_data["Education"] + " " + resume_data["Work Experience"]
    }
]

# Step 3: Save the Q&A pairs to a JSON file
with open("resume_qa_pairs.json", "w") as f:
    json.dump(qa_pairs, f, indent=4)

print("Creative and curative Q&A pairs have been generated and saved to 'resume_qa_pairs.json'.")
