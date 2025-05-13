import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn


@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

    def forward(self, input1, input2):
        return torch.nn.functional.cosine_similarity(input1, input2, dim=1)

siamese_network = SiameseNetwork()


def get_bert_embeddings(text):
    if isinstance(text, str):
        text = [text]
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)


def compare_questions(q1, q2):
    emb1 = get_bert_embeddings(q1)
    emb2 = get_bert_embeddings(q2)
    similarity = siamese_network(emb1, emb2)
    return similarity.item()

predefined_answers = {
    "What is Zewail City?": "Zewail City of Science and Technology is a nonprofit, independent educational and research institution in Egypt founded by Nobel Laureate Dr. Ahmed Zewail. It aims to provide world-class education and cutting-edge research in science, engineering, and technology to help drive national development and global innovation.",
    "What are the admission requirements for Zewail City?": "Applicants must hold a high school diploma (Thanaweya Amma, IGCSE, American Diploma, or equivalent). They should have strong academic records, particularly in math and science subjects. Admission may require passing written exams and an interview.",
    "How can I apply to Zewail City?": "You can apply online through the official Zewail City website by filling out the admission form and uploading all required documents. The admission portal opens during specific periods announced on the site.",
    "What undergraduate programs are offered at Zewail City?": "Zewail City offers undergraduate programs in Nanoscience, Biomedical Sciences, Materials Science, Engineering, and Computer Science.",
    "Are there graduate programs available at Zewail City?": "Yes, Zewail City offers graduate programs, including Master's and Ph.D. degrees in various fields such as Artificial Intelligence Engineering, Nanotechnology, Biomedical Sciences, and more.",
    "What is the language of instruction at Zewail City?": "The primary language of instruction at Zewail City is English.",
    "Does Zewail City offer scholarships?": "Yes, Zewail City offers both merit-based and need-based scholarships to support outstanding and financially challenged students. Applicants can apply for scholarships during the admission process.",
    "What is the tuition fee for undergraduate programs?": "Tuition fees are calculated based on the number of registered credit hours in each academic semester. The fees include access to laboratory supplies, library resources, and career advising services.",
    "What is the tuition fee for graduate programs?": "For the academic year 2024/2025, the tuition fee per credit hour is 3,740 EGP for Egyptian students and 350 USD for non-Egyptian students.",
    "What are the admission deadlines?": "Admission deadlines are updated each academic year and are announced on the official website. It's important to regularly check the site to avoid missing the deadlines.",
    "Do international students qualify for admission?": "Yes, international students are welcome to apply to Zewail City. They must meet the same academic and language requirements and follow the same application process.",
    "What entrance exams are required for admission?": "Applicants may be required to take standardized tests in mathematics, physics, chemistry, and logical reasoning. The exact requirements vary depending on the chosen program.",
    "How do I contact the admission office?": "You can contact the Zewail City admission office by email at admissions@zewailcity.edu.eg or by phone at +2 02 385 40 398.",
    "Is there student accommodation available?": "Yes, Zewail City provides on-campus accommodation. The housing complex contains fully equipped double rooms with necessary amenities and security.",
    "What facilities are available on campus?": "The campus includes state-of-the-art laboratories, libraries, lecture halls, recreational areas, and student housing, all designed to support a comprehensive educational experience.",
    "Are there student clubs and organizations at Zewail City?": "Yes, Zewail City hosts various student clubs and organizations that cater to a wide range of interests, promoting extracurricular engagement and personal development.",
    "Does Zewail City offer internships or practical training?": "Yes, Zewail City provides opportunities for internships and practical training through its Career Advising and Training Services (CATS) to enhance students' professional skills.",
    "What support services are available for students?": "Zewail City offers academic advising, career counseling, mental health services, and various workshops to support student well-being and success.",
    "How can I schedule a campus visit?": "Prospective students can schedule a campus visit by contacting the admissions office via email or phone. Details are available on the official website.",
    "What is the process for transferring to Zewail City from another university?": "Transfer applicants must meet specific criteria and submit their academic transcripts for evaluation. The admissions office provides detailed guidelines on the transfer process.",
    "Does Zewail City offer online courses?": "Currently, Zewail City focuses on in-person instruction, but some programs may offer online components. Check the official website for the most up-to-date information.",
    "What are the housing options for students?": "Students can choose from on-campus housing facilities or seek assistance from the Student Affairs department for off-campus accommodation options.",
    "Is there a meal plan available for students?": "Yes, Zewail City offers meal plans for students residing on campus, providing access to dining facilities that cater to diverse dietary needs.",
    "What is the application fee for admission?": "The application fee is specified on the admissions portal and may vary for domestic and international applicants. Refer to the official website for current rates.",
    "Are there any entrance scholarships for new students?": "Zewail City offers entrance scholarships based on academic merit and other criteria. Applicants are considered automatically during the admission process.",
    "How can I apply for financial aid?": "Students can apply for financial aid by submitting the required documents during the admission process. The Financial Aid Office evaluates applications based on need and merit.",
    "What is the duration of undergraduate programs at Zewail City?": "Most undergraduate programs at Zewail City are designed to be completed in four years, depending on the course load and program requirements.",
    "Does Zewail City accept credit transfers from other institutions?": "Credit transfers are evaluated on a case-by-case basis. Applicants must provide detailed course descriptions and transcripts for assessment.",
    "What is the academic calendar at Zewail City?": "The academic year typically consists of two main semesters. Specific dates and schedules are published annually on the university's website.",
    "Are there opportunities for study abroad programs?": "Zewail City may offer study abroad opportunities through its international partnerships. Interested students should consult the International Relations Office.",
    "What are the criteria for maintaining a scholarship?": "Scholarship recipients must meet specific academic performance standards, such as maintaining a minimum GPA, as outlined in their scholarship agreement.",
    "Does Zewail City provide support for students with disabilities?": "Yes, Zewail City is committed to inclusivity and provides support services to accommodate students with disabilities.",
    "What is the process for obtaining a student visa for international students?": "International students must apply for a student visa through the Egyptian consulate in their home country after receiving an admission offer from Zewail City.",
    "Are there language proficiency requirements for non-native English speakers?": "Yes, non-native English speakers may be required to submit TOEFL or IELTS scores to demonstrate English proficiency.",
    "What is the policy on part-time study?": "Part-time study options may be available for certain programs. Interested students should consult the admissions office for eligibility and program structure.",
    "Does Zewail City offer evening or weekend classes?": "Course schedules vary by program. Some classes may be offered in the evening or on weekends to accommodate different student needs.",
    "What is the process for deferring admission?": "Admitted students wishing to defer their enrollment must submit a formal request to the admissions office, which will be reviewed on a case-by-case basis.",
    "Are there opportunities for community service or volunteering?": "Zewail City encourages student involvement in community service and offers various programs and partnerships to facilitate volunteering.",
    "What career services are available to students?": "The Career Advising and Training Services (CATS) department provides career counseling, internship placements, resume workshops, and job search assistance.",
    "Does Zewail City have an alumni network?": "Yes, Zewail City maintains an active alumni network that offers networking opportunities, events, and support for graduates.",
    "What are the library resources available to students?": "The university library offers a vast collection of books, journals, and digital resources to support academic research and study.",
    "Are there opportunities for entrepreneurship and innovation?": "Zewail City fosters entrepreneurship through incubators, innovation labs, and partnerships that support student-led startups and projects.",
    "What is the grading system at Zewail City?": "Zewail City uses a GPA-based grading system, with detailed policies outlined in the student handbook.",
    "How does Zewail City support student mental health?": "The university provides counseling services, wellness programs, and workshops to support the mental health and well-being of students.",
    "What transportation options are available for students?": "Zewail City offers transportation services with designated pickup and drop-off points across Cairo and Giza for students' convenience.",
    "Are there any orientation programs for new students?": "Yes, Zewail City conducts orientation sessions to help new students acclimate to the campus environment and academic expectations.",
    "What is the policy on academic integrity?": "Zewail City upholds strict academic integrity policies, and violations such as plagiarism or cheating are subject to disciplinary action.",
    "Does Zewail City offer language courses?": "Language courses may be offered as part of the curriculum or as extracurricular options. Availability varies by semester.",
    "How can students provide feedback on courses and teaching?": "Students can provide feedback through course evaluations and surveys conducted by the university to enhance teaching quality.",
    "What is the student-to-faculty ratio at Zewail City?": "Zewail City maintains a low student-to-faculty ratio to ensure personalized attention and support for each student.",
    "Are there opportunities for research at the undergraduate level?": "Yes, undergraduate students at Zewail City are encouraged to participate in research projects, fostering a hands-on learning environment.",
    "What is the process for obtaining official transcripts?": "Students can request official transcripts through the Registrar's Office. Processing times and fees are detailed on the university's website.",
    "Does Zewail City offer dual-degree programs?": "Information about dual-degree programs is available on the official website. Interested students should consult the academic advisor.",
    "Are there any special programs for gifted students?": "Zewail City offers programs and workshops designed to challenge and engage gifted students. Details are available through the Student Affairs Office.",
    "What is the policy on attendance?": "Attendance policies are outlined in the student handbook and may vary by course. Regular attendance is generally expected.",
    "Does Zewail City have a dress code?": "While there is no formal dress code, students are expected to dress appropriately and maintain a professional appearance on campus.",
    "What is the process for changing majors?": "Students wishing to change majors must consult their academic advisor."
}

# Streamlit UI
st.title("🎓 Zewail City Chatbot")
st.write("Ask me anything about Zewail City!")

user_input = st.text_input("Your question:")

if user_input:
    SIMILARITY_THRESHOLD = 0.8  
    best_answer = "Sorry, I don’t have an answer for that. Please try rephrasing."
    highest_score = -1
    best_question = None

    for question, answer in predefined_answers.items():
        score = compare_questions(user_input, question)
        if score > highest_score:
            highest_score = score
            best_answer = answer
            best_question = question


    if highest_score < SIMILARITY_THRESHOLD:
        best_answer = "Sorry, I don’t have an answer for that. Please try rephrasing."

    st.subheader("💬 Answer")
    st.write(best_answer)