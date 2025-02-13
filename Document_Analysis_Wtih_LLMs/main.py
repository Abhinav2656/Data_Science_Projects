import pdfplumber
from transformers import pipeline
import nltk

# Step 1: Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    extracted_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n"
    return extracted_text

pdf_path = "google_terms_of_service_en_in.pdf"
extracted_text = extract_text_from_pdf(pdf_path)

print("\n---- Extracted Text ----\n")
print(extracted_text[:1000])  # Print a snippet for readability

# Step 2: Summarization
def summarize_text(text, max_length=200):
    summarizer = pipeline("summarization", model="t5-small")
    return summarizer(text, max_length=max_length, min_length=50, do_sample=False)[0]['summary_text']

summary = summarize_text(extracted_text)

print("\n---- Summary ----\n")
print(summary)

# Step 3: Split Document into Sentences and Passages
nltk.download('punkt')

def split_into_passages(text, passage_length=3):
    sentences = nltk.sent_tokenize(text)
    passages = [" ".join(sentences[i:i+passage_length]) for i in range(0, len(sentences), passage_length)]
    return passages

passages = split_into_passages(summary)

for i, passage in enumerate(passages):
    print(f"\n--- Passage {i+1} ---\n{passage}")

# Step 4: Generate Questions from Passages Using LLM
def generate_questions(passage):
    question_generator = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")  # Fixed model name
    return question_generator(f"generate question: {passage}")[0]['generated_text']

# Step 5: Answer the Generated Questions Using a QA Model
def answer_questions(passage, question):
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    return qa_pipeline(question=question, context=passage)['answer']

# Step 6: Process Passages, Generate Questions & Answers
for passage in passages:
    question = generate_questions(passage)  # Generate the question once
    answer = answer_questions(passage, question)  # Pass the question for answering

    print(f"\nGenerated Question: {question}")
    print(f"Answer: {answer}")

# Step 7: Main Execution (Avoid Repeating Code)
if __name__ == "__main__":
    pdf_path = "google_terms_of_service_en_in.pdf"
    extracted_text = extract_text_from_pdf(pdf_path)

    if extracted_text.strip():
        summary = summarize_text(extracted_text)
        passages = split_into_passages(summary)

        for i, passage in enumerate(passages):
            print(f"\n--- Passage {i+1} ---\n{passage}")

            question = generate_questions(passage)
            answer = answer_questions(passage, question)

            print(f"\nGenerated Question: {question}")
            print(f"Answer: {answer}")

    else:
        print("No text found in the document.")
