import os
import numpy as np
import faiss
import streamlit as st
import google.generativeai as genai
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_google_genai import GoogleGenerativeAI

# Configuration
COHERE_API_KEY = "01YL5TQxIsz1SXKTffDBWzXuX6M5Yf8HDxvIfe2G"
GOOGLE_API_KEY = "AIzaSyBPg52dZBy6mECGJiMmaiLw4l4tNLvkILY"

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=40,
            length_function=len,
            is_separator_regex=False,
        )
        self.embeddings_model = CohereEmbeddings(
            cohere_api_key=COHERE_API_KEY, 
            model='embed-english-v2.0'
        )

    def process_pdf(self, pdf_file):
        """Process uploaded PDF file"""
        # Save uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.getbuffer())

        # Load and process the PDF
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        
        # Generate embeddings
        texts = [doc.page_content for doc in chunks]
        embeddings = self.embeddings_model.embed_documents(texts)
        
        # Create FAISS index
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        embeddings_array = np.array(embeddings).astype('float32')
        index.add(embeddings_array)

        # Clean up
        os.remove("temp.pdf")
        
        return texts, index

class QuestionAnswerer:
    def __init__(self, embeddings_model):
        self.embeddings_model = embeddings_model
        genai.configure(api_key=GOOGLE_API_KEY)
        self.llm = GoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.7
        )

    def get_answer(self, question: str, texts: list, index: faiss.IndexFlatL2):
        """Get answer for the given question"""
        # Get embedding for the question
        question_embedding = self.embeddings_model.embed_documents([question])[0]
        question_embedding = np.array(question_embedding).astype('float32').reshape(1, -1)
        
        # Search for similar chunks
        distances, indices = index.search(question_embedding, 3)
        relevant_chunks = [texts[i] for i in indices[0]]

        # Create prompt
        prompt = f"""Based on the following context, please answer the question that is asked by
        the user related to the context that is provided.
        
        

        Context:
        {' '.join(relevant_chunks)}

        Question: {question}

        Answer:"""

        # Generate response
        response = self.llm.invoke(prompt)
        return response, relevant_chunks

def main():
    st.title("Document Q&A System")
    st.write("Upload a PDF document and ask questions about it!")

    # Initialize processors
    doc_processor = DocumentProcessor()
    qa_system = QuestionAnswerer(doc_processor.embeddings_model)

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        with st.spinner('Processing document...'):
            texts, index = doc_processor.process_pdf(uploaded_file)
            st.success('Document processed successfully!')

        # Question input
        question = st.text_input("Ask a question about the document:")
        
        if question:
            with st.spinner('Generating answer...'):
                answer, context = qa_system.get_answer(question, texts, index)
                
                st.write("### Answer:")
                st.write(answer)
                
                with st.expander("View Source Context"):
                    for i, chunk in enumerate(context, 1):
                        st.write(f"Chunk {i}:")
                        st.write(chunk)
                        st.write("---")

if __name__ == "__main__":
    main()