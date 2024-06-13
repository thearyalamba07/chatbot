import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
import os
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# Set your OpenAI API key here
os.environ['OPENAI_API_KEY'] = 'sk-proj-NrVEIpuTABu55E6hspfzT3BlbkFJzsO8XJkHrw6YVEo4shaW'

with st.sidebar:
    st.title('Career Bot')
    st.markdown('''This is a career counsellor, who will not burn your pocket, but help you find the answers to your problems''')
    add_vertical_space(5)
    st.write('Made to give you guidance by Arya Lamba')

def main():
    st.header("Silly-Cone Solutions")
    st.write("Ask away all the doubts that you may have")

    # Upload a PDF
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    # Get user's query
    query = st.text_input("Ask questions about your PDF file:")

    if pdf is not None:
        st.write(pdf.name)
        pdf_reader = PdfReader(pdf)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()

        st.write("Extracted text from the PDF")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        st.write("Text split into chunks")

        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai_api_key is None:
            st.error("Please set the OPENAI_API_KEY environment variable.")
            return

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vector_store = pickle.load(f)
            st.write('Embeddings Loaded from the Disk')
        else:
            try:
                vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(vector_store, f)
                st.write(f'Embeddings saved to {store_name}.pkl')
            except Exception as e:
                st.error(f"Error in creating embeddings: {e}")
                return

        if query:
            try:
                docs = vector_store.similarity_search(query=query, k=3)
                llm = OpenAI(temperature=0)
                chain = load_qa_chain(llm=llm, chain_type="stuff")
                answer = chain.run(input_documents=docs, question=query)
                st.write("Answer:")
                st.write(answer)
            except Exception as e:
                st.error(f"Error in similarity search or QA chain: {e}")

if __name__ == '__main__':
    main()