
import streamlit as st



def chat_with_documents():

        import streamlit as st

        from langchain.vectorstores import FAISS
        import base64
        #from PyPDF2 import PdfReader
        from langchain.document_loaders import PyPDFLoader
        import torch
        import os
        #from apikey import apikey
        from langchain.document_loaders import TextLoader
        from langchain.indexes import VectorstoreIndexCreator
        from langchain.text_splitter import CharacterTextSplitter
        import time
        from langchain import HuggingFaceHub
        from langchain.embeddings import HuggingFaceInstructEmbeddings
        from langchain.vectorstores import Chroma
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.chains import RetrievalQA
        import textwrap
        import os
        from langchain.document_loaders import DirectoryLoader
        import shutil
        from langchain.embeddings import HuggingFaceBgeEmbeddings
        from deep_translator import GoogleTranslator
        from langdetect import detect
        HUGGINGFACE_API_TOKEN = "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"
        repo_id = "tiiuae/falcon-7b-instruct"
        st.title("TextGenius: Your Research Chat Buddy 📄🤖")
        llm=HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
                                repo_id=repo_id,
                                model_kwargs={"temperature":0.7, "max_new_tokens":700})


        pdfs_directory = "PDFs"
        if not os.path.exists(pdfs_directory):
            os.makedirs(pdfs_directory)
        for file_name in os.listdir(pdfs_directory):
                                file_path = os.path.join(pdfs_directory, file_name)
                                if os.path.isfile(file_path):
                                    os.remove(file_path)
        #Free_Open Source Model

        if 'exit' not in st.session_state:
                st.session_state['exit'] = False
        def typewriter(text: str, speed: float):
            container = st.empty()
            displayed_text = ""

            for char in text:
                displayed_text += char
                container.markdown(displayed_text)
                time.sleep(1/speed)
        def wrap_text_preserve_newlines(text, width=110):
            # Split the input text into lines based on newline characters
            lines = text.split('\n')

            # Wrap each line individually
            wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

            # Join the wrapped lines back together using newline characters
            wrapped_text = '\n'.join(wrapped_lines)

            return wrapped_text
                # do something with the data
        def process_llm_response(llm_response,llm_originalresponse2):
            result_text = wrap_text_preserve_newlines(llm_originalresponse2)
            typewriter(result_text, speed=40)
        def process_source (llm_response):
            st.write('\n\nSources:')
            unique_sources = []
            for source in llm_response["source_documents"]:
                source_name = source.metadata['source']
                if source_name not in unique_sources:
                    unique_sources.append(source_name)
            for source in unique_sources:
                        pdf_display = display_pdf(source)
                        st.markdown(pdf_display, unsafe_allow_html=True)
        def display_pdf(file_path):
            """Display PDF file.

            Args:
                file_path (str): Path to the PDF file.

            Returns:
                str: PDF display in HTML format.
            """

            with open(file_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode("utf-8")
            pdf_display = (
                f'<embed src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf">'
            )
            return pdf_display

        def save_uploaded_pdfs(uploaded_files):
        # Save uploaded PDF files to the "PDFs" directory
            if uploaded_files:
                    for uploaded_file in uploaded_files:
                        original_filename = uploaded_file.name  # Get the original filename
                        unique_filename = original_filename
                        pdf_path = os.path.join(pdfs_directory, unique_filename)

                        # Extract the content from the UploadedFile
                        file_content = uploaded_file.read()

                        with open(pdf_path, "wb") as pdf_file:
                            pdf_file.write(file_content)
                        success_message = st.empty()
                        success_message.success(f"File '{unique_filename}' successfully uploaded.")
                        time.sleep(10)  # Adjust the duration as needed
                        success_message.empty()
        def launchdoc():

            uploaded_files = st.file_uploader("Please upload all your documents at once", type=["pdf"], accept_multiple_files=True)
            original_question = st.text_input("Once uploaded, you can chat with your document. Enter your question here or type exit to end and upload new documents:")
            question = GoogleTranslator(source='auto', target='en').translate(original_question)
            submit_button = st.button('Generate Response 🪄✨')
            if uploaded_files and submit_button:
                                save_uploaded_pdfs(uploaded_files)
                                loader = DirectoryLoader('./PDFs', glob="./*.pdf", loader_cls=PyPDFLoader)
                                with st.spinner('Processing the Documents...'):
                                  documents = loader.load()
                                  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
                                  texts = text_splitter.split_documents(documents)
                                #instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                with st.spinner('Processing Embeddings...'):                                                          #model_kwargs={"device": "cuda"})
                                  model_name = "BAAI/bge-base-en"
                                  encode_kwargs = {'normalize_embeddings': True}
                                  instructor_embeddings =instructor_embeddings = HuggingFaceBgeEmbeddings(
      model_name=model_name,
      model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
      encode_kwargs=encode_kwargs
  )
                                  persist_directory = 'db'

                                  ## Here is the new embeddings being used
                                  embedding = instructor_embeddings
                                  if os.path.exists(persist_directory):
                                      shutil.rmtree(persist_directory)
                                  vectordb = Chroma.from_documents(documents=texts,embedding=embedding, persist_directory=persist_directory)
                                  #vectordb= FAISS.from_documents(texts, embedding)
                                  retriever = vectordb.as_retriever(search_kwargs={"k": 3})

                                  qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                                                      chain_type="stuff",
                                                                      retriever=retriever,
                                                                      return_source_documents=True)
                                # Initial state

                                while st.session_state['exit'] == False:

                                            #question = st.text_input("Once uploaded, you can chat with your document. Enter your question here or type exit to end and upload new documents:", key=f"question_input_{i}")
                                            with st.spinner('Generating Answer...'):

                                                if question.lower() == 'exit':
                                                    st.session_state['exit'] = True

                                                else:
                                                    detected_source_language = detect(original_question)
                                                    chunk_size = 5000
                                                    # Process the question and display the response
                                                    llm_originalresponse = qa_chain(question)
                                                    llm_originalresponse2=str(llm_originalresponse['result'])

                                                    chunks = [llm_originalresponse2[i:i+chunk_size] for i in range(0, len(llm_originalresponse2), chunk_size)]
                                                    translated_chunks = []
                                                    my_translator=GoogleTranslator(source='auto', target=detected_source_language)
                                                    for chunk in chunks:
                                                            translated_chunk = my_translator.translate(chunk)
                                                            translated_chunks.append(translated_chunk)
                                                    llm_originalresponse2=''.join(translated_chunks)
                                                    process_llm_response(llm_originalresponse,llm_originalresponse2)
                                                    process_source (llm_originalresponse)

                                                    break

                                                if st.session_state['exit'] == True:
                                                    output="Thank you for trying our Tool, We hope you liked it"
                                                    typewriter(output, speed=5)
                                                    # Delete files and folders
                                                    for file_name in os.listdir(pdfs_directory):
                                                        file_path = os.path.join(pdfs_directory, file_name)
                                                        if os.path.isfile(file_path):
                                                            os.remove(file_path)

                                                    # Remove "db" directory


                                                    break



            st.warning("⚠️ Please Keep in mind that the accuracy of the response relies on the :red[PDF's Quality] and the :red[prompt's Quality]. Occasionally, the response may not be entirely accurate. Consider using the response as a reference rather than a definitive answer.")


        launchdoc()

def chat_with_website():
        import torch
        import os
        import argparse
        import shutil
        from langchain.document_loaders import YoutubeLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.vectorstores import Chroma
        from langchain.embeddings import OpenAIEmbeddings
        from langchain.chains import RetrievalQA
        from langchain.llms import OpenAI
        import streamlit as st
        from langchain.chat_models import ChatOpenAI
        from langchain import HuggingFaceHub
        from langchain.embeddings import HuggingFaceInstructEmbeddings
        from urllib.parse import urlparse, parse_qs
        from langchain.embeddings import HuggingFaceBgeEmbeddings
        from transformers import pipeline
        import textwrap
        import time
        from deep_translator import GoogleTranslator
        from langdetect import detect
        from langchain.prompts.chat import (ChatPromptTemplate,
                                            HumanMessagePromptTemplate,
                                            SystemMessagePromptTemplate)
        from langchain.document_loaders import WebBaseLoader


        def typewriter(text: str, speed: float):
                    container = st.empty()
                    displayed_text = ""

                    for char in text:
                        displayed_text += char
                        container.markdown(displayed_text)
                        time.sleep(1/speed)
        def wrap_text_preserve_newlines(text, width=110):
                    # Split the input text into lines based on newline characters
                    lines = text.split('\n')

                    # Wrap each line individually
                    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

                    # Join the wrapped lines back together using newline characters
                    wrapped_text = '\n'.join(wrapped_lines)
                    return wrapped_text
        def process_llm_response(llm_originalresponse2):
                    #result_text = wrap_text_preserve_newlines(llm_originalresponse2["result"])
                    typewriter(llm_originalresponse2["result"], speed=40)

        def extract_video_id(youtube_url):
            try:
                parsed_url = urlparse(youtube_url)
                query_params = parse_qs(parsed_url.query)
                video_id = query_params.get('v', [None])[0]

                return video_id
            except Exception as e:
                print(f"Error extracting video ID: {e}")
                return None



        def launchwebsitecomponent():
                HUGGINGFACE_API_TOKEN = "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"
                model_name = "BAAI/bge-base-en"
                encode_kwargs = {'normalize_embeddings': True}

                st.title('TextGenius: Your Chat with Websites Assistant 🌐🤖')

                url = st.text_input("Insert the Website URL", placeholder="Format should be like: https://platform.openai.com/account/api-keys.")
                query = st.text_input("Ask any question about the Website",help="Suggested queries: Summarize the key points of this webpage - What is this website about - Ask about a specific thing in the webite ")


                if st.button('Generate Response 🪄✨'):
                  with st.spinner('Processing the Website Data...'):

                      loader = WebBaseLoader(url)
                      documents = loader.load()

                      text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                      documents = text_splitter.split_documents(documents)
                      if os.path.exists('./data'):
                          shutil.rmtree('./data')
                      vectordb = Chroma.from_documents(
                      documents,
                      #embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                                                            # model_kwargs={"device": "cuda"})
                      embedding= HuggingFaceBgeEmbeddings( model_name=model_name, model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}, encode_kwargs=encode_kwargs)
                  )

                      repo_id = "tiiuae/falcon-7b-instruct"
                      qa_chain = RetrievalQA.from_chain_type(

                      llm=HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
                                      repo_id=repo_id,
                                      model_kwargs={"temperature":0.7, "max_new_tokens":700}),
                          retriever=vectordb.as_retriever(),
                          return_source_documents=False,
                          verbose=False
                      )
                  with st.spinner('Generating Answer...'):
                        llm_response = qa_chain(query)
                        #llm_originalresponse2=llm_response['result']
                        process_llm_response(llm_response)
                st.warning("⚠️ Please Keep in mind that the accuracy of the response relies on the :red[Website Layout] and the :red[prompt's Quality]. Occasionally, the response may not be entirely accurate. Consider using the response as a reference rather than a definitive answer.")
        launchwebsitecomponent()
def chat_with_youtube():
          import torch
          import os
          import argparse
          import shutil
          from langchain.document_loaders import YoutubeLoader
          from langchain.text_splitter import RecursiveCharacterTextSplitter
          from langchain.vectorstores import Chroma
          from langchain.embeddings import OpenAIEmbeddings
          from langchain.chains import RetrievalQA
          from langchain.llms import OpenAI
          import streamlit as st
          from langchain.chat_models import ChatOpenAI
          from langchain import HuggingFaceHub
          from langchain.embeddings import HuggingFaceInstructEmbeddings
          from urllib.parse import urlparse, parse_qs
          from langchain.embeddings import HuggingFaceBgeEmbeddings
          from transformers import pipeline
          import textwrap
          import time
          from deep_translator import GoogleTranslator
          from langdetect import detect


          def typewriter(text: str, speed: float):
                      container = st.empty()
                      displayed_text = ""

                      for char in text:
                          displayed_text += char
                          container.markdown(displayed_text)
                          time.sleep(1/speed)
          def wrap_text_preserve_newlines(text, width=110):
                      # Split the input text into lines based on newline characters
                      lines = text.split('\n')

                      # Wrap each line individually
                      wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

                      # Join the wrapped lines back together using newline characters
                      wrapped_text = '\n'.join(wrapped_lines)
                      return wrapped_text
          def process_llm_response(llm_originalresponse2):
                      #result_text = wrap_text_preserve_newlines(llm_originalresponse2["result"])
                      typewriter(llm_originalresponse2["result"], speed=40)

          def extract_video_id(youtube_url):
              try:
                  parsed_url = urlparse(youtube_url)
                  query_params = parse_qs(parsed_url.query)
                  video_id = query_params.get('v', [None])[0]

                  return video_id
              except Exception as e:
                  print(f"Error extracting video ID: {e}")
                  return None



          def launchyoutubecomponent():
                  HUGGINGFACE_API_TOKEN = "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"
                  model_name = "BAAI/bge-base-en"
                  encode_kwargs = {'normalize_embeddings': True}

                  st.title('TextGenius: Your Chat with Youtube Assistant')

                  videourl = st.text_input("Insert The video URL",  placeholder="Format should be like: https://www.youtube.com/watch?v=pSLeYvld8Mk")
                  query = st.text_input("Ask any question about the video",help="Suggested queries: Summarize the key points of this video - What is this video about - Ask about a specific thing in the video ")


                  if st.button('Generate Response 🪄✨'):
                    with st.spinner('Processing the Video...'):
                        video_id = extract_video_id(videourl)
                        loader = YoutubeLoader(video_id)
                        documents = loader.load()

                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                        documents = text_splitter.split_documents(documents)
                        persist_directory='db1'
                        if os.path.exists(persist_directory):
                            shutil.rmtree(persist_directory)
                        vectordb = Chroma.from_documents(
                        documents,
                        #embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                                                              # model_kwargs={"device": "cuda"})
                        embedding= HuggingFaceBgeEmbeddings( model_name=model_name, model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}, encode_kwargs=encode_kwargs),persist_directory=persist_directory
                    )

                        repo_id = "tiiuae/falcon-7b-instruct"
                        qa_chain = RetrievalQA.from_chain_type(

                        llm=HuggingFaceHub(huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
                                        repo_id=repo_id,
                                        model_kwargs={"temperature":0.7, "max_new_tokens":700}),
                            retriever=vectordb.as_retriever(),
                            return_source_documents=False,
                            verbose=False
                        )
                    with st.spinner('Generating Answer...'):
                          llm_response = qa_chain(query)
                          #llm_originalresponse2=llm_response['result']
                          process_llm_response(llm_response)
                  st.warning("⚠️ Please Keep in mind that the accuracy of the response relies on the :red[Video's quality] and the :red[prompt's Quality]. Occasionally, the response may not be entirely accurate. Consider using the response as a reference rather than a definitive answer.")
          launchyoutubecomponent()
def intro():
            st.markdown("""
            # Welcome to TextGenius

            TextGenius is an innovative web application designed to harness the power of advanced Large Language Models (LLMs) and provide users with an intuitive platform for interacting with text data. By leveraging cutting-edge AI technology, TextGenius simplifies the extraction of insights from documents, YouTube videos, and website content. Whether you are an academic researcher, industry professional, or student, TextGenius offers a tailored experience that elevates your data interaction to new heights.

            ## Base Models

            Q&A-Assistant is built on Falcon 7B instruct Model to enhance your research experience. Whether you're a student, researcher, or professional, we're here to simplify your interactions with your documents. 💡📚

            ## Standout Features

            - AI-Powered Q&A: Upload your PDF , Input a Youtube video or website Link,  Get precise answers like a personal Q&A expert! 💭🤖

            ## How to Get Started

            1. Upload your Document, or Input a Youtube video or website Link
            3. Ask questions using everyday language using your favorite language
            4. Get detailed, AI-generated answers.
            5. Enjoy a smarter way to interact with text data!


            ## It is Time to Dive in! Welcome aboard the journey to a smarter way of handling text data.


            """)
page_names_to_funcs = {
    "Main Page": intro,
    "Chat with Documents": chat_with_documents,
    "Chat with Youtube Videos": chat_with_youtube,
    "Chat with a Website": chat_with_website

}







demo_name = st.sidebar.selectbox("Please choose your tool 😊 ", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
st.sidebar.markdown('<a href="https://www.linkedin.com/in/mohammed-khalil-ghali-11305119b/"> Connect on LinkedIn <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" alt="LinkedIn" width="30" height="30"></a>', unsafe_allow_html=True)
st.sidebar.markdown('<a href="https://github.com/khalil-ghali"> Check out my GitHub <img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" alt="GitHub" width="30" height="30"></a>', unsafe_allow_html=True)