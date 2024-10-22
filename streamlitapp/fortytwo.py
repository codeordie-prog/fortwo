
import streamlit as st
import io
import tempfile
import os
import requests
import sys,base64
from lxml import html
from PIL import Image
from io import BytesIO
from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict
from git import Repo
from langchain_core.callbacks import BaseCallbackHandler
from langchain_text_splitters import Language
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores.docarray import DocArrayInMemorySearch
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate,MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.memory.buffer import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.chat_message_histories.streamlit import StreamlitChatMessageHistory
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.language import LanguageParser
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import vision,audio,openai_audio
from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import systemprompt
import numpy as np
import pdfgenerator

# You might also need to install some additional dependencies used in the code such as:
# pip install streamlit langchain streamlit-chat gitpython requests lxml pillow pydantic

st.set_page_config(
    page_title="Chatwith42",
    page_icon="👽",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/codeordie-prog/fortwo/blob/master/streamlitapp/fortytwo.py",
        "Report a bug": "https://github.com/codeordie-prog/fortwo/blob/master/streamlitapp/fortytwo.py",
        "About": """
            ## Chatwith42
            
            **GitHub**: https://github.com/codeordie-prog
            
            The AI Assistant named, 42, utilizes RAG to answer queries about your documents in `.pdf`,`.txt`, or `.csv` format,
            participate in general chat sessions.
        """
    }
)


#----------------------------------------------------- Load the image function-----------------------------------------------------#

try:
    def load_image(image_path):
        try:
            with open(image_path, "rb") as image_file:
                return image_file.read()
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return None

    imag_path = "streamlitapp/logo/stimage.jfif"
    image_bytes = load_image(imag_path)

    # Create two columns
    col1, col2,col3,= st.columns([1,1,1])

    #Display the image in the center column
    with col1:
        if image_bytes:
         st.image(io.BytesIO(image_bytes), width=50)
        else:
         st.error("Failed to load image.")

    #------------------------------------------------------------tabs----------------------------------------------------------------------------

    tab1,tab2,tab3,tab4 = st.tabs(["chat","document query","github","scraping"])


    #-----------------------------------------------------------sidebar about section-------------------------------------------------------------#
    st.sidebar.image("streamlitapp/logo/stimage.jfif",width=250)
    #st.sidebar.title("chatwith42")
    with st.sidebar.expander(label="About",expanded=False):
     st.markdown("""
        Hi carbon entity! am 42, a powerful knowledge discovery engine named after the ultimate answer in the Hitchhikers Guide to the Galaxy.
        My brain is powered by GPT models from OpenAI and opensource models from Meta and NVIDIA.
        My capabilities include:
        - chat
        - image generation
        - image description
        - retrieval augmented generation
        - github repositories querying
        - web scrapping
        """)



    #--------------------------------------------------sidebar instructions section-------------------------------------------------------------#

    with st.sidebar.expander(label="Instructions for an API key"):
        st.markdown(""" 
    OpenAI:
    1. Go to [OpenAI API Keys](https://platform.openai.com/account/api-keys).
    2. Click on the `+ Create new secret key` button.
    3. Next, enter an identifier name (optional) and click on the `Create secret key` button.
    
    NVIDIA:
    1. Go to [Nvidia platform](https://www.nvidia.com/en-us/ai/)
    2. Click on the Try now button and register for an account
    3. Generate an API key
    4. Copy it inside the NVIDIA API key section""")

    #-------------------------------------------------------API PROVIDERS--------------------------------------------------------------------------#

    with st.sidebar.expander(label="Choose API provider"):
        api_provider = st.selectbox(
            label="choose API provider",
            options=["openai", "nvidia nim"]

    )
    # Input for OpenAI API key in the sidebar
    with st.sidebar.expander(label = "Add API keys", expanded=False):

        openai_api_key = st.text_input("OpenAI API Key", type="password")
        nvidia_api_key = st.text_input("Nvidia API key", type="password")

    include_audio = st.sidebar.toggle(label="turn on audio responses")

     #---------------------------------------------------def download pdf---------------------------------------------------------#

    def download_pdf(content:str, filename:str):

        file_name = f"{filename}.pdf"
        st.download_button(
            label="download pdf",
            data=content,
            file_name=file_name,
            mime="application/pdf"
        )

    #_____________________________________________set models_______________________________________________________________________________

    with tab1:

        col1, col2 = st.columns([2, 1])  # Adjust ratios to control width

        with col1:  # First column for the model selection
            if api_provider == "openai":
                with st.expander(label="Choose GPT Model", expanded=False):
                    llm_model_chat = st.selectbox(label="Choose chat model",
                                                options=["gpt-4o-mini", "gpt-4o-2024-08-06", "gpt-4o", "gpt-3.5-turbo"],
                                                key="chat_key")
            else:
                with st.expander(label="Choose Model", expanded=False):
                    llm_model_chat = st.selectbox(label="Choose model",
                                                options=["nvidia/llama-3.1-nemotron-70b-instruct",
                                                        "meta/llama-3.1-8b-instruct",
                                                        "meta/llama-3.1-405b-instruct"])

        with col2:  # Second column for the PDF generation section
            with st.expander("Prepare PDF file for download", expanded=False):
                file_name = st.text_input("Enter file name")

                if file_name:
                    # Download PDF
                    text = ""

                    for messages in st.session_state["messages"]:
                        text += messages["content"] + "\n"
                    
                    cleaned_response = pdfgenerator.clean_text(text=text)
                    pdf_file = pdfgenerator.generate_pdf(content=cleaned_response)
                    download_pdf(content=pdf_file, filename=file_name)
                else:
                    st.info("Please provide a file name.")

    with tab2:

        if api_provider == "openai":
             
             with st.expander(label="choose GPT model",expanded=False):
             
                llm_model_docs = st.selectbox(label="choose document query model",
                                      options=["gpt-4o-mini","gpt-4o","gpt-4o-2024-08-06"],key="document_query_key")

        elif api_provider == "nvidia nim":
             
             with st.expander(label="choose model",expanded=False):

                llm_model_docs = st.selectbox(label="choose document query model",
                                      options=["nvidia/llama-3.1-nemotron-70b-instruct","meta/llama-3.1-8b-instruct","meta/llama-3.1-405b-instruct"],key="document_query_key")

         # File uploader in the sidebar
        uploaded_files = st.file_uploader(
            label="Upload files", type=["pdf", "txt", "csv","jpg","png","jpeg"], accept_multiple_files=True
        )
    

        

    with tab3:

        repo_url = st.text_input("Enter repository url: ")

    with tab4:

        llm_model_web = st.selectbox(label="choose scraping model",
                                     options=["gpt-4o","gpt-4o-mini"],key="scraping_key")
        
        url = st.text_input("enter url")
        web_document_name = st.text_input("Enter name for the web document")



    # Inject custom CSS for glowing border effect
    st.markdown(
        """
        <style>
        .cover-glow {
            width: 100%;
            height: auto;
            padding: 3px;
            box-shadow: 
                0 0 5px #330000,
                0 0 10px #660000,
                0 0 15px #990000,
                0 0 20px #CC0000,
                0 0 25px #FF0000,
                0 0 30px #FF3333,
                0 0 35px #FF6666;
            position: relative;
            z-index: -1;
            border-radius: 30px;  /* Rounded corners */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    

    #----------------------------------------------streamhandler and retriever class section------------------------------------------------------#

    
        
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
            self.container = container
            self.text = initial_text
            self.run_id_ignore_token = None
            self.latex_mode = False  # Track if we are in LaTeX mode

        def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
            # Workaround to prevent showing the rephrased question as output
            if prompts[0].startswith("Human"):
                self.run_id_ignore_token = kwargs.get("run_id")

        def on_llm_new_token(self, token: str, **kwargs) -> None:
            if self.run_id_ignore_token == kwargs.get("run_id", False):
                return
            
            # Check if the token indicates the start or end of LaTeX
            if token == '$':
                self.latex_mode = not self.latex_mode  # Toggle LaTeX mode

            # Append the token to the text
            self.text += token
            
            # If in LaTeX mode, format accordingly
            if self.latex_mode:
                formatted_text = f"${self.text}$"  # Wrap in LaTeX
            else:
                formatted_text = self.text
            
            self.container.markdown(formatted_text)


    class PrintRetrievalHandler(BaseCallbackHandler):
            def __init__(self, container):
                self.status = container.status("**Context Retrieval**")

            def on_retriever_start(self, serialized: dict, query: str, **kwargs):
                self.status.write(f"**Question:** {query}")
                self.status.update(label=f"**Context Retrieval:** {query}")
            def on_retriever_end(self, documents, **kwargs):
                for idx, doc in enumerate(documents):
                    source = os.path.basename(doc.metadata["source"])
                    self.status.write(f"**Document {idx} from {source}**")
                    self.status.markdown(doc.page_content)
                self.status.update(state="complete")


        #chat setup
        # Setup LLM and QA chain - msg variable for chat history from streamlitchatmessagehistory
        #set up the memory with chat_memory as the msg variable -use conversational buffer memory
        #set up the prompt
        #initialize the llm with streaming true
        #initialize the chain with all the set up fields i.e promp,memory,verbose false and llm
        #use the chain to invoke chat query



    #----------------------------------------------configuring retriever section----------------------------------------------------------#

    
    @st.cache_resource(ttl="2h")
    def configure_retriever(uploaded_files):
            # Read documents
            docs = []
            image_description_prompt = vision.image_description_prompt
            ext=[".jpeg",".jpg",".png"]
            with tempfile.TemporaryDirectory() as temp_dir:
                 temp_dir_path = temp_dir
                 temp_dir = tempfile.TemporaryDirectory()
                 for file in uploaded_files:
                    temp_filepath = os.path.join(temp_dir.name, file.name)
                    st.write(temp_filepath)
                    with open(temp_filepath, "wb") as f:
                        f.write(file.getvalue())

                    #load pdf,txt and csv
                    if temp_filepath.endswith(".pdf"):
                        loader = PyPDFLoader(temp_filepath)
                        docs.extend(loader.load())
                        
                    elif temp_filepath.endswith(".txt"):
                        loader = TextLoader(temp_filepath)
                        docs.extend(loader.load())
                    
                    elif temp_filepath.endswith(".csv"):
                        loader = CSVLoader(temp_filepath)
                        docs.extend(loader.load())  

                    elif any(temp_filepath.endswith(e) for e in ext):

                        if api_provider == "nvidia nim" and not openai_api_key:
                            st.info("please add api key")
                            st.stop()
                        else:
                            st.image(file,width=380)
                            base64image = vision.encode_image(temp_filepath)
                            description = vision.describe_image(base64image,openai_api_key=openai_api_key,prompt=image_description_prompt)
                            description_file_path = os.path.join(temp_dir_path, file.name + ".txt")
                            with open(description_file_path, "w") as description_file:
                                description_file.write(description)
                            loader = TextLoader(description_file_path)
                        
                            docs.extend(loader.load())
                    

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            # Create embeddings and store in vectordb
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

            # Define retriever
            retriever = vectordb.as_retriever(
                 search_type ="mmr",
                 search_kwargs = {"k" : 8}) #retrieve default

            return retriever

   

    #---------------------------------------------define download txt function-------------------------------------------------------------------#

    #function-3
    #define download txt
    def create_and_download(text_content):
            """Generates a text file in memory and offers a download button."""
            # Create a BytesIO object to store the content
            buffer = BytesIO()

            # Write the content to the buffer
            if isinstance(text_content, bytes):
                buffer.write(text_content)
            else:
                buffer.write(text_content.encode('utf-8'))

            buffer.seek(0)

            # Provide download button
            st.sidebar.download_button(
                label="Download Text File",
                data=buffer,
                file_name="my_text_file.txt",
                mime="text/plain"
            )
        
    #-------------------------------------------------------------chat setup section---------------------------------------------------------#

    #function-4 chat session

    def chat_with_42():
            
            system_prompt = systemprompt.system_prompt
            
            with tab1:
                # Define the system prompt template
                

                response_placeholder = st.empty()
                input_placeholder = st.empty()


                with response_placeholder.container():
                    # Initialize chat history if not already in session state
                    if "messages" not in st.session_state:
                        st.session_state["messages"] = [{"role": "assistant", "content": f'42 : "How can I help you?"'}]

                    #if "scratchpad" not in st.session_state:
                        #st.session_state["scratchpad"] = ""

                    # Display chat history messages
                    for msg in st.session_state["messages"]:
                        st.chat_message(msg["role"]).write(msg["content"])

                    #st.sidebar.subheader("LLM thought process")
                    #st.sidebar.text_area("Scratchpad",st.session_state["scratchpad"],height=300)

                # "Clear Chat History" button
                if st.button("Clear Chat History",key="chat_clear"):
                    st.session_state["messages"] = [{"role": "assistant", "content": f"42 : {'Chat history cleared. How can I help you?'}"}]
                    st.rerun()  # Rerun the app to clear the chat history

                
                user_input =  st.chat_input(key="chat input") 

                with input_placeholder.container():
                
                    try:
                        
                        # Handle user input
                        if user_input != None:

                            if api_provider == "openai":

                                if not openai_api_key:
                                    st.info("Please add your OpenAI API key to continue.")
                                    st.stop()

                                # Initialize OpenAI LLM
                                llm2 = ChatOpenAI(openai_api_key=openai_api_key, model = llm_model_chat, streaming = True)

                            elif api_provider == "nvidia nim":
                                if not nvidia_api_key:
                                        st.info("add NVIDIA API")
                                        st.stop()

                                llm2 = ChatNVIDIA(model=llm_model_chat,api_key = nvidia_api_key, streaming=True)

                            # Initialize Streamlit chat history
                            chat_history = StreamlitChatMessageHistory(key="chat_history")

                            # Set up memory for conversation
                            memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=chat_history, return_messages=True)

                            # Create the LLM chain
                            llm_chain = LLMChain(
                                llm=llm2,
                                verbose=False,
                                memory=memory,
                                prompt=system_prompt
                            )

                            
                            # Append user message to session state
                            st.session_state["messages"].append({"role": "user", "content": f"You : {user_input}"})
                            st.chat_message("user").write(user_input)

                            #introduce streaming in chat session
                            stream_handler = StreamHandler(st.empty())
                    
                            # Get response from LLM chain

                            if api_provider == "openai":
                                with st.spinner("`Thinking..`"):
                                    response = llm_chain.run({"question": user_input}, callbacks = [stream_handler])

                            elif api_provider == "nvidia nim":
                                    nvidia_chain = system_prompt | llm2 | StrOutputParser()
                                    nim_resp = ""
                                    response_display = st.empty()
                                    with st.spinner("`Thinking..`"):
                                        response = nvidia_chain.invoke({"question": user_input,"chat_history":st.session_state["messages"]})
                                    for chunk in response:
                                        nim_resp+=chunk
                                        response_display.write(nim_resp)
                            

                           

                            #image generation function calling
                            if response.startswith("Abracadabra baby.") and openai_api_key:
                                with st.spinner(text="Generating image in progress..."):
                                    image_url= vision.generate_image(description=user_input,openai_api_key=openai_api_key)
                                    
                                    with tempfile.TemporaryDirectory() as temporary_directory:
                                        image_path = vision.download_generated_image(image_url=image_url,image_storage_path=temporary_directory)
                                        st.image(image=image_path,use_column_width=True)

                                        if image_path:
                                            with open(image_path,"rb") as file:
                                                image_bytes = file.read()

                                            st.download_button(
                                                label="download_image",
                                                data=image_bytes,
                                                file_name="image.png",
                                                mime="image/png"
                            
                                            )

                            elif response.startswith("Abracadabra baby.") and not openai_api_key:
                                 st.info("Image generation requires a valid openai api key, please provide one. ")

                            assistant_msg = response  # Adjusted to fetch text from the response

                            if assistant_msg == "Generated image.":
                                st.session_state["messages"].append({"role":"assistant","content":f"Here is your generated image:{image_url}, for the description : {user_input}"})
                                

                            # Append assistant message to session state and display it
                            st.session_state["messages"].append({"role": "assistant", "content": f'42 : {assistant_msg}'})

                             #download pdf
                            text = ""

                            for messages in st.session_state["messages"]:
                                text+=messages["content"] + "\n"
                            cleaned_response = pdfgenerator.clean_text(text=text)
                            pdf_file = pdfgenerator.generate_pdf(content=cleaned_response)
                            download_pdf(content=pdf_file)


                            if include_audio and openai_api_key:

                                responses_path=openai_audio.text_to_speech(response,openai_api_key)

                                if responses_path != None:
                                    st.audio(responses_path,format="audio")

                                else:
                                    st.write(f"Length {len(response)} of the response too long to process the audio.")
                           

                                #download the audio
                                    
                                with open(responses_path, "rb") as audio_file:
                                     data = audio_file.read()
                                     st.download_button(label="download",data=data,file_name="audio.mp3",mime="audio/mp3")
                                        
                            elif include_audio and not openai_api_key:

                                 st.info("add an openai api key to include audio response")
                                 st.stop()

                            
                            
                                

                            

                        

                    except Exception as e:
                        st.write("an Error occured please enter a valid API key",e)

    #---------------------------------------------------------RAG setup section------------------------------------------------------------------#
    #query website function
    @st.cache_resource(ttl="2h")
    def web_page_saver_to_txt(url):

        try:

            if url is not None:
                results = requests.get(url)
                web_content = results.content
    
                # Step 2: Parse the webpage content using lxml
                tree = html.fromstring(web_content)
    
                # Step 3: Extract the desired data (text from <p> tags in this example)
                paragraphs = tree.xpath('//p')
                text_content = '\n'.join([para.text_content() for para in paragraphs])
    
                # Step 4: Save the data to a temporary file with a specified name
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_file_path = os.path.join(temp_dir, web_document_name)
                    with open(temp_file_path, 'w', encoding='utf-8') as temp_file:
                        temp_file.write(text_content)
    
                    # Load the text file using TextLoader
                    loader = TextLoader(temp_file_path)
                    docs = loader.load()
    
                    # Split the documents
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
                    splits = text_splitter.split_documents(docs)
    
                    # Create embeddings and store in vectordb
                    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    
                    # Define retriever
                    retriever = vectordb.as_retriever()
    
                    return retriever
            else:
                st.write("url object returned is Null")
        except Exception:
             st.write("enter valid URL")

            

    #function-4 query documents           
    def query_documents():
            
            with tab2:
            
                if not uploaded_files:
                    st.info("Please upload documents or add url to continue.")
                    st.stop()
                    
                retriever = configure_retriever(uploaded_files)

                llm = None

                response_placeholder = st.empty()
                input_placeholder = st.empty()   
                
                # Setup memory for contextual conversation for the documents part
                msgs = StreamlitChatMessageHistory(key="docs tab")
                memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

                    
                    # Setup LLM and QA chain for the documents part

                if api_provider == "openai" and openai_api_key:
                    llm = ChatOpenAI(
                        model_name=llm_model_docs, openai_api_key=openai_api_key, temperature=0, streaming=True
                        )

                elif api_provider == "nvidia nim" and nvidia_api_key:

                    ext = ["png","jpeg","jpg"]
                    
                    #check whether the uploaded doc is a picture and ensure openai api is provided 
                    if any(doc.name.endswith(tuple(ext)) for doc in uploaded_files):

                        if not openai_api_key:

                            st.info("image analysis requires openai api key please add one")
                            st.stop()

                    llm = ChatNVIDIA(model=llm_model_docs,api_key = nvidia_api_key, streaming=False)

                else:

                    st.info("make sure you have added the API keys")
                    st.stop()

                qa_chain = ConversationalRetrievalChain.from_llm(
                        llm, 
                        retriever=retriever, 
                        memory=memory, 
                        verbose=True
                    )
             

                

                if len(msgs.messages) == 0 or st.button("Clear message history",key="docs_clear"):
                        msgs.clear()
                        msgs.add_ai_message("Hey carbon entity, Want to query your documents? ask me!")

                with response_placeholder.container():
                    avatars = {"human": "user", "ai": "assistant"}
                    for msg in msgs.messages:
                        st.chat_message(avatars[msg.type]).write(msg.content)
                    
                user_query = st.chat_input(placeholder="Ask me about  your documents!",key="document")

                with input_placeholder.container():
                    if user_query !=None:
                        st.chat_message("user").write(f"You : {user_query}")

                        with st.chat_message("ai"):
                                retrieval_handler = PrintRetrievalHandler(st.container())
                                stream_handler = StreamHandler(st.empty())

        
                                response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])

                                if api_provider == "nvidia nim":

                                    string_resp = ""
                                    string_resp_place = st.empty()

                                    for chunk in response:

                                        string_resp+=chunk
                                        string_resp_place.write(string_resp)


                                text = ""

                                for message in msgs.messages:
                                    text+=f"{message.content}" + "\n"

                                clean_text = pdfgenerator.clean_text(text)
                                pdf_file = pdfgenerator.generate_pdf(content=clean_text)

                                download_pdf(content=pdf_file)

                                if include_audio and openai_api_key:

                                    responses_path=openai_audio.text_to_speech(response,openai_api_key)
                                    st.audio(responses_path,format="audio")

                                            #download the audio
                                                
                                    with open(responses_path, "rb") as audio_file:
                                            data = audio_file.read()
                                
                                        
                                            st.download_button(label="download",data=data,file_name="audio.mp3",mime="audio/mp3")


                                elif include_audio and not openai_api_key:
                                    st.info("add openai api key to include audio response")
                                    st.stop()

    def query_web():
            
            with tab4:

                if not url or not web_document_name:
                    st.info("Please add url to continue.")
                    st.stop()
                    
                retriever = web_page_saver_to_txt(url)
                response_placeholder = st.empty()
                input_placeholder = st.empty()

                # Setup memory for contextual conversation for the documents part
                msgs = StreamlitChatMessageHistory(key="web_messages")
                memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

            
                    
                    # Setup LLM and QA chain for the documents part
                llm = ChatOpenAI(
                        model_name=llm_model_web, openai_api_key=openai_api_key, temperature=0, streaming=True
                    )


                qa_chain = ConversationalRetrievalChain.from_llm(
                        llm, 
                        retriever=retriever, 
                        memory=memory, 
                        verbose=True
                    )

                    

                if len(msgs.messages) == 0 or st.button("Clear web query history",key="web clear"):
                        msgs.clear()
                        msgs.add_ai_message("Hey carbon entity, Want to query your documents? ask me!")

                with response_placeholder.container():
                    avatars = {"human": "user", "ai": "assistant"}
                    for msg in msgs.messages:
                        st.chat_message(avatars[msg.type]).write(msg.content)
                    
                user_query = st.chat_input(placeholder="Ask me about  your documents!",key="web query")

                with input_placeholder.container():
                    if user_query != None:
                        st.chat_message("user").write(user_query)

                        with st.chat_message("ai"):
                                retrieval_handler = PrintRetrievalHandler(st.container())
                                stream_handler = StreamHandler(st.empty())

                                qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])

    #define repo query
   
    def github_repo_query(github_repo_url: str, open_ai_key: str):
        try:

            with tab3:
                if repo_url == None:
                    st.info("please add the repository url to proceed.")
                    st.stop()

                else:
                
                    with tempfile.TemporaryDirectory() as temp_dir:
                        # Clone the repo
                        repo_path = os.path.join(temp_dir, "repo")
                        repo = Repo.clone_from(github_repo_url, to_path=repo_path)
                        documents = []

                        #add suffixes
                        language_suffixes = {
                            
                            Language.PYTHON : [".py"],
                            Language.JAVA : [".java"],
                            Language.GO : [".go"],
                            Language.CPP : [".cpp",".hpp", ".cc", ".hh", ".cxx", ".hxx", ".h"],
                            Language.KOTLIN:[".kt",".kts"],
                            Language.TS : [".ts"],
                            Language.CSHARP : [".cs"]
                            
                        }

                        #document loader
                        for language, suffix in language_suffixes.items():
                            loader = GenericLoader.from_filesystem(
                                
                                repo_path,
                                glob="**/*",
                                suffixes=suffix,
                                exclude=["**/non-utf8-encoding.*"],
                                parser=LanguageParser(
                                    language=language, parser_threshold=500
                                )
                            )

                            documents.extend(loader.load())

                            

                        #split

                        split_texts = []

                        for language in language_suffixes.keys():
                            splitter = RecursiveCharacterTextSplitter.from_language(
                                language=language, chunk_size = 1500, chunk_overlap = 200
                            )

                            split_texts.extend(splitter.split_documents(documents))

                       

                        
                        #use docarraysearch


                        # Retriever
                        db = DocArrayInMemorySearch.from_documents(split_texts, embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
                        retriever = db.as_retriever(
                            search_type = "mmr",                        # Also test "similarity"
                            search_kwargs={"k": 8},
                        )

                        
                        llm = ChatOpenAI(model_name="gpt-4o",api_key=open_ai_key)

                        # Prompt
                        prompt_retriever = ChatPromptTemplate.from_messages(
                            [
                                ("placeholder", "{chat_history}"),
                                ("user", "{input}"),
                                (
                                    "user",
                                    "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
                                ),
                            ]
                        )

                        retriever_chain = create_history_aware_retriever(llm, retriever, prompt_retriever)

                        prompt_document = ChatPromptTemplate.from_messages(
                            [
                                (
                                    "system",
                                    "Answer the user's questions based on the below context:\n\n{context}",
                                ),
                                ("placeholder", "{chat_history}"),
                                ("user", "{input}"),
                            ]
                        )
                        document_chain = create_stuff_documents_chain(llm, prompt_document)

                        qa = create_retrieval_chain(retriever_chain, document_chain)

                        
                        return qa

                        
        except Exception as e:
             st.write("an error occured inside the github repo function, check the URL.",e)


    #-----------------------------------------------------------audio---------------------------------------------------------------------------
    
       
    #--------------------------------------------------------------main function------------------------------------------------------------------#
    st.cache_resource(ttl="2h")
    def main():

        try:

            with tab1:
                chat_with_42()
                  
            with tab2:
                  query_documents()

            # Content for "Github" tab
            with tab3:
                with st.container():
                    if repo_url:
                        # Initialize session state for messages if not already set
                        if "messages_github" not in st.session_state:
                            st.session_state["messages_github"] = [{"role": "assistant", "content": "How can I help with the code base?"}]
                        
                        # Create containers for chat messages and user input
                        chat_placeholder = st.empty()  # Placeholder for chat messages
                        input_placeholder = st.empty()  # Placeholder for user input

                        # Display existing chat history in chat_placeholder
                        with chat_placeholder.container():
                            for msg in st.session_state["messages_github"]:
                                st.chat_message(msg["role"]).write(msg["content"])

                        # Handle user input
                        user_input = input_placeholder.chat_input(placeholder="Type your question here...")

                        # Process the user input if provided
                        if user_input:
                            # Add the user's message to the session state
                            st.session_state["messages_github"].append({"role": "user", "content": user_input})
                            
                            # Display updated chat messages with user message
                            with chat_placeholder.container():
                                for msg in st.session_state["messages_github"]:
                                    st.chat_message(msg["role"]).write(msg["content"])

                            # "Clear Chat History" button
                            if st.button("Clear github History",key="github_clear"):
                                st.session_state["messages"] = [{"role": "assistant", "content": "github history cleared. How can I help you?"}]
                                st.rerun()  # Rerun the app to clear the chat history

                            # Query the GitHub repository
                            chain = github_repo_query(repo_url, open_ai_key=openai_api_key)

                            # Use pick to select the desired key
                            stream_chain = chain.pick("answer")
                            
                            # Create a response placeholder and set it to empty; it will be updated with each chunk
                            response = ""
                            for chunk in stream_chain.stream({"input": user_input}):
                                with st.spinner("In progress.."):
                                    response += f"{chunk}"
                                    chat_placeholder.chat_message("assistant").write(response)  # Update the placeholder with each chunk
                            
                            # Update session state with the assistant's message
                            st.session_state["messages_github"].append({"role": "assistant", "content": response})
                            
                            # Display updated chat messages with assistant's response
                            with chat_placeholder.container():
                                for msg in st.session_state["messages_github"]:
                                    st.chat_message(msg["role"]).write(msg["content"])


                 

            with tab4:
                query_web()
               
             



         
        except TypeError :
            st.write("encountered a None type inside main call, check url submitted it might be returning a none type object")
        except Exception :
             st.write("An error was encountered at main call")
    

        
    #call main
    if __name__ == "__main__":
        main()


except Exception:
    st.write("an error occured check the key")

 

