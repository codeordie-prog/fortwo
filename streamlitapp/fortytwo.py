import streamlit as st
import io
import tempfile
import os
import requests
import sys
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

    imag_path = "streamlitapp/logo/lgo.jfif"
    image_bytes = load_image(imag_path)

    # Create two columns
    col1, col2,col3,= st.columns([1,2,1])

    #Display the image in the center column
    with col2:
        if image_bytes:
         st.image(io.BytesIO(image_bytes), width=400)
        else:
         st.error("Failed to load image.")

   



    #-----------------------------------------------------------sidebar about section-------------------------------------------------------------#
    st.sidebar.image("streamlitapp/logo/stimage.jfif",width=300)
    st.sidebar.title("Chatwith42")
    st.sidebar.subheader("About")
    st.sidebar.info("""
        42 named after the Hitch Hiker's Guide to the Galaxy answer to the ultimate question of life,
        is a knowledge discovery engine that utilizes Retrieval Augmented Generation to query uploaded documents in the format `.pdf`,`.csv` and `.txt`,
        42 has a chat section where users can chat with the bot without uploading the documents, and a code understanding section where
        you can query a repository on GitHub. For code understanding select the GitHub toggle button and key in the repository url.
        Use gpt-4o and gpt-4 for a better response. 
    """)



    #--------------------------------------------------sidebar instructions section-------------------------------------------------------------#

    st.sidebar.subheader("Get an openAI API key")
    st.sidebar.info("""
    1. Go to [OpenAI API Keys](https://platform.openai.com/account/api-keys).
    2. Click on the `+ Create new secret key` button.
    3. Next, enter an identifier name (optional) and click on the `Create secret key` button.""")

    
    # Input for OpenAI API key in the sidebar
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    #____________________________________________________________radios_______________________________________________________________________

    sidebar_option = st.sidebar.radio(
         "Select an option",
         ("Chat and query","Github")
    )

    if sidebar_option == "Github":
         repo_url = st.sidebar.text_input("Enter repository url: ")

    #-----------------------------------------------------upload documents sidebar--------------------------------------------------------------
    # File uploader in the sidebar
    uploaded_files = st.sidebar.file_uploader(
        label="Upload files", type=["pdf", "txt", "csv"], accept_multiple_files=True
    )

    #---------------------------------------------------sidebar for query web-------------------------------------------------------------------#

    st.sidebar.subheader("Query web section")
    st.sidebar.info("""
                    use this section to interact with web information. 
                    1. Copy the URL of the webpage and paste it on the URL input section.
                    2. Enter the document saving name. as you desire.
                    3. Query the website for information by asking 42 about it.""")
    
    url = st.sidebar.text_input("enter url")
    web_document_name = st.sidebar.text_input("Enter name for the web document")

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

            def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
                # Workaround to prevent showing the rephrased question as output
                if prompts[0].startswith("Human"):
                    self.run_id_ignore_token = kwargs.get("run_id")

            def on_llm_new_token(self, token: str, **kwargs) -> None:
                if self.run_id_ignore_token == kwargs.get("run_id", False):
                    return
                self.text += token
                self.container.markdown(self.text)


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

        #function-2   
    @st.cache_resource(ttl="2h")
    def configure_retriever(uploaded_files):
            # Read documents
            docs = []
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
            # Define the system prompt template
            system_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content="""You are a very intelligent digital AI system that understands humans. Your name is 42,
                        you were named after the answer to the ultimate question in the hitch hikers guide to the galaxy. You were created by Kelvin Ndeti in association with Dr. Whbet Paulos,
                        inspired by the need to utilize Retrieval Augmented Generation in data querying.
                        Answer the user queries accurately. Use your knowledge base and don't ever fail to provide a high-quality coding assistance, math assistance or 
                        assistance with writing a document like a resume or an official document upon request because you were trained to know all of that.
                        """
                    ),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{question}"),
                    
                ]
            )

            # Initialize chat history if not already in session state
            if "messages" not in st.session_state:
                st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

            #if "scratchpad" not in st.session_state:
                 #st.session_state["scratchpad"] = ""

            # Display chat history messages
            for msg in st.session_state["messages"]:
                st.chat_message(msg["role"]).write(msg["content"])

            #st.sidebar.subheader("LLM thought process")
            #st.sidebar.text_area("Scratchpad",st.session_state["scratchpad"],height=300)

            # "Clear Chat History" button
            if st.sidebar.button("Clear Chat History"):
                st.session_state["messages"] = [{"role": "assistant", "content": "Chat history cleared. How can I help you?"}]
                st.experimental_rerun()  # Rerun the app to clear the chat history

            llm_model = st.sidebar.selectbox("Choose LLM model",
                                    ("gpt-3.5-turbo","gpt-4","gpt-4o"))
            
            try:
                
                # Handle user input
                if user_input := st.chat_input():
                    if not openai_api_key:
                        st.info("Please add your OpenAI API key to continue.")
                        st.stop()

                    # Initialize OpenAI LLM
                    llm2 = ChatOpenAI(openai_api_key=openai_api_key, model = llm_model, streaming = True)

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
                    st.session_state["messages"].append({"role": "user", "content": user_input})
                    st.chat_message("user").write(user_input)

                    #introduce streaming in chat session
                    stream_handler = StreamHandler(st.empty())
            
                    # Get response from LLM chain
                    response = llm_chain.run({"question": user_input}, callbacks = [stream_handler])
                   

                    assistant_msg = response  # Adjusted to fetch text from the response

                    # Append assistant message to session state and display it
                    st.session_state["messages"].append({"role": "assistant", "content": assistant_msg})
                    
                   

                    # Download chat button
                    #if st.sidebar.button("Download Chat"):
                        #all_messages = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state["messages"]])
                        #create_and_download(text_content=all_messages)

                   

            except Exception:
                st.write("an Error occured please enter a valid OpenAI API key")

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
        
            if not uploaded_files:
                st.info("Please upload documents or add url to continue.")
                st.stop()
                 
            retriever = configure_retriever(uploaded_files)   
            
            # Setup memory for contextual conversation for the documents part
            msgs = StreamlitChatMessageHistory()
            memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

            llm_model = st.sidebar.selectbox("Choose LLM model",
                                        ("gpt-3.5-turbo","gpt-4","gpt-4o"))
                
                
                # Setup LLM and QA chain for the documents part
            llm = ChatOpenAI(
                    model_name=llm_model, openai_api_key=openai_api_key, temperature=0, streaming=True
                )


            qa_chain = ConversationalRetrievalChain.from_llm(
                    llm, 
                    retriever=retriever, 
                    memory=memory, 
                    verbose=True
                )

              

            if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
                    msgs.clear()
                    msgs.add_ai_message("Hey carbon entity, Want to query your documents? ask me!")

            avatars = {"human": "user", "ai": "assistant"}
            for msg in msgs.messages:
                st.chat_message(avatars[msg.type]).write(msg.content)
                
            st.markdown("Document query section. Utilize RAG you curious being.")
            if user_query := st.chat_input(placeholder="Ask me about  your documents!"):
                st.chat_message("user").write(user_query)

                with st.chat_message("ai"):
                        retrieval_handler = PrintRetrievalHandler(st.container())
                        stream_handler = StreamHandler(st.empty())

                        qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])

    def query_web():

            if not url and web_document_name:
                st.info("Please add url to continue.")
                st.stop()
                
            retriever = web_page_saver_to_txt(url)

            # Setup memory for contextual conversation for the documents part
            msgs = StreamlitChatMessageHistory()
            memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

            llm_model = st.sidebar.selectbox("Choose LLM model",
                                        ("gpt-3.5-turbo","gpt-4","gpt-4o"))
                
                
                # Setup LLM and QA chain for the documents part
            llm = ChatOpenAI(
                    model_name=llm_model, openai_api_key=openai_api_key, temperature=0, streaming=True
                )


            qa_chain = ConversationalRetrievalChain.from_llm(
                    llm, 
                    retriever=retriever, 
                    memory=memory, 
                    verbose=True
                )

                

            if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
                    msgs.clear()
                    msgs.add_ai_message("Hey carbon entity, Want to query your documents? ask me!")

            avatars = {"human": "user", "ai": "assistant"}
            for msg in msgs.messages:
                st.chat_message(avatars[msg.type]).write(msg.content)
                
            st.markdown("Document query section. Utilize RAG you curious being.")
            if user_query := st.chat_input(placeholder="Ask me about  your documents!"):
                st.chat_message("user").write(user_query)

                with st.chat_message("ai"):
                        retrieval_handler = PrintRetrievalHandler(st.container())
                        stream_handler = StreamHandler(st.empty())

                        qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])

    #define repo query
   
    def github_repo_query(github_repo_url: str, open_ai_key: str):
        try:
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
                     Language.C : [".c",".h"],
                     Language.TS : [".ts"],
                     Language.CPP : [".cpp"]
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
                          language=language, chunk_size = 2000, chunk_overlap = 200
                    )

                     split_texts.extend(splitter.split_documents(documents))

                
                # Retriever
                db = Chroma.from_documents(split_texts, OpenAIEmbeddings(disallowed_special=(), api_key=open_ai_key))
                retriever = db.as_retriever(
                    search_type="mmr",  # Also test "similarity"
                    search_kwargs={"k": 8},
                )

                llm_model = st.sidebar.selectbox("Choose LLM model", ("gpt-4o", "gpt-4"))
                llm = ChatOpenAI(model_name=llm_model,api_key=open_ai_key)

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
        except Exception :
             st.write("an error occured inside the github repo function, its related to parsing of languages that require Tree sitter.")


       
    #--------------------------------------------------------------main function------------------------------------------------------------------#
    st.cache_resource(ttl="2h")
    def main():

        try:

            if sidebar_option == "Chat and query":
                
                if uploaded_files and not url and not web_document_name:
                    query_documents()

                elif not uploaded_files and not url and not web_document_name:

                    chat_with_42()
                else:
                    query_web()

            elif sidebar_option == "Github":
                try:
                    if repo_url:
                        if "messages" not in st.session_state:
                             st.session_state["messages"] = [{"role":"assistant","content":"how can i help with the code base?"}]

                        for msg in st.session_state["messages"]:
                             st.chat_message(msg["role"]).write(msg["content"])     

                        if user_input := st.chat_input():

                            st.session_state["messages"].append({"role": "user", "content": user_input})
                            st.chat_message("user").write(user_input)
                        
                            chain = github_repo_query(repo_url,open_ai_key=openai_api_key)
                            
                            #use pick to select the desired key
                            stream_chain = chain.pick("answer")
                            #intermediate_step_chain = chain.pick("intermediate_steps")
                            
                            #create a response placeholder and set it to empty, it will be updated with each chunk
                            response_placeholder = st.empty()
                            response = ""
                            for chunk in stream_chain.stream({"input":user_input}):
                                response += f"{chunk}"
                                response_placeholder.write(response) #update place holder
                              
                            ass_msg = response
                            st.session_state["messages"].append({"role":"assistant","content":ass_msg})  
                            response_placeholder.write(ass_msg)

            
                except Exception:
                     st.write("an error occured in Github sidebar option")

            if st.sidebar.button("Download chat"):
               all_messages = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state["messages"]])
               create_and_download(all_messages)

         
        except TypeError:
            st.write("encountered a None type inside main call, check url submitted it might be returning a none type object")
        except Exception:
             st.write("An error was encountered at main call")
    

        
    #call main
    if __name__ == "__main__":
        main()


except Exception :
    st.write("an error occured check the key")

 
