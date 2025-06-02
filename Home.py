from langchain_community import embeddings
import streamlit as st
from streamlit_chat import message

from langchain.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
#loader = TextLoader("/content/disease.txt", encoding="latin-1")
#documents = loader.load()
#text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#docs = text_splitter.split_documents(documents)
embedding = GPT4AllEmbeddings()

#db=Chroma.from_documents(docs,embedding, persist_directory="/content/chroma_db")
db1=Chroma(persist_directory="/content/chroma_db",embedding_function=embedding)
llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro",google_api_key="AIzaSyDVaKDIJPx2ngKBqd18nlv1-Ynm1lDnjPk")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = ConversationalRetrievalChain.from_llm(llm=llm,chain_type='stuff',retriever=db1.as_retriever(search_kwargs={"k":2}),memory=memory)
st.title("HealthCare ChatBot 🧑🏽‍⚕️")
def conversation_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Health", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output = conversation_chat(user_input)

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

# Initialize session state
initialize_session_state()
# Display chat history
display_chat_history()

#just checking if the code is working


