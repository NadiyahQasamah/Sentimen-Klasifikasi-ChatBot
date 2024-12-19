import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# path_file = "C:/Users\nadiy\Documents\Semester 5\Stupen\api_key.xlsx"
# API_KEY = pd.read_excel(path_file)["api_key"][1]

API_KEY = "API_KEY" #API_KEY anda

def chat(contexts, history, question):
    llm = ChatGoogleGenerativeAI( 
        model="gemini-1.5-flash",
        temperature=0.7,
        api_key=API_KEY
        )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. You can use the data provided to answer questions about the dataset. Answer in Indonesian",
            ),
            ("human", "This is the data : {contexts}\nUse this chat history to generate relevant answer from recent conversation: {history}\nUser question : {question}"),
        ]
    )
    
    chain = prompt | llm
    completion = chain.invoke(
        {
            "contexts": contexts,
            "history": history,
            "question": question,
        }
    )

    answer = completion.content
    input_tokens = completion.usage_metadata['input_tokens']
    completion_tokens = completion.usage_metadata['output_tokens']

    result = {}
    result["answer"] = answer
    result["input_tokens"] = input_tokens
    result["completion_tokens"] = completion_tokens
    return result

# contexts = pd.read_csv("C:/Users/USER/MSIB Batch 7/51. End-to-End Chatbot Implementation/Message Group - Product.csv").to_string() #dijadikan string/teks
# history = '' # karena pertama kali ngechat
# question = 'Coba rekomendasikan saya produk dengan diskon tertinggi'
# response = chat(contexts, history, question) # respon invoke
# print(response)

# Memulai streamlit dengan judul
st.header("Sentimen klasifikasi ChatBot", divider=True)

## Contexts
# Memberikan konteks data eksternal yang nanti dimasukan/input
# untuk mengupload file
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xls", "xlsx"])


if uploaded_file is not None:
    try:
        # Periksa ekstensi file dan muat file sesuai dengan itu
        #kondisi 1 file csv
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file) #membaca csv
            df = df.drop_duplicates() #drop file
            contexts = df.to_string() #ubah ke string
        #kondisi 2 xls dan xlsx
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
            df = df.drop_duplicates()
            contexts = df.to_string()
        #kondisi 3 klo upload yg kecuali diatas/file yg diupload ga support
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            df = None
            contexts = ''
        
        on = st.toggle("Activate feature")
        
        if on:
            st.write("Feature activated!")
            st.write(f"File has shape : {df.shape}") #file yang sudah di drop dibaca punya brp baris & kolom
            
            plt.figure(figsize=(11, 6))
            sns.countplot(x='Actual', data=df,  palette='pastel')
            plt.title('Sentiment Distribution')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
            
            st.pyplot(plt)
            
            option = st.selectbox(
                "Select the sentiment you want to search for :",
                ("All","positive", "negative", "neutral"),
                index=None
           )
        
            if option == 'All':
                chose = df
            else:
                chose = df[df['Actual'] == option]
            
            if chose.empty:
                st.warning(f"No data found for sentiment: '{option}'")
            else:
                st.write(f"show '{option}' sentiment:")
                st.dataframe(chose)
        else:
            st.info("feature not activated")
                
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}") #salah memasukan file lalu mencoba memasukan ulang file
else:
    st.info("No file uploaded yet.") #informasi file blm ada yg diupload

## History
# inisialisasi chat history/menampung history percakapan
if "messages" not in st.session_state: #session = ketika browser blm di reload/dimatiin masi ada di 1 sesi yg sama
    st.session_state.messages = [] #selama input dan answer ada dimasukan semua ke history

# Menampilkan pesan obrolan dari riwayat saat menjalankan ulang aplikasi
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Bereaksi terhadap masukan pengguna/menampung history
if prompt := st.chat_input("What is up?"): #Menampilkan history selama apa aja yg sudah diinput dan dijawab
    # Dapatkan riwayat obrolan jika tidak Null
    messages_history = st.session_state.get("messages", [])[-4:] #Menampung semua history
    history = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in messages_history]) or " "

    # Menampilkan pesan pengguna dalam wadah pesan obrolan/ diforum chat
    st.chat_message("user").markdown(prompt)
    # Tambahkan pesan user ke riwayat obrolan (role = asal dri user, content = isi pesan pengguna)
    st.session_state.messages.append({"role": "user", "content": prompt}) #ditambahkan dengan append

    #sesuaikan dengan definisi diatas
    response = chat(contexts, history, prompt)
    answer = response["answer"]
    input_tokens = response["input_tokens"]
    completion_tokens = response["completion_tokens"]

    # Menampilkan respons asisten dalam wadah pesan obrolan
    with st.chat_message("assistant"):
        st.markdown(answer)
        container = st.container(border=True)
        container.write(f"Input Tokens : {input_tokens}")
        container.write(f"Completion Tokens: {completion_tokens}")
        
    # TTampilkan history obrolan
    with st.expander("See Chat History"):
        st.write("**History Chat:**")
        st.code(history)

    # Menambahkan respons asisten ke histori obrolan
    st.session_state.messages.append({"role": "assistant", "content": answer})