import os
import streamlit as st
from dotenv import load_dotenv
from pytube import YouTube
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable
)
# Only import TextLoader from langchain
# from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
# Import YoutubeLoader only from langchain_community
from langchain_community.document_loaders import YoutubeLoader



# ✅ Load .env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ✅ Create the LLM
llm = ChatGroq(
    api_key=groq_api_key,
    model_name="llama3-8b-8192"
)

# ✅ Streamlit App
st.title("🎓 AI Powered YouTube Tutor")
st.write("Ask questions from YouTube lecture videos using Groq + LangChain")

# ✅ Get transcript
def get_transcript_from_youtube(url):
    try:
        video_id = YouTube(url).video_id
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript_data = transcript_list.find_transcript(["en"])
        text = " ".join([item['text'] for item in transcript_data.fetch()])
        return text
    except TranscriptsDisabled:
        st.error("🚫 Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("🚫 No transcript found.")
    except VideoUnavailable:
        st.error("🚫 Video unavailable.")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
    return None

# ✅ Save transcript to file
def save_transcript_to_file(text, filename="transcript.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)

# ✅ UI input
video_url = st.text_input("🔗 Enter YouTube URL")

if st.button("📄 Process Video"):
    if video_url:
        transcript_text = get_transcript_from_youtube(video_url)
        if transcript_text:
            save_transcript_to_file(transcript_text)

            loader = TextLoader("transcript.txt", encoding="utf-8")
            documents = loader.load()

            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

            st.session_state.qa_chain = qa_chain
            st.success("✅ Transcript processed. You can now ask questions!")
        else:
            st.warning("⚠️ Invalid YouTube URL or no transcript found.")
    else:
        st.warning("Please enter a YouTube URL.")

# ✅ QA input
if "qa_chain" in st.session_state:
    user_question = st.text_input("❓ Ask a question based on the video")
    if user_question:
        answer = st.session_state.qa_chain.run(user_question)
        st.markdown("💡 **Answer:**")
        st.write(answer)




# import streamlit as st
# from youtube_qa import get_transcript_from_youtube, get_qa_chain

# st.title("🎓 YouTube Tutor Chatbot")
# video_url = st.text_input("Enter a YouTube video URL")

# if video_url:
#     with st.spinner("Processing video..."):
#         docs = get_transcript_from_youtube(video_url)
#         retriever, chain = get_qa_chain(docs)
#     st.success("Ready! Ask your question below 👇")

#     question = st.text_input("Ask a question about the video")
#     if question:
#         docs = retriever.get_relevant_documents(question)
#         response = chain.run(input_documents=docs, question=question)
#         st.write("📘 Answer:", response)
