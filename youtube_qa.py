# import os
# import streamlit as st
# from dotenv import load_dotenv
# from pytube import YouTube
# from youtube_transcript_api import (
#     YouTubeTranscriptApi,
#     TranscriptsDisabled,
#     NoTranscriptFound,
#     VideoUnavailable
# )

# # Loaders
# from langchain_community.document_loaders import YoutubeLoader
# from langchain.document_loaders import TextLoader

# # Splitter
# from langchain.text_splitter import CharacterTextSplitter

# # Vectorstore & embeddings (community versions to avoid deprecation warnings)
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings

# # QA & LLM
# from langchain.chains import RetrievalQA
# from langchain_groq import ChatGroq


# # ✅ Load .env
# load_dotenv()
# groq_api_key = os.getenv("GROQ_API_KEY")

# # ✅ Create the LLM
# llm = ChatGroq(
#     api_key=groq_api_key,
#     model_name="llama3-8b-8192"
# )

# # ✅ Streamlit App
# st.title("🎓 AI Powered YouTube Tutor")
# st.write("Ask questions from YouTube lecture videos using Groq + LangChain")

# # ✅ Get transcript
# def get_transcript_from_youtube(url):
#     try:
#         video_id = YouTube(url).video_id
#         transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
#         transcript_data = transcript_list.find_transcript(["en"])
#         text = " ".join([item['text'] for item in transcript_data.fetch()])
#         return text
#     except TranscriptsDisabled:
#         st.error("🚫 Transcripts are disabled for this video.")
#     except NoTranscriptFound:
#         st.error("🚫 No transcript found.")
#     except VideoUnavailable:
#         st.error("🚫 Video unavailable.")
#     except Exception as e:
#         st.error(f"❌ Unexpected error: {e}")
#     return None

# # ✅ Save transcript to file
# def save_transcript_to_file(text, filename="transcript.txt"):
#     with open(filename, "w", encoding="utf-8") as f:
#         f.write(text)

# # ✅ UI input
# video_url = st.text_input("🔗 Enter YouTube URL")

# if st.button("📄 Process Video"):
#     if video_url:
#         transcript_text = get_transcript_from_youtube(video_url)
#         if transcript_text:
#             save_transcript_to_file(transcript_text)

#             loader = TextLoader("transcript.txt", encoding="utf-8")
#             documents = loader.load()

#             splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#             docs = splitter.split_documents(documents)

#             embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
#             vectorstore = FAISS.from_documents(docs, embeddings)
#             retriever = vectorstore.as_retriever()
#             qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

#             st.session_state.qa_chain = qa_chain
#             st.success("✅ Transcript processed. You can now ask questions!")
#         else:
#             st.warning("⚠️ Invalid YouTube URL or no transcript found.")
#     else:
#         st.warning("Please enter a YouTube URL.")

# # ✅ QA input
# if "qa_chain" in st.session_state:
#     user_question = st.text_input("❓ Ask a question based on the video")
#     if user_question:
#         answer = st.session_state.qa_chain.run(user_question)
#         st.markdown("💡 **Answer:**")
#         st.write(answer)


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

# ✅ Updated imports (no deprecation warnings)
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq


# -------------------------
# ✅ Load environment
# -------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ✅ Create Groq LLM
llm = ChatGroq(
    api_key=groq_api_key,
    model_name="llama3-8b-8192"
)




# -------------------------
# ✅ Extract video ID safely
# -------------------------
def extract_video_id(url: str) -> str:
    import urllib.parse as urlparse
    parsed_url = urlparse.urlparse(url)

    video_id = None
    if parsed_url.hostname in ["youtu.be"]:
        video_id = parsed_url.path[1:]
    elif parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        query = urlparse.parse_qs(parsed_url.query)
        video_id = query.get("v", [None])[0]

    # ✅ Strip out junk like ?si= or &t=
    if video_id and "&" in video_id:
        video_id = video_id.split("&")[0]
    if video_id and "?" in video_id:
        video_id = video_id.split("?")[0]

    return video_id


# -------------------------
# ✅ Fetch transcript
# -------------------------
def get_transcript_from_youtube(url):
    try:
        video_id = YouTube(url).video_id
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        text = " ".join([item["text"] for item in transcript])
        return text

    except TranscriptsDisabled:
        st.error("🚫 Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("🚫 No transcript found for this video.")
    except VideoUnavailable:
        st.error("🚫 Video unavailable.")
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
    return None


# -------------------------
# ✅ Save transcript
# -------------------------
def save_transcript_to_file(text, filename="transcript.txt"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(text)


# -------------------------
# ✅ Streamlit App
# -------------------------
st.title("🎓 AI Powered YouTube Tutor")
st.write("Ask questions from YouTube lecture videos using Groq + LangChain")

video_url = st.text_input("🔗 Enter YouTube URL")

if st.button("📄 Process Video"):
    if video_url:
        st.info("⏳ Fetching transcript...")
        transcript_text = get_transcript_from_youtube(video_url)
        if transcript_text:
            save_transcript_to_file(transcript_text)

            loader = TextLoader("transcript.txt", encoding="utf-8")
            documents = loader.load()

            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_documents(docs, embeddings)
            retriever = vectorstore.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

            st.session_state.qa_chain = qa_chain
            st.success("✅ Transcript processed. You can now ask questions!")
        else:
            st.warning("⚠️ Invalid YouTube URL or no transcript found.")
    else:
        st.warning("Please enter a YouTube URL.")

# ✅ Question Answering
if "qa_chain" in st.session_state:
    user_question = st.text_input("❓ Ask a question based on the video")
    if user_question:
        answer = st.session_state.qa_chain.run(user_question)
        st.markdown("💡 **Answer:**")
        st.write(answer)
