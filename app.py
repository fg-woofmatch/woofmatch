import streamlit as st
import os
from dotenv import load_dotenv

# 1. Configuration & API
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

if not api_key:
    st.error("🚨 Clé API manquante ! Vérifie tes Secrets Streamlit.")
    st.stop()

st.set_page_config(page_title="WOOF MATCH", page_icon="🦴")

# 2. CSS : Look épuré, Rose Pâle et Interface Pro
st.markdown("""
    <style>
    /* Fond global blanc */
    .stApp { background-color: #FFFFFF !important; }
    
    /* CIBLAGE DU BLOC BAS */
    [data-testid="stBottom"] {
        background-color: #FFFFFF !important;
        border-top: none !important;
    }
    
    [data-testid="stBottom"] > div {
        background-color: #FFFFFF !important;
        background-image: none !important;
    }

    /* BARRE DE SAISIE AGRANDIE */
    [data-testid="stChatInput"] {
        background-color: #FFFFFF !important;
        padding-bottom: 30px !important;
    }

    [data-testid="stChatInput"] textarea {
        background-color: #FDF6E3 !important;
        border: 1px solid #E6E0D0 !important;
        border-radius: 15px !important;
        color: #1A1A1A !important;
        caret-color: #1A1A1A !important;
        padding: 15px !important;
        min-height: 60px !important;
    }

    /* Placeholder visible */
    [data-testid="stChatInput"] textarea::placeholder {
        color: #888888 !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] button {
        background-color: #1A1A1A !important;
        border-radius: 50% !important;
        bottom: 12px !important;
    }

    /* NETTOYAGE UI */
    [data-testid="stHeader"], footer { visibility: hidden; height: 0px; }
    #MainMenu { visibility: hidden; }
    [data-testid="collapsedControl"] { display: none; }
    section[data-testid="stSidebar"] { display: none; }

/* --- COULEUR DE L'AVATAR USER (Cercle Rose) --- */
    [data-testid="stChatMessageAvatarUser"] {
        background-color: #FFD1DC !important;
        color: #1A1A1A !important;
    }

    /* --- FORCE LE FOND DE LA BULLE USER (Adieu le gris foncé) --- */
    /* On cible le conteneur de message qui contient l'avatar user */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #FFF0F5 !important; /* Rose très pâle */
        border: 1px solid #FFD1DC !important; /* Petite bordure pour la structure */
    }

    /* On force aussi le texte à l'intérieur en noir au cas où */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div {
        color: #1A1A1A !important;
    }

    /* Pour l'assistant, on peut aussi forcer un fond blanc pur ou gris très clair */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background-color: #F8F9FA !important;
        border: 1px solid #E6E0D0 !important;
    }

    /* Texte global noir */
    html, body, .stMarkdown, p, h1, span { color: #1A1A1A !important; }
    </style>
    """, unsafe_allow_html=True)

# 3. Base de Données
@st.cache_resource
def init_db():
    if not os.path.exists("chiens.txt"):
        st.error("Fichier chiens.txt introuvable !")
        return None
    loader = TextLoader("chiens.txt", encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    final_docs = splitter.split_documents(docs)
    return FAISS.from_documents(final_docs, OpenAIEmbeddings(openai_api_key=api_key))

db = init_db()

# 4. EN-TÊTE
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("logo.png", use_container_width=True)

st.markdown("<h1 style='text-align: center;'>🐾 WOOF MATCH</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-style: italic;'>L'expert qui déniche votre compagnon idéal, sans filtre.</p>", unsafe_allow_html=True)
st.markdown("---")

# 5. INITIALISATION & HISTORIQUE
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bienvenue dans mon bureau. 🕶️ Trouver le bon chien, c'est du sérieux. Dis-moi tout : tu vis en ville ou au grand air ? Tu es plutôt marathon ou canapé ?"}]

for message in st.session_state.messages:
    if message["role"] == "assistant":
        avatar = "logo.png"
    else:
        avatar = "🙋‍♂️"
    
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# 6. LOGIQUE IA
if db:
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=api_key)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Tu es 'Le Parrain des Chiens', un expert canin légendaire, drôle et psychologue. 🕶️
        Tu maries des âmes. Découvre le logement, le sport, et les contraintes.
        Affiche l'image ainsi : ![nom](lien).
        Contexte : {context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (prompt | llm | StrOutputParser())

    if user_input := st.chat_input("Demande à Woof Match..."):
        # --- ICI L'INDENTATION EST CORRIGÉE ---
        st.chat_message("user", avatar="🙋‍♂️").markdown(user_input)
        
        context_docs = retriever.invoke(user_input)
        formatted_context = format_docs(context_docs)
        chat_history = st.session_state.messages 

        with st.chat_message("assistant", avatar="logo.png"):
            response = chain.invoke({
                "context": formatted_context,
                "question": user_input,
                "history": chat_history
            })
            st.markdown(response)
        
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response})
