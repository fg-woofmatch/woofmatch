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

# 2. CSS pour un look épuré et centré
st.markdown("""
    <style>
    .stApp { background-color: #FFFFFF !important; }
    [data-testid="stHeader"] { background-color: rgba(0,0,0,0) !important; }
    
    /* On cache la sidebar complètement */
    [data-testid="collapsedControl"] { display: none; }
    section[data-testid="stSidebar"] { display: none; }

    html, body, .stMarkdown, p, h1, span {
        color: #1A1A1A !important;
    }
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

# 5. HISTORIQUE
    for message in st.session_state.messages:
    # On définit l'icône selon le rôle
    # "assistant" devient un emoji chien, "user" reste par défaut ou ce que tu veux
    avatar = "logo.png" if message["role"] == "assistant" else None
    
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Bienvenue dans mon bureau. 🕶️ Trouver le bon chien, c'est du sérieux. Dis-moi tout : tu vis en ville ou au grand air ? Tu es plutôt marathon ou canapé ?"}]

    for message in st.session_state.messages:
    # On définit l'icône selon le rôle
    # "assistant" devient un emoji chien, "user" reste par défaut ou ce que tu veux
    avatar = "logo.png" if message["role"] == "assistant" else None
    
    with st.chat_message(message["role"], avatar="logo.png"):
        st.markdown(message["content"])

# 6. LOGIQUE IA (PROMPT AMÉLIORÉ)
if db:
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=api_key)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Tu es 'Le Parrain des Chiens', un expert canin légendaire, drôle et psychologue. 🕶️
        
        TON RÔLE : 
        Tu ne listes pas des chiens, tu maries des âmes. Tu n'as plus de formulaires (sidebar), tu dois TOUT découvrir par la discussion.

        PROTOCOLE D'ENQUÊTE :
        1. FLEXIBILITÉ : Si l'utilisateur pose une question directe (ex: "C'est quoi un chien hypoallergénique ?"), réponds précisément avec ton expertise, puis relance l'enquête.
        2. PRO-ACTIVITÉ : Tu dois impérativement connaître ces points avant de proposer quoi que ce soit :
           - Le logement (étage, ascenseur, jardin ?).
           - Le sport (actif ou sédentaire ?).
           - Les contraintes (enfants, chats, allergies, temps seul).
        3. ANTI-PRÉCIPITATION : Ne donne JAMAIS de race au premier message. Analyse d'abord l'humain.
        4. IMAGE : Quand tu proposes enfin 2 races max, affiche impérativement l'image avec la syntaxe : ![nom](lien).

        TON STYLE : Direct, plein d'emojis, avec des punchlines de coach, mais une bienveillance totale.
        
        Contexte : {context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (prompt | llm | StrOutputParser())

    if user_input := st.chat_input("Parle au Parrain..."):
        st.chat_message("user").markdown(user_input)
        
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
