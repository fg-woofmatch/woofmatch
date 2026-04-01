import streamlit as st
import os
from dotenv import load_dotenv
# Juste après load_dotenv()
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# On vérifie si la clé est bien chargée
if not api_key:
    st.error("🚨 Clé API manquante ! Vérifie ton fichier .env")
    st.stop()
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(
    page_title="WOOF MATCH", 
    page_icon="🦴", 
    initial_sidebar_state="expanded" 
)

# 2. CSS Corrigé : On ne cache plus TOUT le header
st.markdown("""
    <style>
    /* On cache la barre de décoration du haut mais PAS le bouton sidebar */
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
        color: #1A1A1A !important;
    }
    
    /* On s'assure que le bouton de la sidebar (les 3 traits ou la flèche) est NOIR */
    [data-testid="stSidebarCollapseButton"] {
        color: #1A1A1A !important;
        background-color: #F0E6D2 !important;
        inset: 10px auto auto 10px !important;
    }

    .stApp { background-color: #F0E6D2 !important; }
    [data-testid="stSidebar"] { background-color: #F0E6D2 !important; }

    /* Tes cases encadrées */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
        background-color: white !important;
        padding: 15px !important;
        border-radius: 12px !important;
        border: 1px solid #E6E0D0 !important;
        margin-bottom: 15px !important;
    }

    html, body, [data-testid="stWidgetLabel"], .stMarkdown, p, h1, label, span {
        color: #1A1A1A !important;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Initialisation Base de Données
@st.cache_resource
def init_db():
    if not os.path.exists("chiens.txt"):
        st.error("Crée ton fichier chiens.txt d'abord !")
        return None
    loader = TextLoader("chiens.txt", encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    final_docs = splitter.split_documents(docs)
    return FAISS.from_documents(final_docs, OpenAIEmbeddings())

db = init_db()

# 4. SIDEBAR : Structure demandée
with st.sidebar:
    # A. Image Drôle
    st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNHJndmI3Z3NueHpxZ3R4Z3R4Z3R4Z3R4Z3R4Z3R4Z3R4Z3R4JmVwPXYxX2ludGVybmFsX2dpZl9ieV9pZCZjdD1n/3ndAvMC5LFPNMCzq7m/giphy.gif", 
             use_container_width=True)
    
    st.title("🔍 Tes filtres Wouf")
    
    # B. Étage du logement (Case 1)
    with st.container():
        etage = st.select_slider("Étage du logement 🏢", options=["0", "1", "2", "3", "4", "5", "6", "7", "8+"])

    # C. Ascenseur (Case 2)
    with st.container():
        ascenseur = st.toggle("Y'a t'il un ascenseur ? 🛗", value=True)

    # D. Motivation Sortie (Case 3)
    with st.container():
        motivation = st.select_slider("Motivation pour sortir :", 
                                     options=["Canapé-Pizza 🍕", "Marcheur du dimanche 🌳", "Ultra-Trail 🏃‍♂️"])

    # E. Allergique (Case 4)
    with st.container():
        allergie = st.toggle("Allergique ? 🤧")

# 5. ZONE DE CHAT PRINCIPALE
# Affichage du logo centré (use_container_width=True l'étire, 
# on peut aussi utiliser une colonne pour le garder petit au centre)
col1, col2, col3 = st.columns([1, 10, 1])
with col2:
    st.image("logo.png", use_container_width=True)

# Titre centré en HTML
st.markdown("<h1 style='text-align: center;'>🐾 WOOF MATCH</h1>", unsafe_allow_html=True)

# Texte de description centré en HTML
st.markdown("<p style='text-align: center;'>Dis-moi ce que tu cherches, et je te trouverai le toutou idéal !</p>", unsafe_allow_html=True)

# Ligne de séparation optionnelle pour faire propre
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Salut ! Je suis ton expert canin. Quel genre de compagnon cherches-tu ? 🐕"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. LOGIQUE IA (LCEL)
if db:
    retriever = db.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    # Calcul escaliers pour le système
    has_stairs = (etage != "0") and (not ascenseur)

# Définition du Prompt "Coach Canin"
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""Tu es 'Le Parrain des Chiens', un expert canin plein d'humour et de sagesse. 
        
        TON OBJECTIF : 
        Trouver la race parfaite pour l'utilisateur, mais tu ne dois JAMAIS donner de nom de race lors du premier échange.
        
        CONSIGNE IMAGE : 
        Dans le contexte, chaque race a une ligne 'IMAGE'. 
        Quand tu proposes une race à l'utilisateur, tu DOIS afficher l'image en utilisant la syntaxe Markdown suivante : 
        ![nom de la race](lien_de_l_image)
        
        RÈGLES DE CONVERSATION :
        1. ANALYSE d'abord les infos de la sidebar (Etage: {etage}, Ascenseur: {ascenseur}, Sport: {motivation}).
        2. PHASE D'ENQUÊTE : Tant que tu n'as pas au moins 2 ou 3 réponses précises sur le quotidien de l'utilisateur, refuse poliment de donner une race.
        3. TES QUESTIONS : Pose 2 ou 3 questions ciblées (ex: présence d'enfants, temps passé seul à la maison, budget pour le toilettage, jardin clos ou non).
        4. REBONDI sur ses filtres : Si l'utilisateur est 'Canapé-Pizza', demande-lui s'il est prêt à changer pour un chien qui demande un peu d'exercice ou s'il veut un vrai paresseux.
        5. RÉPONSE FINALE : Une fois que tu as assez d'infos, propose 2 races max issues du contexte, avec tes punchlines habituelles et affiche TOUJOURS l'image correspondante juste en dessous du nom de la race.. 🦴🕶️
        
        Contexte des races : {{context}}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

# 1. On définit la fonction de formatage
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 2. La chaîne est simplifiée (on ne met pas le lambda session_state ici)
    chain = (
        prompt | llm | StrOutputParser()
    )

    # 3. --- ZONE D'INTERACTION ---
    if user_input := st.chat_input("Pose ta question ici..."):
        # Afficher le message utilisateur
        st.chat_message("user").markdown(user_input)
        
        # ON RÉCUPÈRE LES INFOS ICI (Thread principal)
        context_docs = retriever.invoke(user_input)
        formatted_context = format_docs(context_docs)
        
        # ON EXTRAIT L'HISTORIQUE ICI
        chat_history = st.session_state.messages 

        with st.chat_message("assistant"):
            # On envoie tout au Coach
            response = chain.invoke({
                "context": formatted_context,
                "question": user_input,
                "history": chat_history # On lui donne l'historique "en main propre"
            })
            st.markdown(response)
        
        # Sauvegarde
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response})
