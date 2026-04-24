import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import os
import pickle

st.set_page_config(page_title="PsyHelper", page_icon="🧠", layout="centered")

# ====================== BANNER BETA ======================
st.markdown("""
<style>
    .beta-banner {
        background: linear-gradient(90deg, #4338ca, #6366f1);
        color: white;
        padding: 14px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    .feedback-box {
        background: #1f2937;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #6d28d9;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="beta-banner">🔬 PsyHelper - VERSIONE BETA<br>Stiamo testando l’app. Il tuo feedback è molto prezioso!</div>', unsafe_allow_html=True)

# ====================== DESIGN ======================
st.markdown("""
<style>
    .stApp {background: #0a0a0a;}
    h1 {color: #c4b5fd; text-align: center;}
    .subtitle {text-align: center; color: #a5b4fc; font-size: 1.3em;}
    .stChatMessageUser {background-color: #1f2937; color: #e0e7ff; border-radius: 20px; padding: 14px 18px;}
    .stChatMessageAssistant {background-color: #6d28d9; color: white; border-radius: 20px; padding: 14px 18px;}
    .stButton > button {height: 2.1em !important; font-size: 0.82em !important; padding: 0 10px !important; background: #3b0764; color: #c4b5fd;}
    .mindfulness-box {background: #1f2937; border: 1px solid #6d28d9; color: #e0e7ff; padding: 18px; border-radius: 16px; margin: 12px 0;}
</style>
""", unsafe_allow_html=True)

# ====================== SALVATAGGIO ======================
save_dir = os.path.expanduser("~/psyhelper_data")
os.makedirs(save_dir, exist_ok=True)

def load_data():
    try:
        with open(f"{save_dir}/profile.pkl", "rb") as f: st.session_state.profile = pickle.load(f)
        with open(f"{save_dir}/messages.pkl", "rb") as f: st.session_state.messages = pickle.load(f)
        st.session_state.onboarding_done = True
    except:
        st.session_state.profile = {}
        st.session_state.messages = []
        st.session_state.onboarding_done = False

def save_data():
    with open(f"{save_dir}/profile.pkl", "wb") as f: pickle.dump(st.session_state.profile, f)
    with open(f"{save_dir}/messages.pkl", "wb") as f: pickle.dump(st.session_state.messages, f)

if "profile" not in st.session_state: load_data()
if "show_mindfulness" not in st.session_state: st.session_state.show_mindfulness = False

# ====================== LLM ======================
llm = ChatOllama(model="llama3.2:3b", temperature=0.55, num_ctx=8192)

# ====================== ONBOARDING (2 passi) ======================
def show_onboarding():
    st.title("🧠 PsyHelper")
    st.markdown("<p class='subtitle'>Prima di iniziare, vorrei conoscerti meglio</p>", unsafe_allow_html=True)

    if st.session_state.get("onboarding_step", 1) == 1:
        st.subheader("Passo 1 di 2")
        with st.form("step1"):
            st.session_state.profile["nome"] = st.text_input("Come ti chiami?", placeholder="Es. Marco")
            st.session_state.profile["età"] = st.number_input("Età", 14, 90, 30)
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.profile["umore_generale"] = st.selectbox("Umore generale", ["Sereno", "Ansioso", "Triste", "Irritabile", "Altro"])
                st.session_state.profile["umore_intensità"] = st.slider("Intensità malessere (1-10)", 1, 10, 6)
            with col2:
                st.session_state.profile["stress"] = st.slider("Livello stress (1-10)", 1, 10, 6)
            if st.form_submit_button("Continua →", use_container_width=True):
                st.session_state.onboarding_step = 2
                st.rerun()
    else:
        st.subheader("Passo 2 di 2")
        with st.form("step2"):
            st.session_state.profile["sonno"] = st.selectbox("Sonno ultimamente?", ["Dormo bene", "Faccio fatica ad addormentarmi", "Mi sveglio spesso", "Rimugino e non dormo", "Altro"])
            st.session_state.profile["pensieri"] = st.text_area("Cosa ti passa spesso per la testa?")
            st.session_state.profile["trigger"] = st.text_area("In quali situazioni ti senti peggio?")
            st.session_state.profile["obiettivi"] = st.text_area("Cosa vorresti migliorare?")
            st.session_state.profile["motivazione"] = st.slider("Motivazione (1-10)", 1, 10, 7)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("← Indietro", use_container_width=True):
                    st.session_state.onboarding_step = 1
                    st.rerun()
            with col2:
                if st.form_submit_button("Inizia con PsyHelper 💜", use_container_width=True):
                    st.session_state.onboarding_done = True
                    save_data()
                    st.rerun()

# ====================== PROMPT (versione concreta) ======================
def get_response(user_input):
    profile = st.session_state.profile
    nome = profile.get("nome") or ""
    profile_text = "\n".join([f"- {k}: {v}" for k, v in profile.items() if k != "nome" and v])

    system_prompt = f"""Sei PsyHelper, un assistente pratico e concreto.
Nome utente: {nome}
Profilo: {profile_text}
Sii diretto, dai consigli pratici e varia le risposte."""
    
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), MessagesPlaceholder("history"), ("human", "{input}")])
    chain = prompt | llm
    chain_with_history = RunnableWithMessageHistory(chain, lambda x: ChatMessageHistory(), input_messages_key="input", history_messages_key="history")
    response = chain_with_history.invoke({"input": user_input}, config={"configurable": {"session_id": "psyhelper_user"}})
    return response.content

# ====================== APP ======================
if not st.session_state.onboarding_done:
    show_onboarding()
else:
    nome = st.session_state.profile.get("nome")
    st.title("🧠 PsyHelper")
    st.markdown(f"<p class='subtitle'>Ciao {nome if nome else ''}, sono qui per aiutarti</p>", unsafe_allow_html=True)
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if user_input := st.chat_input("Scrivi qui cosa stai provando..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"): st.markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Sto pensando..."):
                reply = get_response(user_input)
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
        save_data()
    
    st.divider()
    col1, col2, col3, col4 = st.columns([1.2, 1.2, 1.2, 6])
    with col1:
        if st.button("🧘", help="Mindfulness"): st.session_state.show_mindfulness = not st.session_state.show_mindfulness
    with col2:
        if st.button("🔄", help="Nuova"): 
            st.session_state.messages = []
            save_data()
            st.rerun()
    with col3:
        if st.button("🗑️", help="Cancella"):
            if st.checkbox("Confermi?"):
                st.session_state.clear()
                for f in ["profile.pkl", "messages.pkl"]:
                    p = f"{save_dir}/{f}"
                    if os.path.exists(p): os.remove(p)
                st.rerun()

    # ====================== PULSANTE FEEDBACK ======================
    if st.button("💡 Invia Feedback sulla Beta"):
        with st.expander("Scrivi il tuo feedback", expanded=True):
            feedback = st.text_area("Cosa ti è piaciuto? Cosa vorresti migliorare? (tono, utilità, bug, idee...)", height=150)
            if st.button("Invia Feedback"):
                try:
                    with open(f"{save_dir}/feedback.txt", "a", encoding="utf-8") as f:
                        f.write(f"Nome: {nome}\nData: {time.strftime('%Y-%m-%d %H:%M')}\nFeedback: {feedback}\n{'-'*50}\n")
                    st.success("✅ Grazie mille! Il tuo feedback è stato salvato.")
                except:
                    st.error("Errore nel salvataggio del feedback.")

    if st.session_state.show_mindfulness:
        st.subheader("🧘 Esercizi di Mindfulness")
        st.caption("Scegli in base a come ti senti")
        st.markdown('<div class="mindfulness-box"><strong>Respirazione 4-7-8</strong><br>Calma ansia velocemente</div>', unsafe_allow_html=True)
        st.markdown('<div class="mindfulness-box"><strong>Grounding 5-4-3-2-1</strong><br>Riporta la mente al presente</div>', unsafe_allow_html=True)
