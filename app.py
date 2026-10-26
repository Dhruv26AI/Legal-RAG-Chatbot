# python -m streamlit run app.py

import streamlit as st
from retriever import retrieve
from generate import generate_answer

st.set_page_config(
    page_title="BNSS Legal Assistant",
    page_icon="⚖️",
    layout="centered"
)

st.title("BNSS Legal Assistant")
st.caption("Ask questions about the Bharatiya Nagarik Suraksha Sanhita, 2023.")
st.warning(
    "For legal information only. Verify important legal matters "
    "with a qualified legal professional."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about BNSS...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching BNSS and generating an answer..."):
            try:
                chunks = retrieve(question)
                answer = generate_answer(question, chunks)
                st.markdown(answer)
            except Exception as error:
                answer = f"Something went wrong: {error}"
                st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
