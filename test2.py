import streamlit as st

# App title
st.title("Text Combiner App")

# Input fields for two texts
text1 = st.text_input("Enter the first text:", placeholder="Type something here...")
text2 = st.text_input("Enter the second text:", placeholder="Type something here...")

# Button to combine texts
if st.button("Combine Texts"):
    if text1 and text2:
        # Combine the texts and output the result
        combined_text = f"{text1} {text2}"
        st.subheader("Combined Text:")
        st.write(combined_text)
    else:
        st.warning("Please provide both texts to combine.")