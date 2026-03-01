import streamlit as st
from generate import load_model, generate, get_device, find_latest_checkpoint

st.set_page_config(page_title="Taylor Swift Lyrics Generator", page_icon="🎤")

st.title("Taylor Swift Lyrics Generator")
st.markdown("Generate Taylor Swift-style lyrics using a character-level LSTM trained on her discography.")


@st.cache_resource
def init_model():
    device = get_device()
    checkpoint_path = find_latest_checkpoint()
    if not checkpoint_path:
        return None, None, None, None
    model, char_to_idx, idx_to_char = load_model(checkpoint_path, "checkpoints/vocab.json", device)
    return model, char_to_idx, idx_to_char, device


model, char_to_idx, idx_to_char, device = init_model()

if model is None:
    st.error("No model checkpoint found in `checkpoints/`. Train the model first with `python train.py`.")
    st.stop()

seed = st.text_input("Seed phrase", value="I remember when")
temperature = st.slider("Temperature", min_value=0.2, max_value=1.5, value=0.8, step=0.1,
                        help="Lower = more predictable, higher = more creative")
length = st.slider("Length (characters)", min_value=100, max_value=1000, value=500, step=50)

if st.button("Generate"):
    if not seed.strip():
        st.warning("Please enter a seed phrase.")
    else:
        with st.spinner("Writing lyrics..."):
            text = generate(model, seed, char_to_idx, idx_to_char, device,
                            length=length, temperature=temperature)
        st.subheader("Generated Lyrics")
        st.text(text)
