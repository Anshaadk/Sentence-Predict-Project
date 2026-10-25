import streamlit as st
import numpy as np
import pickle

from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="LSTM Next Word Prediction",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 LSTM Next Word Prediction")

st.write(
    "Enter a sentence and let your trained LSTM model "
    "predict the next words automatically."
)


# ==========================================
# 2. FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "lstm_model.h5"
TOKENIZER_PATH = BASE_DIR / "tokenizer.pkl"
MAX_LEN_PATH = BASE_DIR / "max_len.pkl"


# ==========================================
# 3. LOAD MODEL, TOKENIZER AND MAX LENGTH
# ==========================================

@st.cache_resource
def load_prediction_resources():

    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)

    with open(MAX_LEN_PATH, "rb") as file:
        max_len = pickle.load(file)

    model = load_model(
        MODEL_PATH,
        compile=False
    )

    return model, tokenizer, max_len


try:

    model, tokenizer, max_len = load_prediction_resources()

    st.success("Model and tokenizer loaded successfully!")

except Exception as error:

    st.error(f"Error loading model or tokenizer: {error}")

    st.info(
        "Make sure these files exist in the same directory "
        "as app.py:\n"
        "- lstm_model.h5\n"
        "- tokenizer.pkl\n"
        "- max_len.pkl"
    )

    st.stop()


# ==========================================
# 4. PREDICT NEXT WORD
# ==========================================

def predict_next_word(text):

    # Convert text into token IDs
    sequence = tokenizer.texts_to_sequences([text])[0]

    if not sequence:
        return None

    # Keep the most recent tokens
    sequence = sequence[-(max_len - 1):]

    # Pad the sequence
    padded_sequence = pad_sequences(
        [sequence],
        maxlen=max_len - 1,
        padding="pre",
        truncating="pre"
    )

    # Predict the next word
    predictions = model.predict(
        padded_sequence,
        verbose=0
    )

    predicted_index = int(np.argmax(predictions[0]))

    # Convert token ID into a word
    predicted_word = tokenizer.index_word.get(
        predicted_index
    )

    return predicted_word


# ==========================================
# 5. USER INPUT
# ==========================================

st.subheader("✍️ Enter your text")

input_text = st.text_area(
    "Starting sentence",
    placeholder="Example: The sun is",
    height=120
)


# ==========================================
# 6. GENERATION SETTINGS
# ==========================================

max_words = st.slider(
    "Maximum number of words to generate",
    min_value=1,
    max_value=100,
    value=30
)

st.caption(
    "Generation stops when an end token is predicted "
    "or the word limit is reached."
)


# ==========================================
# 7. GENERATE SENTENCE
# ==========================================

if st.button(
    "🚀 Generate Sentence",
    use_container_width=True
):

    if not input_text.strip():

        st.warning("Please enter some text first.")

    else:

        generated_text = input_text.strip()

        stop_tokens = {
            "<END>",
            "<EOS>"
        }

        generated_words = []

        with st.spinner("Generating text..."):

            for _ in range(max_words):

                next_word = predict_next_word(
                    generated_text
                )

                # Stop if prediction is unavailable
                if next_word is None:
                    break

                # Stop at a trained end-of-sequence token
                if next_word in stop_tokens:
                    break

                # Append the predicted word
                generated_words.append(next_word)

                generated_text += " " + next_word

        # Display the result
        st.subheader("📝 Generated Sentence")

        st.success(generated_text)

        st.write(
            f"**Words generated:** {len(generated_words)}"
        )

        # Display generated words individually
        if generated_words:

            with st.expander("View prediction steps"):

                for index, word in enumerate(
                    generated_words,
                    start=1
                ):

                    st.write(
                        f"{index}. {word}"
                    )

        else:

            st.warning(
                "No words were generated. "
                "Check your input and tokenizer."
            )


# ==========================================
# 8. MODEL INFORMATION
# ==========================================

with st.expander("🔍 Model Information"):

    st.write("**Architecture:** LSTM")

    st.write(
        f"**Maximum sequence length:** {max_len}"
    )

    st.write(
        f"**Vocabulary size:** {len(tokenizer.word_index)}"
    )

    st.write(
        f"**Model input shape:** {model.input_shape}"
    )

    st.write(
        f"**Model output shape:** {model.output_shape}"
    )

    st.write(
        "**End tokens:** <END>, <EOS>"
    )

    st.caption(
        "End tokens only work if the model was trained "
        "to predict them."
    )
