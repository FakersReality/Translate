import streamlit as st
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

st.set_page_config(
    page_title="Translator",
    page_icon=":robot:",
    layout="wide",
)

st.header("Language translator")

# UI language choices
languages = {
    "ar": "Arabic",
    "en": "English",
}

# Preferred higher-quality model for en<->ar
# (You can swap to "facebook/m2m100_418M" if you prefer smaller)
PREFERRED_QA_MODEL = "facebook/nllb-200-distilled-600M"

# NLLB language code map
NLLB_LANG = {
    "en": "eng_Latn",
    "ar": "arb_Arab",
}

source_lang, target_lang = st.columns(2)
with source_lang:
    src_lang_code = st.selectbox("Translate from", list(languages.keys()))
with target_lang:
    tgt_lang_code = st.selectbox("Translate to", list(languages.keys()))

# Legacy Helsinki models (fast, but often lower quality)
model_options = {
    "en-ar": "Helsinki-NLP/opus-mt-en-ar",
    "ar-en": "Helsinki-NLP/opus-mt-ar-en",
}
model_key = f"{src_lang_code}-{tgt_lang_code}"
helsinki_model = model_options.get(model_key, "")

# Let user pick model family (quality vs speed)
family = st.radio(
    "Model family",
    ["NLLB (better quality)", "Helsinki (faster)"],
    index=0,
    help="NLLB gives noticeably better English↔Arabic quality."
)

query = st.text_area(
    label=f"Your input text ({languages[src_lang_code]} → {languages[tgt_lang_code]})",
    placeholder="Enter text to translate",
    key="question_text"
)

@st.cache_resource
def load_nllb(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tok, mdl

@st.cache_resource
def load_helsinki(model_name: str):
    return pipeline("translation", model=model_name)

def translate_with_nllb(text: str, src: str, tgt: str) -> str:
    tok, mdl = load_nllb(PREFERRED_QA_MODEL)
    # Set correct language tags
    tok.src_lang = NLLB_LANG[src]
    forced_bos = tok.convert_tokens_to_ids(NLLB_LANG[tgt])

    inputs = tok(text, return_tensors="pt", truncation=True)
    outputs = mdl.generate(
        **inputs,
        forced_bos_token_id=forced_bos,
        num_beams=5,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        max_new_tokens=512,
    )
    return tok.batch_decode(outputs, skip_special_tokens=True)[0]

if st.button("Translate"):
    if not query or len(query.strip()) <= 1:
        st.warning("Please enter some text to translate.")
    elif src_lang_code == tgt_lang_code:
        st.info("Source and target languages are the same. Nothing to translate.")
    else:
        try:
            with st.spinner("In progress..."):
                if family.startswith("NLLB"):
                    output = translate_with_nllb(query, src_lang_code, tgt_lang_code)
                else:
                    if not helsinki_model:
                        st.error("Selected pair not supported by the Helsinki models.")
                        st.stop()
                    translator = load_helsinki(helsinki_model)
                    # Use beam search for better quality
                    result = translator(
                        query,
                        num_beams=5,
                        no_repeat_ngram_size=3,
                        max_length=512,
                    )
                    output = result[0]["translation_text"]

            height = min(2 * len(output), 240)
            st.text_area(
                label='Translation',
                value=output,
                height=height,
                key="translation_output",
            )
        except Exception as e:
            st.error(f"Sorry: Cannot translate! {e}")
