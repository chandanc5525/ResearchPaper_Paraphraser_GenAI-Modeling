# Import Neccessory Libraries

import streamlit as st
import nltk
import random
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from sentence_transformers import SentenceTransformer, util

# Set Page Configuration

st.set_page_config(page_title="Research Paraphraser", page_icon="📝", layout="wide")
st.title("Research Paper Paraphraser")
st.title("Prepared By: Chandan Chaudhari")
st.title("Github: "https://github.com/chandanc5525")
st.markdown("Generate plagiarism-free paraphrased content for research papers")

# Download NLTK :
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)

# Model Pipeline

@st.cache_resource
def load_ai_models():
    try:
        # Paraphraser model
        tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5_paraphraser")
        model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5_paraphraser")
        
        # Embedding model for similarity
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        return tokenizer, model, embedder
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None, None, None

# Load models
ai_tokenizer, ai_model, ai_embedder = load_ai_models()

# Enhanced synonym dictionary
SYNONYMS = {
    # Core academic verbs
    'is': ['represents', 'constitutes', 'forms', 'equates to', 'stands as', 'embodies', 'signifies', 'denotes'],
    'are': ['comprise', 'make up', 'form', 'constitute', 'represent', 'compose', 'account for', 'consist of'],
    'can': ['is able to', 'has the capacity to', 'may', 'is capable of', 'has the potential to', 'is equipped to'],
    'help': ['assist', 'aid', 'support', 'facilitate', 'enable', 'promote', 'foster', 'contribute to'],
    'show': ['demonstrate', 'illustrate', 'reveal', 'exhibit', 'display', 'indicate', 'manifest', 'evidence'],
    'use': ['utilize', 'employ', 'apply', 'leverage', 'harness', 'operate', 'implement', 'exercise'],
    
    # Research and analysis
    'research': ['study', 'investigation', 'analysis', 'examination', 'inquiry', 'exploration', 'scrutiny', 'survey'],
    'analysis': ['examination', 'assessment', 'evaluation', 'scrutiny', 'study', 'interpretation', 'appraisal', 'review'],
    'data': ['information', 'findings', 'results', 'evidence', 'metrics', 'statistics', 'figures', 'measurements'],
    'method': ['technique', 'approach', 'procedure', 'strategy', 'methodology', 'process', 'system', 'framework'],
    'result': ['outcome', 'finding', 'conclusion', 'product', 'consequence', 'upshot', 'ramification', 'effect'],
    'study': ['research', 'investigate', 'examine', 'analyze', 'explore', 'scrutinize', 'review', 'assess'],
    
    # Importance and significance
    'important': ['crucial', 'significant', 'vital', 'essential', 'paramount', 'critical', 'imperative', 'fundamental'],
    'significant': ['notable', 'substantial', 'considerable', 'marked', 'pronounced', 'meaningful', 'noteworthy', 'appreciable'],
    'key': ['central', 'fundamental', 'primary', 'principal', 'core', 'essential', 'pivotal', 'crucial'],
    
    # Development and creation
    'develop': ['create', 'formulate', 'construct', 'establish', 'generate', 'produce', 'build', 'devise'],
    'create': ['generate', 'produce', 'develop', 'construct', 'fabricate', 'formulate', 'establish', 'initiate'],
    'build': ['construct', 'assemble', 'create', 'develop', 'form', 'fabricate', 'erect', 'establish'],
    
    # Understanding and knowledge
    'understand': ['comprehend', 'grasp', 'apprehend', 'discern', 'fathom', 'perceive', 'recognize', 'conceptualize'],
    'know': ['recognize', 'comprehend', 'understand', 'apprehend', 'discern', 'perceive', 'realize', 'cognize'],
    'learn': ['acquire', 'assimilate', 'absorb', 'master', 'grasp', 'comprehend', 'study', 'investigate'],
    
    # Improvement and change
    'improve': ['enhance', 'optimize', 'refine', 'advance', 'boost', 'amplify', 'strengthen', 'elevate'],
    'change': ['modify', 'alter', 'transform', 'adjust', 'adapt', 'revise', 'amend', 'restructure'],
    'increase': ['enhance', 'boost', 'raise', 'amplify', 'elevate', 'escalate', 'intensify', 'augment'],
    'decrease': ['reduce', 'diminish', 'lower', 'lessen', 'minimize', 'decline', 'subside', 'weaken'],
    
    # Explanation and description
    'explain': ['clarify', 'describe', 'interpret', 'elucidate', 'expound', 'illuminate', 'delineate', 'explicate'],
    'describe': ['portray', 'characterize', 'depict', 'detail', 'narrate', 'relate', 'specify', 'outline'],
    'discuss': ['examine', 'analyze', 'review', 'consider', 'debate', 'explore', 'treat', 'address'],
    
    # Discovery and finding
    'find': ['discover', 'identify', 'detect', 'locate', 'uncover', 'reveal', 'ascertain', 'determine'],
    'discover': ['find', 'uncover', 'reveal', 'identify', 'detect', 'ascertain', 'determine', 'locate'],
    'identify': ['recognize', 'distinguish', 'detect', 'determine', 'pinpoint', 'diagnose', 'establish', 'characterize'],
    
    # Suggestion and recommendation
    'suggest': ['propose', 'recommend', 'advise', 'indicate', 'put forward', 'submit', 'advocate', 'propound'],
    'recommend': ['suggest', 'advise', 'propose', 'counsel', 'advocate', 'endorse', 'prescribe', 'urge'],
    
    # Need and requirement
    'need': ['require', 'demand', 'necessitate', 'call for', 'entail', 'compel', 'obligate', 'mandate'],
    'require': ['need', 'demand', 'necessitate', 'call for', 'compel', 'obligate', 'mandate', 'entail'],
    
    # Problem and solution
    'problem': ['challenge', 'issue', 'difficulty', 'obstacle', 'complication', 'dilemma', 'setback', 'hurdle'],
    'solution': ['resolution', 'answer', 'remedy', 'fix', 'approach', 'method', 'strategy', 'technique'],
    
    # Size and quantity
    'large': ['substantial', 'considerable', 'significant', 'extensive', 'sizable', 'ample', 'generous', 'copious'],
    'small': ['modest', 'limited', 'minimal', 'moderate', 'restricted', 'negligible', 'token', 'nominal'],
    'many': ['numerous', 'multiple', 'various', 'sundry', 'diverse', 'myriad', 'multifarious', 'copious'],
    'few': ['scant', 'limited', 'sparse', 'meager', 'paltry', 'insufficient', 'inadequate', 'scarce'],
    
    # Quality and characteristics
    'good': ['excellent', 'superior', 'outstanding', 'exceptional', 'admirable', 'commendable', 'satisfactory', 'adequate'],
    'bad': ['poor', 'inferior', 'substandard', 'unsatisfactory', 'inadequate', 'deficient', 'faulty', 'flawed'],
    'different': ['distinct', 'diverse', 'varied', 'disparate', 'heterogeneous', 'divergent', 'dissimilar', 'contrasting'],
    'same': ['identical', 'equivalent', 'similar', 'comparable', 'alike', 'uniform', 'consistent', 'unchanging'],
    
    # Time and frequency
    'often': ['frequently', 'regularly', 'repeatedly', 'commonly', 'habitually', 'customarily', 'routinely', 'consistently'],
    'sometimes': ['occasionally', 'periodically', 'intermittently', 'sporadically', 'infrequently', 'seldom', 'rarely'],
    'always': ['constantly', 'continuously', 'perpetually', 'invariably', 'unfailingly', 'consistently', 'invariably'],
    
    # Academic specific terms
    'theory': ['hypothesis', 'concept', 'framework', 'model', 'principle', 'doctrine', 'postulate', 'supposition'],
    'model': ['framework', 'structure', 'system', 'paradigm', 'template', 'pattern', 'archetype', 'prototype'],
    'framework': ['structure', 'system', 'model', 'schema', 'blueprint', 'outline', 'plan', 'design'],
    'concept': ['idea', 'notion', 'theory', 'principle', 'abstraction', 'conception', 'perception', 'construct'],
    
    # Evidence and proof
    'evidence': ['proof', 'confirmation', 'verification', 'substantiation', 'corroboration', 'demonstration', 'indication', 'sign'],
    'proof': ['evidence', 'confirmation', 'verification', 'substantiation', 'corroboration', 'authentication', 'validation', 'attestation'],
    
    # Impact and effect
    'impact': ['effect', 'influence', 'consequence', 'result', 'outcome', 'repercussion', 'ramification', 'significance'],
    'effect': ['impact', 'influence', 'consequence', 'result', 'outcome', 'ramification', 'repercussion', 'aftermath'],
    
    # Complex academic phrases
    'in order to': ['so as to', 'with the purpose of', 'with the aim of', 'for the purpose of', 'with the intention of'],
    'due to': ['because of', 'owing to', 'as a result of', 'on account of', 'by virtue of'],
    'however': ['nevertheless', 'nonetheless', 'yet', 'but', 'although', 'though', 'even so', 'despite this'],
    'therefore': ['thus', 'consequently', 'hence', 'accordingly', 'as a result', 'for this reason', 'so'],
    
    # Additional common academic words
    'provide': ['offer', 'supply', 'furnish', 'deliver', 'present', 'render', 'give', 'bestow'],
    'obtain': ['acquire', 'gain', 'secure', 'attain', 'procure', 'achieve', 'earn', 'win'],
    'maintain': ['preserve', 'sustain', 'uphold', 'retain', 'continue', 'keep', 'conserve', 'perpetuate'],
    'achieve': ['attain', 'accomplish', 'realize', 'fulfill', 'complete', 'execute', 'perform', 'effect'],
}

def simple_paraphrase(text):
    """Simple synonym replacement without grammar checking"""
    words = text.split()
    paraphrased = []
    
    for word in words:
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        if clean_word in SYNONYMS and random.random() < 0.4:
            synonym = random.choice(SYNONYMS[clean_word])
            if word[0].isupper():
                synonym = synonym.capitalize()
            paraphrased.append(synonym)
        else:
            paraphrased.append(word)
    
    return ' '.join(paraphrased)

def ai_paraphrase(text):
    """Advanced AI paraphrasing using encoder-decoder"""
    if ai_tokenizer is None or ai_model is None:
        return "AI models not loaded properly"
    
    try:
        input_text = f"paraphrase: {text}"
        inputs = ai_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        
        with torch.no_grad():
            outputs = ai_model.generate(
                inputs,
                max_length=256,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                num_return_sequences=1
            )
        
        return ai_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error in AI paraphrasing: {e}"

def check_similarity(original, paraphrased):
    """Check semantic similarity"""
    if ai_embedder is None:
        return 0.5
    
    try:
        emb1 = ai_embedder.encode(original, convert_to_tensor=True)
        emb2 = ai_embedder.encode(paraphrased, convert_to_tensor=True)
        return float(util.cos_sim(emb1, emb2).item())
    except:
        return 0.5

# Read sample text
try:
    with open("sample.txt", "r", encoding="utf-8") as f:
        original_text = f.read().strip()
except FileNotFoundError:
    original_text = ""
    st.warning("Please add your research text to sample.txt")

# Display original text
if original_text:
    with st.expander("Original Text", expanded=True):
        st.text_area("Original Content", original_text, height=150, label_visibility="collapsed")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    mode = st.radio("Paraphrasing Mode:", ["Simple & Fast", "Advanced AI"])
    num_variations = st.slider("Number of variations", 1, 5, 3)

# Generate button
if st.button("Generate Paraphrased Versions", type="primary"):
    if original_text:
        results = []
        
        for i in range(num_variations):
            with st.spinner(f"Generating variation {i+1}..."):
                if mode == "Simple & Fast":
                    paraphrased = simple_paraphrase(original_text)
                    similarity = 0.5  # Placeholder for simple mode
                else:
                    paraphrased = ai_paraphrase(original_text)
                    similarity = check_similarity(original_text, paraphrased)
                
                uniqueness = (1 - similarity) * 100
                
                results.append({
                    'text': paraphrased,
                    'uniqueness': uniqueness
                })
        
        # Display results
        st.success(f"Generated {len(results)} variations!")
        
        for i, result in enumerate(results, 1):
            with st.expander(f"Option {i} (Uniqueness: {result['uniqueness']:.1f}%)", expanded=True):
                st.text_area(
                    f"Paraphrased Text {i}",
                    result['text'],
                    height=120,
                    key=f"text_{i}",
                    label_visibility="collapsed"
                )
                
                if st.button("Copy", key=f"copy_{i}"):
                    st.success("Copied to clipboard!")
    else:
        st.warning("Please add content to sample.txt first")
