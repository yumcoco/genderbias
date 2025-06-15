# GENIA @ P4G

**Hack4Her Challenge: Detecting Gender Bias in Job Vacancies**

A comprehensive web application for analyzing and improving gender bias in job descriptions, developed for the Randstad Digital Inclusive Hiring Challenge.

##Project Overview

This project addresses the critical challenge of gender bias in hiring by creating an AI-powered tool that detects, analyzes, and reduces biased language in job postings. Built for the Hack4Her hackathon's "Inclusive Hiring: Detecting Gender Bias in Job Vacancies" challenge, our solution helps organizations write more inclusive job descriptions to attract diverse talent, particularly women in tech and finance.

## Problem We're Solving

Research shows that subtle differences in job description wording—tone, requirements, and phrasing—significantly affect whether women apply for positions. Our tool tackles this by:

- **Identifying** masculine-coded language that deters female applicants
- **Analyzing** the linguistic patterns that create bias
- **Providing** evidence-based recommendations for improvement
- **Predicting** the impact of language choices on application rates

## Core Features

### 1. **Bias Detection Engine**
- Detects masculine-coded words
- Identifies inclusive language patterns
- Flags exclusive requirements that create barriers
- Real-time analysis with color-coded highlighting

### 2. **Predictive Modeling**
- Calculates inclusivity scores (0-100 scale)
- Predicts women application rates based on language patterns
- Provides bias direction analysis (Masculine/Feminine/Neutral/Inclusive)
- Replacement bonus scoring for successful improvements

### 3. **Intelligent Rewriting System**
- Evidence-based word replacement suggestions
- Context-aware alternative recommendations
- Automatic requirement softening (e.g., "required" → "preferred")
- Before/after comparison with highlighted changes

## echnical Implementation

### **NLP & Natural Language Processing Techniques**

- **Text Preprocessing**: Tokenization, normalization, and cleaning
- **Pattern Recognition**: Regex-based word boundary matching for accurate detection
- **Linguistic Analysis**: Word categorization based on gender-coding research
- **Sentiment Analysis**: Tone assessment for aggressive vs. collaborative language
- **Feature Extraction**: N-gram analysis and contextual word embeddings
- **Text Generation**: Rule-based rewriting with linguistic constraints

### **Core Technologies**

- **Frontend**: Streamlit for interactive web interface
- **Backend**: Python with regex-based NLP processing
- **Data Processing**: JSON configuration for flexible rule management
- **Visualization**: Real-time highlighting and comparison tools
- **Algorithm**: Custom scoring system based on linguistic research

##Key Highlights

### **Evidence-Based Approach**
- Built on peer-reviewed research (Gaucher et al. 2011, Born & Taris 2010)
- Configurable rules based on academic findings
- Transparent scoring methodology with research citations

### **Real-Time Visual Feedback**
- Color-coded highlighting for different bias types
- Interactive before/after comparisons
- Strikethrough for replaced words, highlighting for improvements
- Intuitive metrics dashboard

### **Practical Usability**
- Demo examples for immediate testing
- Customizable word lists and replacement rules
- Actionable recommendations with specific suggestions
- Export-ready improved job descriptions

### **Scalable Architecture**
- Modular design for easy extension
- JSON-based configuration for rule updates
- Portable codebase for integration into existing HR systems

##Social Impact

### **Promoting Gender Equality in Tech & Finance**
GENIA directly addresses women's underrepresentation in technology and finance sectors by:

- **Removing Barriers**: Eliminating language that unconsciously discourages women from applying
- **Increasing Representation**: Potentially increasing women application rates
- **Raising Awareness**: Educating recruiters about the impact of language choices
- **Democratizing Inclusion**: Making bias detection accessible to organizations of all sizes

### **Ethical Considerations**
- **Transparency**: Clear explanations of scoring methodology and bias detection
- **Bias Mitigation**: Awareness of potential model limitations and false positives
- **Inclusive Design**: Focus on adding inclusive language rather than just removing masculine terms
- **User Education**: Providing context and research backing for all recommendations


## Validation & Results

Our tool has been tested on both synthetic and real-world job descriptions, showing:
- Accurate detection of biased language patterns
- Meaningful improvement in inclusivity scores after rewriting
- Preservation of job requirements while improving accessibility
- User-friendly interface suitable for HR professionals

## 🛠️ Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd gender-bias-analysis

# Install dependencies
pip install streamlit

# Run the application
streamlit run app.py
```

Navigate to the web interface and start analyzing job descriptions immediately with our demo examples or your own content.



**Built for Hack4Her 2025 @ P4G - Making hiring more inclusive, one job description at a time.**