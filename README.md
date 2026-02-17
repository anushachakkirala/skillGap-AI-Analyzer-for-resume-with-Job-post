An intelligent, full-stack web application that analyzes and matches resumes against job descriptions using advanced NLP techniques — featuring OTP-based authentication, 
Sentence-BERT semantic embeddings, with interactive visualizations, and professional PDF report generation.

@Features:
1.Authentication
OTP-based two-factor login via email
User registration with profile management
Profile picture upload and editing
Secure session management

2.Document Parsing
Supports PDF, DOCX, and TXT file formats
Automatic text cleaning and normalization
Word count and page metrics

3.Skill Extraction (3 Methods)
Pattern Matching — direct keyword detection with frequency scoring
Noun Chunk Extraction — using spaCy for phrase-level detection
Named Entity Recognition (NER) — using spaCy for context-aware extraction
Configurable confidence threshold
Optional word stemming via NLTK

4.Sentence-BERT Embeddings
Semantic similarity using sentence-transformers
Configurable model selection
Normalized embedding support
Real-time similarity matrix generation

5.Gap Analysis
Weighted importance scoring based on JD frequency
Exact and semantic skill matching
Priority-level classification (High / Medium / Low)
Best-match resume skill identification per JD requirement
Top 3 critical missing skills highlighted

6. Visualizations 
Interactive charts using Plotly
Organized in 5 visualization sub-tabs

7.Report Generation
CSV: Full skill-by-skill analysis
TXT: Plain text summary
PDF: Multi-page professional report via ReportLab
