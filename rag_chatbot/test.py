from rag_chatbot.eval import  QAGenerator

generator = QAGenerator()

generator.generate(
    input_files=[
    "How does retrieval augmented generation work?",
    "What are the benefits of using RAG?",
    "How can I improve the accuracy of my chatbot?",
    "What embedding model is best for technical documentation?",
    "How should I chunk my documents for optimal retrieval?"
]
)