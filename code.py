from transformers import pipeline

qa_pipeline = pipeline(
    task="question-answering",
    model="distilbert-base-cased-distilled-squad"
)

def get_answer(context, question):
    """
    This function takes context and question
    and returns a clean formatted answer.
    """

    
    input_data = {
        "context": context,
        "question": question
    }

    
    result = qa_pipeline(input_data)

   
    answer = result["answer"]

    
    confidence = round(result["score"], 2)

    return f"Answer: {answer} (Confidence: {confidence})"



if __name__ == "__main__":
    context_text = """
    Hugging Face is an AI company that develops tools for building 
    applications using machine learning. It is well known for its 
    Transformers library which provides pre-trained models for NLP tasks.
    """

    question_text = "What is Hugging Face known for?"

    output = get_answer(context_text, question_text)
    print(output)
