from transformers import pipeline

class QuestionAnsweringApp:
    """
    A simple Question Answering application
    using a HuggingFace pre-trained model.
    """

    def __init__(self):
       
        self.model = pipeline(
            "question-answering",
            model="distilbert-base-cased-distilled-squad"
        )

    def answer_question(self, context, question):
        """
        Takes context and question as input
        and returns formatted output.
        """

        if not context.strip() or not question.strip():
            return "Context and Question cannot be empty."

        # Pass input to the model
        response = self.model({
            "context": context,
            "question": question
        })

        # Format output
        answer = response["answer"]
        score = round(response["score"] * 100, 2)

        return f"\nAnswer: {answer}\nConfidence: {score}%"


def main():
    print("🔹 Question Answering App (HuggingFace)\n")

    context = input("Enter Context:\n")
    question = input("\nEnter Question:\n")

    qa_app = QuestionAnsweringApp()
    result = qa_app.answer_question(context, question)

    print("\nResult:")
    print(result)

