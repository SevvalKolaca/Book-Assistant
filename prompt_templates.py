class PromptTemplate:
    def __init__(self, book_name, question):
        self.input_book_name = book_name
        self.input_question = question
        self.template = f'''You are a book expert. Answer the question according to the book named below.

            Book: {{book_name}}
            Question: {{question}}?
            '''

    def get_prompt(self):
        return self.template.format(book_name=self.input_book_name, question=self.input_question)

    def set_book_name(self, book_name):
        self.input_book_name = book_name
    
    def set_question(self, question):
        self.input_question = question

