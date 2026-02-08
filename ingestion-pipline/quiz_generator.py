import os
import re
import json
from typing import List, Dict, Optional
from datetime import datetime

from dotenv import load_dotenv
load_dotenv(override=True)

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ============================================================
# CONFIGURATION
# ============================================================
PERSIST_DIR = "quiz_chroma_store"
COLLECTION_NAME = "student_notes"
NOTES_FOLDER = "data"  # Folder containing your notes

# Quiz settings
DEFAULT_NUM_QUESTIONS = 5
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]


# ============================================================
# 1) NOTES INGESTION - Load and store notes in vector database
# ============================================================
class NotesIngestion:
    """Handles loading and storing notes in ChromaDB."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "]
        )
        
    def load_notes_from_file(self, file_path: str) -> str:
        """Load text from a single file."""
        print(f"Loading: {os.path.basename(file_path)}")
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    
    def load_all_notes(self, folder: str) -> List[Document]:
        """Load all notes from a folder."""
        print(f"\nLoading notes from: {folder}")
        print("=" * 50)
        
        documents = []
        
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            
            # Skip directories
            if os.path.isdir(file_path):
                continue
                
            # Only process text and markdown files
            ext = os.path.splitext(filename)[1].lower()
            if ext not in [".txt", ".md"]:
                continue
            
            # Load content
            content = self.load_notes_from_file(file_path)
            
            # Split into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Create documents with metadata
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "chunk_index": i,
                        "topic": self._extract_topic(chunk)
                    }
                )
                documents.append(doc)
            
            print(f"   Created {len(chunks)} chunks from {filename}")
        
        print(f"\nTotal documents created: {len(documents)}")
        return documents
    
    def _extract_topic(self, text: str) -> str:
        """Extract topic from chunk (looks for headers)."""
        # Look for markdown headers
        header_match = re.search(r'^#+\s*(.+)$', text, re.MULTILINE)
        if header_match:
            return header_match.group(1).strip()
        
        # Return first line as topic
        first_line = text.split('\n')[0][:50]
        return first_line.strip()
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Create/update ChromaDB with documents."""
        print("\nCreating vector store...")
        
        # Remove existing store if it exists
        if os.path.exists(PERSIST_DIR):
            import shutil
            shutil.rmtree(PERSIST_DIR)
        
        # Create new vector store
        db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR
        )
        
        print(f"Vector store created with {len(documents)} documents")
        return db


# ============================================================
# 2) QUIZ GENERATOR - Generate questions from notes using RAG
# ============================================================
class QuizGenerator:
    """Generates quizzes from notes using RAG."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.db = None
        self._load_vector_store()
    
    def _load_vector_store(self):
        """Load existing vector store."""
        if os.path.exists(PERSIST_DIR):
            self.db = Chroma(
                collection_name=COLLECTION_NAME,
                persist_directory=PERSIST_DIR,
                embedding_function=self.embeddings
            )
            print("Vector store loaded successfully")
        else:
            print("No vector store found. Please ingest notes first!")
    
    def retrieve_relevant_content(self, topic: str, k: int = 5) -> List[str]:
        """Retrieve relevant content for a topic."""
        if not self.db:
            return []
        
        retriever = self.db.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(topic)
        
        return [doc.page_content for doc in docs]
    
    def generate_mcq_questions(
        self, 
        topic: str, 
        num_questions: int = 3,
        difficulty: str = "medium"
    ) -> List[Dict]:
        """Generate Multiple Choice Questions."""
        
        # Retrieve relevant content
        context = self.retrieve_relevant_content(topic)
        if not context:
            print("No relevant content found for this topic")
            return []
        
        context_text = "\n\n".join(context)
        
        prompt = f"""You are a quiz generator for students. Based on the following study notes, 
create {num_questions} multiple choice questions.

STUDY NOTES:
{context_text}

INSTRUCTIONS:
- Create {difficulty} difficulty questions
- Each question should have 4 options (A, B, C, D)
- Only one option should be correct
- Questions should test understanding, not just memorization
- Make wrong options plausible but clearly incorrect

Return the questions in this EXACT JSON format:
{{
    "questions": [
        {{
            "question": "Your question here?",
            "options": {{
                "A": "First option",
                "B": "Second option", 
                "C": "Third option",
                "D": "Fourth option"
            }},
            "correct_answer": "A",
            "explanation": "Brief explanation why this is correct"
        }}
    ]
}}

Generate exactly {num_questions} questions. Return ONLY valid JSON."""

        response = self.llm.invoke(prompt)
        
        try:
            # Parse JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                result = json.loads(json_match.group())
                questions = result.get("questions", [])
                for q in questions:
                    q["type"] = "mcq"
                return questions
        except json.JSONDecodeError:
            print("Error parsing questions. Trying again...")
            
        return []
    
    def generate_true_false_questions(
        self, 
        topic: str, 
        num_questions: int = 3
    ) -> List[Dict]:
        """Generate True/False Questions."""
        
        context = self.retrieve_relevant_content(topic)
        if not context:
            return []
        
        context_text = "\n\n".join(context)
        
        prompt = f"""You are a quiz generator for students. Based on the following study notes,
create {num_questions} True/False questions.

STUDY NOTES:
{context_text}

INSTRUCTIONS:
- Create clear statements that are either true or false
- Mix true and false answers (not all same)
- Statements should be based ONLY on the provided notes
- Include some tricky but fair questions

Return in this EXACT JSON format:
{{
    "questions": [
        {{
            "statement": "The statement to evaluate",
            "correct_answer": true,
            "explanation": "Why this is true/false"
        }}
    ]
}}

Generate exactly {num_questions} questions. Return ONLY valid JSON."""

        response = self.llm.invoke(prompt)
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                result = json.loads(json_match.group())
                questions = result.get("questions", [])
                for q in questions:
                    q["type"] = "true_false"
                return questions
        except json.JSONDecodeError:
            pass
            
        return []
    
    def generate_fill_blank_questions(
        self, 
        topic: str, 
        num_questions: int = 3
    ) -> List[Dict]:
        """Generate Fill-in-the-Blank Questions."""
        
        context = self.retrieve_relevant_content(topic)
        if not context:
            return []
        
        context_text = "\n\n".join(context)
        
        prompt = f"""You are a quiz generator for students. Based on the following study notes,
create {num_questions} fill-in-the-blank questions.

STUDY NOTES:
{context_text}

INSTRUCTIONS:
- Replace key terms/concepts with blanks (shown as _____)
- The blank should test important concepts
- Provide hints if the answer might be ambiguous
- Accept reasonable variations of the answer

Return in this EXACT JSON format:
{{
    "questions": [
        {{
            "question": "The _____ period for new employees is 6 months.",
            "correct_answer": "probation",
            "acceptable_answers": ["probation", "probationary"],
            "hint": "It's a trial period for new hires"
        }}
    ]
}}

Generate exactly {num_questions} questions. Return ONLY valid JSON."""

        response = self.llm.invoke(prompt)
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response.content)
            if json_match:
                result = json.loads(json_match.group())
                questions = result.get("questions", [])
                for q in questions:
                    q["type"] = "fill_blank"
                return questions
        except json.JSONDecodeError:
            pass
            
        return []
    
    def generate_mixed_quiz(
        self, 
        topic: str, 
        num_questions: int = 5,
        difficulty: str = "medium"
    ) -> List[Dict]:
        """Generate a mixed quiz with different question types."""
        
        # Distribute questions across types
        num_mcq = max(1, num_questions // 2)
        num_tf = max(1, (num_questions - num_mcq) // 2)
        num_fill = num_questions - num_mcq - num_tf
        
        all_questions = []
        
        print(f"Generating quiz on: {topic}")
        print(f"   {num_mcq} MCQ, {num_tf} True/False, {num_fill} Fill-blank")
        
        # Generate each type
        mcq = self.generate_mcq_questions(topic, num_mcq, difficulty)
        all_questions.extend(mcq)
        
        tf = self.generate_true_false_questions(topic, num_tf)
        all_questions.extend(tf)
        
        fill = self.generate_fill_blank_questions(topic, num_fill)
        all_questions.extend(fill)
        
        # Shuffle questions
        import random
        random.shuffle(all_questions)
        
        return all_questions


# ============================================================
# 3) QUIZ RUNNER - Interactive quiz for students
# ============================================================
class QuizRunner:
    """Runs interactive quizzes and tracks scores."""
    
    def __init__(self):
        self.current_score = 0
        self.total_questions = 0
        self.quiz_history = []
    
    def display_question(self, question: Dict, question_num: int) -> None:
        """Display a single question."""
        q_type = question.get("type", "unknown")
        
        print(f"\n{'='*50}")
        print(f"Question {question_num}")
        print(f"{'='*50}")
        
        if q_type == "mcq":
            print(f"\n{question['question']}\n")
            for opt, text in question['options'].items():
                print(f"   {opt}) {text}")
                
        elif q_type == "true_false":
            print(f"\n{question['statement']}")
            print("\n   Enter: True (T) or False (F)")
            
        elif q_type == "fill_blank":
            print(f"\n{question['question']}")
            if question.get('hint'):
                print(f"\n   Hint: {question['hint']}")
    
    def get_answer(self, question: Dict) -> str:
        """Get and validate user answer."""
        q_type = question.get("type")
        
        while True:
            answer = input("\nYour answer: ").strip()
            
            if q_type == "mcq":
                if answer.upper() in ['A', 'B', 'C', 'D']:
                    return answer.upper()
                print("   Please enter A, B, C, or D")
                
            elif q_type == "true_false":
                if answer.lower() in ['true', 't', 'false', 'f']:
                    return answer.lower() in ['true', 't']
                print("   Please enter True/T or False/F")
                
            elif q_type == "fill_blank":
                if answer:
                    return answer.lower()
                print("   Please enter an answer")
            
            else:
                return answer
    
    def check_answer(self, question: Dict, user_answer) -> bool:
        """Check if the answer is correct."""
        q_type = question.get("type")
        
        if q_type == "mcq":
            return user_answer == question['correct_answer']
            
        elif q_type == "true_false":
            return user_answer == question['correct_answer']
            
        elif q_type == "fill_blank":
            correct = question.get('correct_answer', '').lower()
            acceptable = [a.lower() for a in question.get('acceptable_answers', [correct])]
            return user_answer in acceptable
        
        return False
    
    def show_result(self, question: Dict, is_correct: bool) -> None:
        """Show result and explanation."""
        if is_correct:
            print("\n   CORRECT! Great job!")
            self.current_score += 1
        else:
            print("\n   INCORRECT")
            
            # Show correct answer
            q_type = question.get("type")
            if q_type == "mcq":
                correct = question['correct_answer']
                print(f"   Correct answer: {correct}) {question['options'][correct]}")
            elif q_type == "true_false":
                print(f"   Correct answer: {question['correct_answer']}")
            elif q_type == "fill_blank":
                print(f"   Correct answer: {question['correct_answer']}")
        
        # Show explanation if available
        explanation = question.get('explanation')
        if explanation:
            print(f"\n   Explanation: {explanation}")
    
    def run_quiz(self, questions: List[Dict], quiz_name: str = "Quiz") -> Dict:
        """Run a complete quiz session."""
        
        if not questions:
            print("\nNo questions available for this quiz!")
            return {}
        
        self.current_score = 0
        self.total_questions = len(questions)
        
        print(f"\n{'='*50}")
        print(f"   STARTING: {quiz_name}")
        print(f"   Total Questions: {self.total_questions}")
        print(f"{'='*50}")
        
        input("\nPress ENTER to begin...")
        
        # Run through each question
        for i, question in enumerate(questions, 1):
            self.display_question(question, i)
            user_answer = self.get_answer(question)
            is_correct = self.check_answer(question, user_answer)
            self.show_result(question, is_correct)
            
            # Pause between questions
            if i < self.total_questions:
                input("\nPress ENTER for next question...")
        
        # Show final results
        return self.show_final_results(quiz_name)
    
    def show_final_results(self, quiz_name: str) -> Dict:
        """Display final quiz results."""
        
        percentage = (self.current_score / self.total_questions) * 100
        
        print(f"\n{'='*50}")
        print(f"QUIZ COMPLETE: {quiz_name}")
        print(f"{'='*50}")
        print(f"\n   Your Score: {self.current_score}/{self.total_questions}")
        print(f"   Percentage: {percentage:.1f}%")
        
        # Performance message
        if percentage >= 90:
            print("\n   EXCELLENT! You're a star student!")
        elif percentage >= 70:
            print("\n   GOOD JOB! Keep up the great work!")
        elif percentage >= 50:
            print("\n   FAIR! Review the material and try again.")
        else:
            print("\n   KEEP TRYING! Practice makes perfect!")
        
        # Save to history
        result = {
            "quiz_name": quiz_name,
            "score": self.current_score,
            "total": self.total_questions,
            "percentage": percentage,
            "timestamp": datetime.now().isoformat()
        }
        self.quiz_history.append(result)
        
        return result


# ============================================================
# 4) MAIN APPLICATION - CLI Interface
# ============================================================
class QuizApp:
    """Main application combining all components."""
    
    def __init__(self):
        self.ingestion = NotesIngestion()
        self.generator = None
        self.runner = QuizRunner()
    
    def show_banner(self):
        """Display app banner."""
        print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║              NOTES TO QUIZ GENERATOR                         ║
║       ─────────────────────────────                          ║
║       Turn your study notes into interactive quizzes!        ║
║                                                              ║
║              Perfect for beginner students                   ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """)
    
    def show_menu(self):
        """Display main menu."""
        print("""
┌─────────────────────────────────────┐
│           MAIN MENU                 │
├─────────────────────────────────────┤
│  1. Load/Reload Notes               │
│  2. Generate MCQ Quiz               │
│  3. Generate True/False Quiz        │
│  4. Generate Fill-in-Blank Quiz     │
│  5. Generate Mixed Quiz             │
│  6. View Quiz History               │
│  7. Help                            │
│  8. Exit                            │
└─────────────────────────────────────┘
        """)
    
    def load_notes(self):
        """Load notes from folder."""
        print("\n" + "="*50)
        print("LOADING NOTES")
        print("="*50)
        
        # Check if data folder exists
        if not os.path.exists(NOTES_FOLDER):
            print(f"\nNotes folder '{NOTES_FOLDER}' not found!")
            print(f"   Please create it and add your .txt or .md files.")
            return
        
        # Load and ingest notes
        documents = self.ingestion.load_all_notes(NOTES_FOLDER)
        
        if documents:
            self.ingestion.create_vector_store(documents)
            self.generator = QuizGenerator()
            print("\nNotes loaded successfully! Ready to generate quizzes.")
        else:
            print("\nNo valid notes found in the folder.")
            print("   Add .txt or .md files to the 'data' folder.")
    
    def get_topic_input(self) -> str:
        """Get topic from user."""
        print("\nEnter a topic for the quiz")
        print("   (or press ENTER for 'general' topic)")
        topic = input("\n   Topic: ").strip()
        return topic if topic else "general overview"
    
    def get_num_questions(self) -> int:
        """Get number of questions from user."""
        print(f"\nHow many questions? (default: {DEFAULT_NUM_QUESTIONS})")
        try:
            num = input("   Number: ").strip()
            return int(num) if num else DEFAULT_NUM_QUESTIONS
        except ValueError:
            return DEFAULT_NUM_QUESTIONS
    
    def get_difficulty(self) -> str:
        """Get difficulty level from user."""
        print("\nSelect difficulty:")
        print("   1. Easy")
        print("   2. Medium")
        print("   3. Hard")
        
        choice = input("\n   Choice (1-3): ").strip()
        
        if choice == "1":
            return "easy"
        elif choice == "3":
            return "hard"
        else:
            return "medium"
    
    def generate_and_run_quiz(self, quiz_type: str):
        """Generate and run a quiz."""
        
        if not self.generator:
            print("\nPlease load notes first (Option 1)!")
            return
        
        topic = self.get_topic_input()
        num_questions = self.get_num_questions()
        
        print(f"\nGenerating {quiz_type} quiz...")
        
        questions = []
        
        if quiz_type == "mcq":
            difficulty = self.get_difficulty()
            questions = self.generator.generate_mcq_questions(
                topic, num_questions, difficulty
            )
            quiz_name = f"MCQ Quiz - {topic}"
            
        elif quiz_type == "true_false":
            questions = self.generator.generate_true_false_questions(
                topic, num_questions
            )
            quiz_name = f"True/False Quiz - {topic}"
            
        elif quiz_type == "fill_blank":
            questions = self.generator.generate_fill_blank_questions(
                topic, num_questions
            )
            quiz_name = f"Fill-in-Blank Quiz - {topic}"
            
        elif quiz_type == "mixed":
            difficulty = self.get_difficulty()
            questions = self.generator.generate_mixed_quiz(
                topic, num_questions, difficulty
            )
            quiz_name = f"Mixed Quiz - {topic}"
        
        if questions:
            print(f"\nGenerated {len(questions)} questions!")
            self.runner.run_quiz(questions, quiz_name)
        else:
            print("\nCould not generate questions. Try a different topic.")
    
    def view_history(self):
        """View quiz history."""
        print("\n" + "="*50)
        print("QUIZ HISTORY")
        print("="*50)
        
        if not self.runner.quiz_history:
            print("\n   No quizzes taken yet!")
            return
        
        for i, result in enumerate(self.runner.quiz_history, 1):
            print(f"\n   {i}. {result['quiz_name']}")
            print(f"      Score: {result['score']}/{result['total']} ({result['percentage']:.1f}%)")
            print(f"      Date: {result['timestamp'][:10]}")
    
    def show_help(self):
        """Show help information."""
        print("""
╔══════════════════════════════════════════════════════════════╗
║                        HELP GUIDE                            ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  HOW TO USE THIS APP:                                        ║
║  ────────────────────                                        ║
║                                                              ║
║  1. ADD YOUR NOTES:                                          ║
║     • Put your .txt or .md files in the 'data' folder        ║
║     • Notes can be on any subject                            ║
║                                                              ║
║  2. LOAD NOTES:                                              ║
║     • Select option 1 to load your notes                     ║
║     • This creates a searchable database                     ║
║                                                              ║
║  3. GENERATE QUIZZES:                                        ║
║     • Choose a quiz type (MCQ, True/False, etc.)             ║
║     • Enter a topic from your notes                          ║
║     • Set number of questions and difficulty                 ║
║                                                              ║
║  4. TAKE THE QUIZ:                                           ║
║     • Answer each question                                   ║
║     • Get instant feedback and explanations                  ║
║     • See your final score!                                  ║
║                                                              ║
║  QUIZ TYPES:                                                 ║
║  ───────────                                                 ║
║  • MCQ: Multiple choice with 4 options                       ║
║  • True/False: Evaluate statements                           ║
║  • Fill-in-Blank: Complete the sentence                      ║
║  • Mixed: Combination of all types                           ║
║                                                              ║
║  TIPS:                                                       ║
║  ─────                                                       ║
║  • Use specific topics for better questions                  ║
║  • Start with 'easy' difficulty and work up                  ║
║  • Review explanations to learn from mistakes                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
        """)
    
    def run(self):
        """Main application loop."""
        self.show_banner()
        
        # Check if vector store exists
        if os.path.exists(PERSIST_DIR):
            print("Found existing notes database. Loading...")
            self.generator = QuizGenerator()
        else:
            print("Tip: Load your notes first using option 1!")
        
        while True:
            self.show_menu()
            
            choice = input("Enter your choice (1-8): ").strip()
            
            if choice == "1":
                self.load_notes()
            elif choice == "2":
                self.generate_and_run_quiz("mcq")
            elif choice == "3":
                self.generate_and_run_quiz("true_false")
            elif choice == "4":
                self.generate_and_run_quiz("fill_blank")
            elif choice == "5":
                self.generate_and_run_quiz("mixed")
            elif choice == "6":
                self.view_history()
            elif choice == "7":
                self.show_help()
            elif choice == "8":
                print("\nThanks for using Notes to Quiz Generator!")
                print("   Happy studying!\n")
                break
            else:
                print("\nInvalid choice. Please enter 1-8.")
            
            input("\nPress ENTER to continue...")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    app = QuizApp()
    app.run()
