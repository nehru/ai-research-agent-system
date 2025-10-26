"""
Enhanced Multi-Agent Research System - FINAL WORKING VERSION
Uses CrewAI's LLM class instead of langchain-ollama
"""

from crewai import Agent, Task, Crew, Process, LLM
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
import os
from datetime import datetime

os.environ["OPENAI_API_KEY"] = "not-needed"

class EnhancedMultiAgentSystem:
    def __init__(self, use_memory=True, concurrent=False):
        """
        Initialize enhanced multi-agent system
        
        Args:
            use_memory (bool): Enable ChromaDB memory between runs
            concurrent (bool): Use concurrent processing instead of sequential
        """
        self.use_memory = use_memory
        self.concurrent = concurrent
        
        # Initialize models using CrewAI's LLM class (CRITICAL FIX!)
        self.llama_model = LLM(
            model="ollama/llama3",
            base_url="http://localhost:11434"
        )
        
        self.deepseek_model = LLM(
            model="ollama/deepseek-r1:7b",
            base_url="http://localhost:11434"
        )
        
        self.qwen_model = LLM(
            model="ollama/qwen2.5-coder",
            base_url="http://localhost:11434"
        )
        
        # Initialize memory if enabled
        self.memory_store = None
        if use_memory:
            self.memory_store = self._initialize_memory()
        
        # Create agents
        self.research_agent = self._create_research_agent()
        self.analysis_agent = self._create_analysis_agent()
        self.writer_agent = self._create_writer_agent()
        self.manager_agent = self._create_manager_agent()
    
    def _initialize_memory(self):
        """Initialize ChromaDB for persistent memory"""
        try:
            embeddings = OllamaEmbeddings(
                model="llama3",
                base_url="http://localhost:11434"
            )
            
            persist_directory = "src/memory_db"
            os.makedirs(persist_directory, exist_ok=True)
            
            vectorstore = Chroma(
                collection_name="research_memory",
                embedding_function=embeddings,
                persist_directory=persist_directory
            )
            
            print("Memory store initialized successfully")
            return vectorstore
        except Exception as e:
            print(f"Warning: Could not initialize memory store: {e}")
            return None
    
    def _create_research_agent(self):
        """Create research agent with Llama 3"""
        return Agent(
            role='Research Specialist',
            goal='Extract and organize relevant information from documents',
            backstory="""You are an expert researcher with a keen eye for detail.
            You excel at finding key information, facts, and data points from documents.
            Your strength lies in thorough analysis and systematic information extraction.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llama_model
        )
    
    def _create_analysis_agent(self):
        """Create analysis agent with DeepSeek-R1"""
        return Agent(
            role='Data Analyst',
            goal='Analyze extracted information and generate insights',
            backstory="""You are a skilled data analyst with expertise in pattern recognition
            and insight generation. You can identify trends, correlations, and draw meaningful
            conclusions from research data.""",
            verbose=True,
            allow_delegation=False,
            llm=self.deepseek_model
        )
    
    def _create_writer_agent(self):
        """Create writer agent with Qwen 2.5"""
        return Agent(
            role='Technical Writer',
            goal='Create comprehensive and well-structured reports',
            backstory="""You are a professional technical writer who excels at synthesizing
            complex information into clear, readable reports. Your reports are well-organized,
            concise, and actionable.""",
            verbose=True,
            allow_delegation=False,
            llm=self.qwen_model
        )
    
    def _create_manager_agent(self):
        """Create manager agent with Llama 3"""
        return Agent(
            role='Project Manager',
            goal='Coordinate the research workflow and ensure quality',
            backstory="""You are an experienced project manager who coordinates team efforts
            and ensures high-quality deliverables. You delegate tasks effectively and maintain
            quality standards throughout the process.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llama_model
        )
    
    def _save_to_memory(self, content, metadata):
        """Save content to memory store"""
        if self.memory_store:
            try:
                self.memory_store.add_texts(
                    texts=[content],
                    metadatas=[metadata]
                )
                print("Saved to memory store")
            except Exception as e:
                print(f"Warning: Could not save to memory: {e}")
    
    def _search_memory(self, query, k=3):
        """Search memory store for relevant past information"""
        if self.memory_store:
            try:
                results = self.memory_store.similarity_search(query, k=k)
                return [doc.page_content for doc in results]
            except Exception as e:
                print(f"Warning: Could not search memory: {e}")
        return []
    
    def _save_report(self, content):
        """Save report with timestamp"""
        try:
            os.makedirs("src/output", exist_ok=True)
            
            # Save latest report
            with open("src/output/latest_report.txt", "w", encoding="utf-8") as f:
                f.write(content)
            
            # Save timestamped report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"src/output/report_{timestamp}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            
            print(f"\nReport saved to:")
            print(f"  - src/output/latest_report.txt")
            print(f"  - {filename}")
            
            return filename
        except Exception as e:
            print(f"Error saving report: {e}")
            return None
    
    def analyze(self, document_content):
        """
        Analyze document using multi-agent system
        
        Args:
            document_content (str): Document text to analyze
            
        Returns:
            str: Generated report
        """
        # Search memory for relevant past information
        memory_context = ""
        if self.use_memory:
            past_findings = self._search_memory(document_content[:500])
            if past_findings:
                memory_context = "\n\nRelevant past findings:\n" + "\n".join(past_findings[:2])
        
        # Create tasks
        research_task = Task(
            description=f"""Extract and organize key information from the following document.
            Focus on main topics, key facts, data points, and important concepts.
            
            Document:
            {document_content}
            {memory_context}
            
            Provide a structured summary of your findings.""",
            expected_output="Structured findings with clear categories and key points",
            agent=self.research_agent
        )
        
        analysis_task = Task(
            description="""Analyze the research findings and generate insights.
            Identify patterns, trends, and relationships in the data.
            Draw meaningful conclusions and implications.""",
            expected_output="Detailed analysis with patterns, insights, and implications",
            agent=self.analysis_agent,
            context=[research_task]
        )
        
        writing_task = Task(
            description="""Create a comprehensive research report based on the findings and analysis.
            
            Structure the report with:
            1. Executive Summary
            2. Key Findings
            3. Detailed Analysis
            4. Conclusions
            5. Recommendations (if applicable)
            
            Ensure the report is clear, professional, and actionable.""",
            expected_output="Complete professional research report",
            agent=self.writer_agent,
            context=[research_task, analysis_task]
        )
        
        # Create crew with selected process type
        process_type = Process.concurrent if self.concurrent else Process.sequential
        process_name = "Concurrent" if self.concurrent else "Sequential"
        
        print(f"\nStarting {process_name} Processing...")
        print("=" * 80)
        
        crew = Crew(
            agents=[self.research_agent, self.analysis_agent, self.writer_agent, self.manager_agent],
            tasks=[research_task, analysis_task, writing_task],
            process=process_type,
            verbose=True
        )
        
        # Execute the crew
        result = crew.kickoff()
        
        print("\n" + "=" * 80)
        print(f"SUCCESS: {process_name} Analysis Complete!")
        print("=" * 80)
        
        # Save report with timestamp
        report_file = self._save_report(result.raw)
        
        # Save to memory if enabled
        if self.use_memory:
            self._save_to_memory(
                content=result.raw,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "type": "research_report",
                    "file": report_file
                }
            )
        
        return result.raw


def main():
    """Test the enhanced system"""
    print("Enhanced Multi-Agent Research System - WORKING VERSION")
    print("=" * 80)
    
    system = EnhancedMultiAgentSystem(use_memory=True, concurrent=False)
    
    sample_doc = """
    Artificial Intelligence in Healthcare 2025
    
    The healthcare industry is experiencing a transformation through AI adoption.
    Machine learning models are now assisting in diagnostics with 95% accuracy.
    
    Key applications include:
    - Medical imaging analysis
    - Drug discovery acceleration
    - Patient monitoring systems
    - Personalized treatment plans
    
    Challenges remain in data privacy and regulatory compliance.
    """
    
    result = system.analyze(sample_doc)
    print("\n\nAnalysis complete! Check src/output/ for saved reports.")


if __name__ == "__main__":
    main()