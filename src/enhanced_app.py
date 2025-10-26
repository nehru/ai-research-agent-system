"""
Enhanced Gradio UI for Multi-Agent Research System
Features: Memory toggle, Concurrent/Sequential processing options
"""

import gradio as gr
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from enhanced_multi_agent_system import EnhancedMultiAgentSystem


def analyze_document(document_content, use_memory, use_concurrent):
    """
    Analyze document using multi-agent system with options
    
    Args:
        document_content (str): Document text to analyze
        use_memory (bool): Enable memory between runs
        use_concurrent (bool): Use concurrent processing
        
    Returns:
        str: Analysis results
    """
    if not document_content or not document_content.strip():
        return "Please provide document content to analyze."
    
    try:
        # Create system with selected options
        system = EnhancedMultiAgentSystem(
            use_memory=use_memory,
            concurrent=use_concurrent
        )
        
        # Analyze the document
        result = system.analyze(document_content)
        
        # Read the actual report from the saved file (workaround for result.raw issue)
        try:
            with open("src/output/latest_report.txt", "r", encoding="utf-8") as f:
                report_content = f.read()
            
            if report_content and len(report_content) > 100:
                return report_content
            else:
                # Fallback to result if file is empty
                return result if result else "Report generated but content appears empty. Check src/output/latest_report.txt"
        except:
            # If file read fails, return result
            return result if result else "Report generated. Check src/output/latest_report.txt for full content."
        
    except Exception as e:
        return f"Error during analysis: {str(e)}\n\nMake sure Ollama is running with required models:\n- ollama pull llama3\n- ollama pull deepseek-r1:7b\n- ollama pull qwen2.5-coder"


def create_ui():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Multi-Agent Research System", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Multi-Agent Research System
            
            Analyze documents using specialized AI agents with different LLM models.
            Each agent brings unique expertise to create comprehensive research reports.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Configuration Options")
                
                use_memory = gr.Checkbox(
                    label="Enable Memory (ChromaDB)",
                    value=True,
                    info="Save and retrieve context from previous runs"
                )
                
                use_concurrent = gr.Checkbox(
                    label="Use Concurrent Processing",
                    value=False,
                    info="Run agents in parallel (faster) vs sequential (more accurate)"
                )
        
        with gr.Row():
            with gr.Column():
                document_input = gr.Textbox(
                    label="Document Content",
                    placeholder="Paste your document text here...",
                    lines=15,
                    max_lines=20
                )
                
                analyze_btn = gr.Button("Analyze Document", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column():
                output_text = gr.Textbox(
                    label="Analysis Results",
                    lines=20,
                    max_lines=30
                )
        
        # Agent information
        with gr.Accordion("Agent Information", open=False):
            gr.Markdown(
                """
                ### Agent Roles & Models
                
                | Agent | Model | Specialization |
                |-------|-------|----------------|
                | Research Agent | Llama 3.2 | Information extraction & organization |
                | Analysis Agent | DeepSeek-R1 | Deep reasoning & insight generation |
                | Writer Agent | Qwen 2.5 | Report synthesis & clear communication |
                | Manager Agent | Llama 3.2 | Workflow coordination & quality control |
                
                ### Processing Modes
                
                **Sequential Processing** (Default):
                - Agents work one after another
                - Each agent builds on previous work
                - Higher quality, more coherent outputs
                - Takes longer (~2-5 minutes)
                
                **Concurrent Processing** (Experimental):
                - Agents work simultaneously
                - Faster processing (~1-3 minutes)
                - May have less coherent integration
                - Good for independent analysis tasks
                
                ### Memory Feature
                
                When enabled, the system:
                - Saves reports to ChromaDB vector store
                - Retrieves relevant past findings
                - Provides context from previous runs
                - Builds knowledge over time
                """
            )
        
        # Example documents
        with gr.Accordion("Sample Documents", open=False):
            gr.Markdown(
                """
                ### Quick Test Documents
                
                Click on any example below to load it:
                """
            )
            
            gr.Examples(
                examples=[
                    ["AI Industry Trends 2025\n\nThe artificial intelligence market has experienced unprecedented growth in 2025, with global market valuation reaching $450 billion, representing a 35% year-over-year increase.\n\nKey Technology Developments:\n- Large Language Models stabilizing at 70B-400B parameters\n- Multi-agent systems emerging as dominant architecture\n- Open source models reaching GPT-4 level performance\n- Local deployment becoming standard for enterprises"],
                    
                    ["Q4 2024 Company Performance\n\nRevenue: $45.2M (up 28% YoY)\nGross Margin: 72%\nOperating Margin: 18%\nCustomer Count: 362 enterprise clients\n\nKey Achievements:\n- Released 3 major features with 95% satisfaction\n- Reduced time-to-market by 40% using AI\n- Sales team achieved 115% of quota\n- Employee satisfaction: 4.2/5.0"],
                    
                    ["Multi-Agent Systems Research\n\nAbstract: This paper examines multi-agent systems as a paradigm shift in software architecture.\n\nKey Concepts:\n- Agent autonomy and reactivity\n- Hierarchical vs peer-to-peer organization\n- Sequential vs concurrent processing\n- Applications in research, development, and automation\n\nChallenges: Coordination complexity, cost management, reliability concerns"]
                ],
                inputs=document_input
            )
        
        # Usage instructions
        with gr.Accordion("How to Use", open=False):
            gr.Markdown(
                """
                ### Instructions
                
                1. **Paste your document** into the text area above
                2. **Configure options**:
                   - Enable Memory to save context between runs
                   - Enable Concurrent for faster (but less integrated) processing
                3. **Click "Analyze Document"** and wait 2-5 minutes
                4. **View the report** in the output area
                5. **Check saved reports** in `src/output/` directory
                
                ### Requirements
                
                Make sure Ollama is running with these models:
                ```bash
                ollama pull llama3.2
                ollama pull deepseek-r1:7b
                ollama pull qwen2.5:7b
                ```
                
                ### Tips
                
                - Start with shorter documents (500-2000 words)
                - Sequential mode produces better quality for complex documents
                - Memory feature improves with multiple runs on related topics
                - Reports are saved with timestamps in `src/output/`
                """
            )
        
        # Connect button to function
        analyze_btn.click(
            fn=analyze_document,
            inputs=[document_input, use_memory, use_concurrent],
            outputs=output_text
        )
    
    return demo


if __name__ == "__main__":
    print("Starting Enhanced Multi-Agent Research System UI...")
    print("=" * 80)
    print("\nFeatures enabled:")
    print("  - Memory persistence (ChromaDB)")
    print("  - Concurrent/Sequential processing options")
    print("  - Timestamped report saving")
    print("\nMake sure Ollama is running with required models:")
    print("  - ollama pull llama3.2")
    print("  - ollama pull deepseek-r1:7b")
    print("  - ollama pull qwen2.5:7b")
    print("\nOpen your browser to the URL shown below")
    print("=" * 80)
    print()
    
    demo = create_ui()
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
