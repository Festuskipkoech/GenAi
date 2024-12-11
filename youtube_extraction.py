import os
from pytube import YouTube
import whisper
from googletrans import Translator
import markdown2
import re
import json
from pathlib import Path
import ast
import logging
from typing import Dict, List, Tuple, Optional

class YouTubeContentExtractor:
    def __init__(self, output_dir: str = "extracted_content"):
        """
        Initialize the content extractor with configuration.
        
        Args:
            output_dir: Directory to store extracted content
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = whisper.load_model("base")
        self.translator = Translator()
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def download_video(self, url: str) -> str:
        """
        Download YouTube video and return path to audio file.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Path to downloaded audio file
        """
        try:
            self.logger.info(f"Downloading video: {url}")
            yt = YouTube(url)
            audio_stream = yt.streams.filter(only_audio=True).first()
            audio_file = audio_stream.download(
                output_path=self.output_dir,
                filename=f"{yt.video_id}.mp3"
            )
            return audio_file
        except Exception as e:
            self.logger.error(f"Error downloading video: {str(e)}")
            raise

    def transcribe_audio(self, audio_path: str, target_language: str = "en") -> Dict:
        """
        Transcribe audio and translate if necessary.
        
        Args:
            audio_path: Path to audio file
            target_language: Target language code
            
        Returns:
            Dictionary containing transcription and translation
        """
        try:
            self.logger.info("Transcribing audio...")
            result = self.model.transcribe(audio_path)
            
            if result["language"] != target_language:
                self.logger.info(f"Translating from {result['language']} to {target_language}")
                translated = self.translator.translate(
                    result["text"],
                    dest=target_language
                )
                result["translated_text"] = translated.text
            
            return result
        except Exception as e:
            self.logger.error(f"Error in transcription: {str(e)}")
            raise

    def extract_content(self, transcript: str) -> Dict[str, List[str]]:
        """
        Extract key concepts, definitions, and action items from transcript.
        
        Args:
            transcript: Video transcript
            
        Returns:
            Dictionary containing categorized content
        """
        content = {
            "concepts": [],
            "definitions": [],
            "action_items": [],
            "code_segments": []
        }

        # Extract definitions (typically following "is", "means", "refers to")
        definition_patterns = [
            r'(?:is|means|refers to|defined as)[:]?\s+([^\.]+)',
            r'(?:definition of|meaning of)\s+(\w+)[:]?\s+([^\.]+)'
        ]
        
        for pattern in definition_patterns:
            definitions = re.finditer(pattern, transcript, re.IGNORECASE)
            content["definitions"].extend([d.group(1).strip() for d in definitions])

        # Extract concepts (typically following "concept", "principle", "idea")
        concept_patterns = [
            r'(?:concept|principle|idea)[:]?\s+([^\.]+)',
            r'important\s+(?:to understand|point)[:]?\s+([^\.]+)'
        ]
        
        for pattern in concept_patterns:
            concepts = re.finditer(pattern, transcript, re.IGNORECASE)
            content["concepts"].extend([c.group(1).strip() for c in concepts])

        # Extract action items (typically following "need to", "should", "must")
        action_patterns = [
            r'(?:need to|should|must|have to)[:]?\s+([^\.]+)',
            r'(?:step\s+\d+)[:]?\s+([^\.]+)'
        ]
        
        for pattern in action_patterns:
            actions = re.finditer(pattern, transcript, re.IGNORECASE)
            content["action_items"].extend([a.group(1).strip() for a in actions])

        # Extract code segments (typically between code markers or indented blocks)
        code_patterns = [
            r'```[\w]*\n(.*?)\n```',
            r'(?:example|code)[:]?\s*\n((?:\s{4}.*\n)+)'
        ]
        
        for pattern in code_patterns:
            code_segments = re.finditer(pattern, transcript, re.DOTALL)
            content["code_segments"].extend([c.group(1).strip() for c in code_segments])

        return content

    def extract_code_with_comments(self, transcript: str) -> List[Dict[str, str]]:
        """
        Extract code segments with their explanations.
        
        Args:
            transcript: Video transcript
            
        Returns:
            List of dictionaries containing code and explanations
        """
        code_blocks = []
        
        # Look for code blocks with explanations
        pattern = r'(?P<explanation>(?:let|now|here)?\s*(?:we|I)?\s*(?:have|write|create)\s+[^`]+)\s*```(?P<language>\w+)?\s*(?P<code>.*?)```'
        matches = re.finditer(pattern, transcript, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            code_block = {
                "explanation": match.group("explanation").strip(),
                "language": match.group("language") if match.group("language") else "text",
                "code": match.group("code").strip()
            }
            
            # Try to parse Python code to add additional comments
            if code_block["language"].lower() == "python":
                try:
                    tree = ast.parse(code_block["code"])
                    code_block["structure"] = self._analyze_code_structure(tree)
                except Exception:
                    pass
                    
            code_blocks.append(code_block)
            
        return code_blocks

    def _analyze_code_structure(self, tree: ast.AST) -> Dict:
        """
        Analyze Python code structure for better documentation.
        
        Args:
            tree: AST tree of Python code
            
        Returns:
            Dictionary containing code structure analysis
        """
        analysis = {
            "imports": [],
            "functions": [],
            "classes": [],
            "variables": []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                analysis["imports"].extend(n.name for n in node.names)
            elif isinstance(node, ast.ImportFrom):
                analysis["imports"].append(f"{node.module}: {[n.name for n in node.names]}")
            elif isinstance(node, ast.FunctionDef):
                analysis["functions"].append({
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "docstring": ast.get_docstring(node)
                })
            elif isinstance(node, ast.ClassDef):
                analysis["classes"].append({
                    "name": node.name,
                    "docstring": ast.get_docstring(node)
                })
            
        return analysis

    def save_content(self, content: Dict[str, List[str]], video_id: str):
        """
        Save extracted content in multiple formats.
        
        Args:
            content: Extracted content dictionary
            video_id: YouTube video ID for file naming
        """
        output_path = self.output_dir / video_id
        output_path.mkdir(exist_ok=True)
        
        # Save as JSON
        with open(output_path / "content.json", "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        
        # Save as Markdown
        markdown_content = "# Video Content Summary\n\n"
        
        for category, items in content.items():
            if items:  # Only include non-empty categories
                markdown_content += f"## {category.replace('_', ' ').title()}\n\n"
                for item in items:
                    markdown_content += f"- {item}\n"
                markdown_content += "\n"
        
        with open(output_path / "content.md", "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        # Save code segments in separate files
        if content["code_segments"]:
            code_path = output_path / "code"
            code_path.mkdir(exist_ok=True)
            
            for i, code in enumerate(content["code_segments"], 1):
                try:
                    # Try to detect the language and use appropriate extension
                    if "def " in code or "class " in code:
                        ext = ".py"
                    elif "<html" in code.lower():
                        ext = ".html"
                    elif "{" in code and ":" in code:
                        ext = ".json"
                    else:
                        ext = ".txt"
                        
                    with open(code_path / f"segment_{i}{ext}", "w", encoding="utf-8") as f:
                        f.write(code)
                except Exception as e:
                    self.logger.error(f"Error saving code segment {i}: {str(e)}")

def main():
    """
    Main function to demonstrate usage of the YouTubeContentExtractor.
    """
    # Initialize the extractor
    extractor = YouTubeContentExtractor()
    
    # Get video URL from user
    url = input("Enter YouTube video URL: ")
    
    try:
        # Download and process the video
        audio_path = extractor.download_video(url)
        
        # Get target language from user (default to English)
        target_lang = input("Enter target language code (e.g., 'en' for English, press Enter for default): ").strip() or "en"
        
        # Transcribe and translate if necessary
        result = extractor.transcribe_audio(audio_path, target_lang)
        transcript = result.get("translated_text", result["text"])
        
        # Extract content
        content = extractor.extract_content(transcript)
        
        # Extract code with explanations
        code_blocks = extractor.extract_code_with_comments(transcript)
        content["code_segments"] = [block["code"] for block in code_blocks]
        content["code_explanations"] = [block["explanation"] for block in code_blocks]
        
        # Save content
        video_id = YouTube(url).video_id
        extractor.save_content(content, video_id)
        
        print(f"\nContent has been extracted and saved in the '{extractor.output_dir}/{video_id}' directory.")
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main()