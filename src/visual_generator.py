import re
from typing import Dict, List, Optional
import logging
from collections import Counter # ✅ FIXED: Added missing import
from src.system_prompt import SystemPrompts
from src.utils import ConfigManager
# Setup logging
logger = logging.getLogger(__name__)
# Load configuration
MAX_STEPS = ConfigManager.get('MAX_DIAGRAM_STEPS', 100, config_type=int)
SIMPLE_THRESHOLD = ConfigManager.get('VISUAL_SIMPLE_THRESHOLD', 3, config_type=int)
MEDIUM_THRESHOLD = ConfigManager.get('VISUAL_MEDIUM_THRESHOLD', 6, config_type=int)
COMPLEX_THRESHOLD = ConfigManager.get('VISUAL_COMPLEX_THRESHOLD', 10, config_type=int)
# Initialize prompts
prompts = SystemPrompts()
class EnhancedContentAnalyzer:
    """Extract complete steps with advanced pattern detection."""
    def analyze_complexity(self, steps: List[Dict], content: str) -> Dict:
        """Determine complexity level and recommend graph type."""
        step_count = len(steps)
        # Check for decision points
        has_decisions = bool(re.search(
            r'\b(if|else|decision|choice|yes|no|approve|reject|condition)\b',
            content,
            re.IGNORECASE
        ))
        # Check for multiple actors/roles
        actors = re.findall(
            r'\b(committee|secretary|hod|hop|officer|bidder|system|egp|hodr|ao|do)\b',
            content,
            re.IGNORECASE
        )
        unique_actors = len(set([a.lower() for a in actors]))
        logger.info(
            f"Complexity Analysis: steps={step_count}, decisions={has_decisions}, "
            f"actors={unique_actors}"
        )
        # 🔥 NEW LOGIC: Never summarize, always show all steps
        if step_count <= 8:
            return {
                'level': 'simple',
                'recommended_graph': 'simple_flowchart',
                'should_summarize': False
            }
        elif step_count <= 15:
            return {
                'level': 'medium',
                'recommended_graph': 'simple_flowchart', # Keep flowchart
                'should_summarize': False # NEVER summarize
            }
        else: # step_count > 15
            return {
                'level': 'complex',
                'recommended_graph': 'simple_flowchart', # Still flowchart
                'should_summarize': False # NO summarization even for long processes
            }
    def summarize_steps(self, steps: List[Dict]) -> List[Dict]:
        """Group steps into high-level phases for very long processes."""
        if len(steps) <= 10:
            return steps
       
        logger.info(f"📊 Summarizing {len(steps)} steps into phases...")
       
        # Determine group size based on total steps
        if len(steps) > 20:
            group_size = 4
        elif len(steps) > 15:
            group_size = 3
        else:
            group_size = 3
       
        phases = []
       
        for i in range(0, len(steps), group_size):
            group = steps[i:i+group_size]
            phase_num = (i // group_size) + 1
           
            # Extract keywords from all steps in this group
            all_keywords = []
            for s in group:
                words = s['description'].split()[:5]
                all_keywords.extend([w.lower() for w in words if len(w) > 4])
            # Find most common keyword
            common_word = Counter(all_keywords).most_common(1)[0][0] if all_keywords else "Process"
            # Create descriptive title
            phase_title = f"{common_word.title()} Phase"
            # Create detailed sub-summary
            sub_steps_text = " | ".join([
                self.clean_for_mermaid(s['description'], max_length=50)
                for s in group
            ])
           
            phases.append({
                'number': phase_num,
                'description': phase_title,
                'sub_steps': sub_steps_text,
                'original_count': len(group)
            })
       
        logger.info(f"✅ Summarized into {len(phases)} phases")
        return phases
    def extract_complete_steps(self, content: str) -> List[Dict]:
        """Extract steps from responses with structured content."""
        steps = []
        # Pattern 1: Numbered steps
        pattern1 = r'(?:Step\s+)?(\d+)[:\.\-]\s*([^\n]+(?:\n(?!(?:Step\s+)?\d+[:\.\-])[^\n]+)*)'
        matches = list(re.finditer(pattern1, content, re.MULTILINE | re.IGNORECASE))
        # 🔥 ADD THIS NEW PATTERN for bullet points with colons
        if not matches:
            pattern_colon = r'[•\-\*]\s*([^:]+):\s*([^\n•\-\*]+)'
            colon_matches = list(re.finditer(pattern_colon, content, re.MULTILINE))
            if colon_matches:
                logger.info(f"✓ Found {len(colon_matches)} bullet points with descriptions")
                for i, match in enumerate(colon_matches, 1):
                    title = match.group(1).strip()
                    desc = match.group(2).strip()
                    full_text = f"{title}: {desc}"
                    cleaned = self._clean_step_description(full_text)
                    if cleaned and len(cleaned) > 10:
                        matches.append(type('obj', (object,), {
                            'group': lambda x: i if x == 1 else cleaned
                        })())
        # Pattern 2: Bullet points
        if not steps:
            logger.info("No numbered steps found, trying bullet points...")
            bullet_pattern = r'(?:^|\n)\s*[•\-\*▪▫◦●○]\s*([^\n]+(?:\n(?![•\-\*▪▫◦●○])[^\n]+)*)'
            bullet_matches = list(re.finditer(bullet_pattern, content, re.MULTILINE))
           
            if bullet_matches:
                logger.info(f"✓ Found {len(bullet_matches)} bullet points")
                for i, match in enumerate(bullet_matches, 1):
                    text = match.group(1).strip()
                    cleaned = self._clean_step_description(text)
                    if cleaned and len(cleaned) > 10:
                        steps.append({'number': i, 'description': cleaned})
        # Deduplicate
        seen = set()
        unique_steps = []
        for step in steps:
            desc_key = step['description'][:40].lower()
            if desc_key not in seen:
                seen.add(desc_key)
                unique_steps.append(step)
        # Re-number sequentially
        unique_steps.sort(key=lambda x: x['number'])
        for i, step in enumerate(unique_steps, 1):
            step['number'] = i
        logger.info(f"✅ Final: Extracted {len(unique_steps)} unique steps")
        return unique_steps[:MAX_STEPS]
    def _clean_step_description(self, text: str) -> str:
        """Clean and format step descriptions."""
        if not text:
            return ""
       
        # Remove citations
        text = re.sub(r'\(slide \d+\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[Document \d+\]', '', text, flags=re.IGNORECASE)
       
        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        text = text.strip()
       
        # Remove leading numbers/bullets
        text = re.sub(r'^[\d\.\)\-•▪▫◦●○\*\s]+', '', text)
       
        # Remove trailing period if text is short
        if len(text) < 100:
            text = text.rstrip('.')
       
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
       
        return text
    def clean_for_mermaid(self, text: str, max_length: int = 300) -> str:
        """Clean text for Mermaid - BULLETPROOF version."""
        if not text:
            return "Step"
       
        # STEP 1: Remove ALL problematic characters FIRST
        dangerous_chars = {
            '"': '', "'": '', '`': '',
            '(': '', ')': '', '[': '', ']': '',
            '{': '', '}': '', '<': '', '>': '',
            '|': ' ', ';': ',', '#': 'No',
            '&': 'and', '\n': ' ', '\r': ' ',
            '\t': ' ', '*': '', '_': ''
        }
       
        for char, replacement in dangerous_chars.items():
            text = text.replace(char, replacement)
       
        # STEP 2: Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
       
        # STEP 3: Intelligent truncation (keep complete words)
        if len(text) > max_length:
            text = text[:max_length].rsplit(' ', 1)[0] + '...'
       
        # STEP 4: Safety check
        return text if text and len(text) > 3 else "Process Step"
class ProfessionalVisualizationGenerator:
    """Generate professional flowcharts with swimlanes."""
    def __init__(self):
        self.analyzer = EnhancedContentAnalyzer()
    def generate_mindmap(self, steps: List[Dict], title: str, language: str = 'en') -> Optional[str]:
        """Generate mindmap for hierarchical information."""
        if not steps:
            return None
        logger.info(f"Generating mindmap with {len(steps)} nodes")
       
        lines = [
            "%%{init: {'theme':'base', 'themeVariables': {'fontSize':'14px'}}}%%",
            "mindmap",
            f" root(({title}))"
        ]
        if len(steps) > 8:
            mid = len(steps) // 2
            categories = {
                "First Half": steps[:mid],
                "Second Half": steps[mid:]
            }
            for cat_name, cat_steps in categories.items():
                lines.append(f" {cat_name}")
                for step in cat_steps:
                    # 🔥 Extract only the main idea (first 30 chars)
                    desc_text = step['description']
                    if '.' in desc_text[:50]:
                        desc_text = desc_text.split('.')[0]
                    else:
                        desc_text = desc_text[:30]
                    cleaned = self.analyzer.clean_for_mermaid(desc_text, max_length=30)
                    lines.append(f" {cleaned}")
        else:
            for step in steps:
                cleaned = self.analyzer.clean_for_mermaid(step['description'], max_length=150)
                lines.append(f" {cleaned}")
        return "\n".join(lines)
    def generate_hierarchical_mindmap(self, steps: List[Dict], title: str, language: str = 'en') -> Optional[str]:
       
        if not steps or len(steps) < 3:
            return None
       
        logger.info(f"Generating numbered flowchart with {len(steps)} steps")
       
        labels = {
            'en': {'start': 'START', 'end': 'END'},
            'sw': {'start': 'ANZA', 'end': 'MWISHO'}
        }
        lang_labels = labels.get(language, labels['en'])
       
        lines = [
            "%%{init: {'theme':'base', 'themeVariables': {'fontSize':'14px'}}}%%",
            "graph TD"
        ]
       
        # Add START node
        lines.append(f" Start([{lang_labels['start']}]):::startClass")
        lines.append("")
       
        # Add numbered steps with VERY SHORT labels
        prev_node = "Start"
        for i, step in enumerate(steps, 1):
            node_id = f"S{i}"
           
            # 🔥 CRITICAL: Extract ONLY the first sentence or 40 chars
            desc = step['description']
           
            # Get first sentence (up to first period or 40 chars, whichever comes first)
            if '.' in desc[:60]:
                desc = desc.split('.')[0]
            else:
                desc = desc[:40]
           
            # Clean for Mermaid
            desc = self.analyzer.clean_for_mermaid(desc, max_length=40)
           
            # Add step number prefix for clarity
            label = f"{i}. {desc}"
           
            # Add node
            lines.append(f' {node_id}["{label}"]:::stepClass')
           
            # Add connection from previous node
            lines.append(f" {prev_node} --> {node_id}")
            lines.append("")
           
            prev_node = node_id
       
        # Add END node
        lines.append(f" End([{lang_labels['end']}]):::endClass")
        lines.append(f" {prev_node} --> End")
        lines.append("")
       
        # Add modern styling
        lines.extend([
            " classDef startClass fill:#10b981,stroke:#059669,stroke-width:3px,color:#fff,font-weight:bold",
            " classDef endClass fill:#ef4444,stroke:#dc2626,stroke-width:3px,color:#fff,font-weight:bold",
            " classDef stepClass fill:#60a5fa,stroke:#3b82f6,stroke-width:2px,color:#fff,font-weight:600"
        ])
       
        return "\n".join(lines)
    def generate_swimlane_diagram(self, steps: List[Dict], content: str,
                                  title: str, language: str = 'en') -> Optional[str]:
        """Generate swimlane diagram with actor-based lanes."""
        actor_keywords = {
            'eGP System': ['egp', 'system', 'platform', 'portal'],
            'Committee': ['committee', 'tender opening committee', 'evaluation committee'],
            'Secretary': ['secretary', 'secretariat'],
            'Head of Procurement': ['hop', 'head of procurement', 'procurement head'],
            'Bidder/Supplier': ['bidder', 'supplier', 'vendor', 'tenderer'],
            'HODR': ['hodr', 'hod', 'department head'],
            'Procurement Officer': ['procurement officer', 'officer', 'pe'],
            'Accounting Officer': ['ao', 'accounting officer', 'finance']
        }
        active_actors = [
            actor for actor, kw in actor_keywords.items()
            if any(k in content.lower() for k in kw)
        ]
        if not active_actors:
            active_actors = ['System', 'User', 'Process']
        logger.info(f"Generating swimlane with actors: {active_actors}")
        labels = {
            'en': {'start': 'START', 'end': 'END'},
            'sw': {'start': 'ANZA', 'end': 'MWISHO'}
        }
        lang_labels = labels.get(language, labels['en'])
        lines = [
            "%%{init: {'theme':'base'}}%%",
            "graph TB",
            f" Start([{lang_labels['start']}]):::startNode"
        ]
        prev_node = "Start"
        node_counter = 0
       
        for step in steps:
            node_counter += 1
            node_id = f"N{node_counter}"
           
            # 🔥 FIXED: Much shorter text for readability
            desc_text = step['description']
            # Extract only first sentence or 50 chars
            if '.' in desc_text[:70]:
                desc_text = desc_text.split('.')[0]
            else:
                desc_text = desc_text[:50]
            desc = self.analyzer.clean_for_mermaid(desc_text, max_length=150)
            # Add step number for clarity
            desc = f"{step['number']}. {desc}"
           
            is_decision = any(
                word in step['description'].lower()
                for word in ['if', 'decision', 'yes', 'no', '?', 'whether', 'check', 'approved', 'rejected']
            )
           
            if is_decision:
                lines.append(f' {node_id}{{{{{desc}}}}}:::decisionNode')
            else:
                lines.append(f' {node_id}["{desc}"]:::processNode')
           
            lines.append(f' {prev_node} --> {node_id}')
            prev_node = node_id
        lines.append(f" End([{lang_labels['end']}]):::endNode")
        lines.append(f" {prev_node} --> End")
        lines.extend([
            "",
            " classDef startNode fill:#90EE90,stroke:#2E8B57,stroke-width:3px,color:#000",
            " classDef endNode fill:#FFB6C1,stroke:#C71585,stroke-width:3px,color:#000",
            " classDef processNode fill:#87CEEB,stroke:#4682B4,stroke-width:2px,color:#000",
            " classDef decisionNode fill:#FFD700,stroke:#FF8C00,stroke-width:2px,color:#000"
        ])
        return "\n".join(lines)
    def generate_simple_flowchart(self, steps: List[Dict],
                                title: str = "Process Flow",
                                language: str = 'en') -> Optional[str]:
        """Generate detailed flowchart with FULL step descriptions."""
        if not steps or len(steps) < 2:
            return None
       
        if len(steps) > 20:
            logger.warning(f"⚠️ Too many steps ({len(steps)}) for flowchart")
            return None
        labels = {
            'en': {'start': 'START', 'end': 'END'},
            'sw': {'start': 'ANZA', 'end': 'MWISHO'}
        }
        lang_labels = labels.get(language, labels['en'])
        mermaid_lines = [
            "%%{init: {'theme':'base', 'themeVariables': {'fontSize':'13px'}}}%%",
            "flowchart TD"
        ]
       
        # Add START node with gradient styling
        mermaid_lines.append(f" Start([📍 {lang_labels['start']}]):::startClass")
        mermaid_lines.append("")
       
        # Add steps with FULL descriptions (up to 150 chars)
        prev_node = "Start"
        for i, step in enumerate(steps, 1):
            node_id = f"Step{i}"
           
            # Get FULL description (keep it detailed)
            desc = step['description']
           
            # Only trim if really too long, keep first sentence structure
            if len(desc) > 150:
                # Try to keep complete sentences
                sentences = desc.split('.')
                desc = sentences[0]
                if len(desc) > 150:
                    desc = desc[:147] + "..."
           
            # Clean for Mermaid but keep detail
            desc = self.analyzer.clean_for_mermaid(desc, max_length=150)
           
            # Format: "Step X: Description"
            label = f"Step {i}:<br/>{desc}"
           
            # Add node with proper styling
            mermaid_lines.append(f' {node_id}["{label}"]:::stepClass')
           
            # Connect from previous
            mermaid_lines.append(f" {prev_node} --> {node_id}")
            mermaid_lines.append("")
           
            prev_node = node_id
       
        # Add END node
        mermaid_lines.append(f" End([✅ {lang_labels['end']}]):::endClass")
        mermaid_lines.append(f" {prev_node} --> End")
        mermaid_lines.append("")
       
        # Professional gradient styling
        mermaid_lines.extend([
            " classDef startClass fill:#10b981,stroke:#059669,stroke-width:3px,color:#fff,font-weight:bold,font-size:14px",
            " classDef endClass fill:#ef4444,stroke:#dc2626,stroke-width:3px,color:#fff,font-weight:bold,font-size:14px",
            " classDef stepClass fill:#3b82f6,stroke:#1e40af,stroke-width:2px,color:#fff,font-weight:500,font-size:13px"
        ])
       
        return "\n".join(mermaid_lines)
   
    def generate_ultra_simple_numbered_flow(self, steps: List[Dict],
                                            title: str = "Process",
                                            language: str = 'en') -> Optional[str]:
        """
        🔥 NEW: Ultra-simple numbered flowchart for maximum readability.
        Each step gets: Number + Short Label (max 30 chars)
        """
        if not steps or len(steps) < 2:
            return None
       
        logger.info(f"Generating ultra-simple flowchart with {len(steps)} steps")
       
        labels = {
            'en': {'start': 'START', 'end': 'COMPLETE'},
            'sw': {'start': 'ANZA', 'end': 'MWISHO'}
        }
        lang_labels = labels.get(language, labels['en'])
       
        lines = [
            "%%{init: {'theme':'base', 'themeVariables': {'fontSize':'16px'}}}%%",
            "flowchart TD"
        ]
       
        # START
        lines.append(f" A([📍 {lang_labels['start']}])")
        lines.append(" style A fill:#10b981,stroke:#059669,stroke-width:4px,color:#fff")
        lines.append("")
       
        prev = "A"
       
        # Steps with ULTRA-SHORT labels
        for i, step in enumerate(steps, 1):
            node_id = chr(65 + i) if i < 25 else f"N{i}" # A, B, C... or N26, N27...
           
            # Get ONLY the first few words (max 25 chars)
            desc = step['description'].split()[0:4] # First 4 words only
            short_label = ' '.join(desc)
           
            if len(short_label) > 25:
                short_label = short_label[:22] + "..."
           
            # Clean thoroughly
            short_label = self.analyzer.clean_for_mermaid(short_label, max_length=25)
           
            # Format: "Step 1: Label"
            label = f"Step {i}<br/>{short_label}"
           
            lines.append(f" {node_id}[\"{label}\"]")
            lines.append(f" style {node_id} fill:#3b82f6,stroke:#1d4ed8,stroke-width:2px,color:#fff")
            lines.append(f" {prev} --> {node_id}")
            lines.append("")
           
            prev = node_id
       
        # END
        end_id = chr(65 + len(steps) + 1) if len(steps) < 24 else f"END"
        lines.append(f" {end_id}([✅ {lang_labels['end']}])")
        lines.append(f" style {end_id} fill:#ef4444,stroke:#dc2626,stroke-width:4px,color:#fff")
        lines.append(f" {prev} --> {end_id}")
       
        return "\n".join(lines)
class StrictVisualContentGenerator:
    """Generate visuals with DYNAMIC graph selection and smart summarization."""
    def __init__(self):
        self.generator = ProfessionalVisualizationGenerator()
        self.analyzer = EnhancedContentAnalyzer()
    def should_generate_visual(self, query: str, rag_response: str = "") -> bool:
            """Generate visual only if response contains steps/process."""
           
            # Check if response is too short
            if not rag_response or len(rag_response.strip()) < 100:
                logger.info("✗ Response too short for visual")
                return False
           
            # Check for step indicators in the response
            step_indicators = [
                'step', 'steps', 'process', 'procedure', 'workflow',
                'first', 'second', 'third', 'next', 'then', 'finally',
                'stage', 'phase', '1.', '2.', '3.',
                'hatua', # Swahili for step
                '•', '-', '*' # Bullet points
            ]
           
            response_lower = rag_response.lower()
            has_steps = any(indicator in response_lower for indicator in step_indicators)
           
            if has_steps:
                logger.info("✓ Visual generation ENABLED (steps detected)")
                return True
            else:
                logger.info("✗ No steps detected - skipping visual")
                return False
    def process_visual_request(self, query: str, rag_response: str,
                        language: str = 'en', force_generate: bool = False) -> Optional[Dict]:
        """Process visual generation with detailed flowcharts."""
       
        if not self.should_generate_visual(query, rag_response) and not force_generate:
            logger.info("⊘ Visual generation skipped")
            return None
        logger.info("✓ Processing visual request...")
        steps = self.analyzer.extract_complete_steps(rag_response)
        if force_generate and (not steps or len(steps) < 2):
            logger.info("🔥 Force mode: Creating steps from response")
           
            clean_response = rag_response
            if "You might also want to ask" in clean_response:
                clean_response = clean_response.split("You might also want to ask")[0]
            if "Do you want to see visuals" in clean_response:
                clean_response = clean_response.split("Do you want to see visuals")[0]
           
            paragraphs = [p.strip() for p in clean_response.split('\n\n') if p.strip() and len(p.strip()) > 30]
           
            if not paragraphs:
                import re
                sentences = [s.strip() + '.' for s in re.split(r'[.!?]+', clean_response) if s.strip() and len(s.strip()) > 30]
                paragraphs = sentences
           
            steps = []
            for i, para in enumerate(paragraphs[:15], 1):
                steps.append({
                    'number': i,
                    'description': para[:300]
                })
           
            if not steps:
                logger.warning("❌ Could not create steps")
                return None
        if not steps or len(steps) < 2:
            logger.warning(f"⚠ Insufficient steps ({len(steps)})")
            return None
        logger.info(f"✓ Extracted {len(steps)} steps")
        analysis = self.analyzer.analyze_complexity(steps, rag_response)
        complexity = analysis['level']
       
        # 🔥 ALWAYS use simple flowchart, NEVER summarize
        viz_type = 'simple_flowchart'
        mermaid_code = self.generator.generate_simple_flowchart(
            steps, self._extract_title(query, language), language
        )
        if not mermaid_code:
            logger.error("❌ Flowchart generation failed")
            return None
        logger.info(f"✅ Visual generated: {viz_type} with {len(steps)} steps")
       
        return {
            'type': 'mermaid',
            'content': mermaid_code,
            'title': self._extract_title(query, language),
            'language': language,
            'visualization_type': viz_type,
            'complexity': complexity,
            'explicit_request': True,
            'step_count': len(steps),
            'original_step_count': len(steps),
            'was_summarized': False
        }
    def _extract_title(self, query: str, language: str) -> str:
        """Extract meaningful title from query."""
        visual_words = [
            'show', 'create', 'draw', 'make', 'diagram', 'flowchart',
            'visual', 'chart', 'graph', 'visuals', 'visualize',
            'illustrate', 'display', 'me', 'a', 'the', 'in', 'of',
            'please', 'can', 'you', 'with', 'for', 'flow',
            'onyesha', 'saketi', 'picha'
        ]
       
        words = [w for w in query.split() if w.lower() not in visual_words and len(w) > 2]
        title = ' '.join(words[:6]).title()
       
        if not title:
            title = 'Process Workflow' if language == 'en' else 'Mtiririko wa Mchakato'
       
        return title
def process_visual_request(query: str, rag_response: str,
                          language: str = 'en', force_generate: bool = False) -> Optional[Dict]:
    """MAIN ENTRY POINT: Process visual generation."""
    generator = StrictVisualContentGenerator()
    return generator.process_visual_request(query, rag_response, language, force_generate)
__all__ = [
    'process_visual_request',
    'EnhancedContentAnalyzer',
    'ProfessionalVisualizationGenerator',
    'StrictVisualContentGenerator'
]
