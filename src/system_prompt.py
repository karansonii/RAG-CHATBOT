# src/system_prompt.py
"""
COMPLETELY FIXED - Proper formatting for steps and bullet points + PURE DYNAMIC GREETING SYSTEM
"""
class SystemPrompts:
    """Container for all system prompts used across the application."""
   
    # ==================== RAG PIPELINE PROMPTS ====================
   
    @staticmethod
    def get_rag_base_prompt(context: str, query: str, is_doc_context: bool = False) -> str:
        """
        COMPLETELY FIXED: Forces AI to output in clean, parseable format.
       
        Args:
            context: Retrieved document chunks
            query: User's question
            is_doc_context: Whether this is a document summarization request
        """
       
        # 🔥 DETECT formatting keywords
        query_lower = query.lower()
       
        wants_steps = any(keyword in query_lower for keyword in [
            'step', 'steps', 'step by step', 'step-by-step',
            'procedure', 'process', 'how to', 'guide',
            'instructions', 'one by one', 'explain how',
            'tutorial', 'walkthrough', 'method'
        ])
       
        wants_points = any(keyword in query_lower for keyword in [
            'points', 'point', 'bullet', 'list',
            'summarize', 'summary', 'briefly',
            'key points', 'main points', 'overview'
        ])
       
        # Build format instruction based on request type
        if wants_steps:
            format_instruction = """
🔥 CRITICAL FORMATTING INSTRUCTION - YOU MUST FOLLOW THIS EXACTLY:
Format your answer as NUMBERED STEPS using this EXACT structure:
Step 1: [Clear title of the step]
[Explanation of this step - keep it clear and detailed]
Step 2: [Clear title of the step]
[Explanation of this step - keep it clear and detailed]
Step 3: [Clear title of the step]
[Explanation of this step - keep it clear and detailed]
MANDATORY FORMATTING RULES:
✓ Start each step with "Step X:" (where X is the step number)
✓ Put the step title on the SAME LINE as "Step X:"
✓ Put the explanation on a NEW LINE after the title
✓ Add ONE BLANK LINE between steps
✓ Number steps sequentially (1, 2, 3, 4...)
✓ Keep step titles concise (3-8 words)
✓ Make explanations detailed but clear
For sub-steps, use this format:
Step 1: Main Step Title
Main explanation here.
  • Sub-step detail one
  • Sub-step detail two
  • Sub-step detail three
EXAMPLE OF PERFECT FORMAT:
Step 1: Register on the eGP Portal
Visit the official eGP website at egp.go.tz and click on "New Registration". You'll need a valid email address and business details.
Step 2: Verify Your Email Address
Check your inbox for the verification email sent by eGP. Click the verification link within 24 hours to activate your account.
Step 3: Complete Your Business Profile
Log in to your account and fill in all required business information including registration number, tax details, and contact information.
Step 4: Upload Required Documents
Prepare and upload the following documents:
  • Business registration certificate
  • Tax clearance certificate
  • Bank account details
  • Valid business license
NOW WRITE YOUR ANSWER USING THE EXACT FORMAT SHOWN ABOVE.
DO NOT use bold markdown (**). Just use plain text with proper spacing.
"""
       
        elif wants_points:
            format_instruction = """
🔥 CRITICAL FORMATTING INSTRUCTION - YOU MUST FOLLOW THIS EXACTLY:
Format your answer as BULLET POINTS using this EXACT structure:
- [First point title or key concept]
[Detailed explanation of this point]
- [Second point title or key concept]
[Detailed explanation of this point]
- [Third point title or key concept]
[Detailed explanation of this point]
MANDATORY FORMATTING RULES:
✓ Start each point with a bullet symbol (•)
✓ Put the main point on the SAME LINE as the bullet
✓ Put detailed explanation on NEW LINES below
✓ Add ONE BLANK LINE between points
✓ Keep point titles concise and clear
✓ Make explanations informative but readable
For sub-points, use this format:
- Main Point Title
  Explanation here.
  - Sub-point one
  - Sub-point two
  - Sub-point three
EXAMPLE OF PERFECT FORMAT:
- Access to Government Procurement Opportunities (AGPO)
AGPO is a government initiative that reserves 30% of all procurement opportunities for youth, women, and persons with disabilities. This helps special groups participate in government tenders.
- Business Registration Service (BRS)
BRS is an online platform where businesses can register and obtain necessary licenses. It streamlines the registration process and reduces the time needed to start a business.
- Electronic Government Procurement (eGP)
eGP is the digital platform for submitting tender applications. It allows businesses to view tender opportunities, download documents, and submit bids electronically.
- Integrated Financial Management Information System (IFMIS)
IFMIS is used for government financial operations including payments and budget tracking. Suppliers can check payment status through this system.
NOW WRITE YOUR ANSWER USING THE EXACT FORMAT SHOWN ABOVE.
DO NOT use bold markdown (**). Just use plain text with proper spacing.
"""
       
        else:
            format_instruction = """
Format your answer in clear paragraphs:
- Write naturally and conversationally
- Separate different ideas with blank lines
- Use bullet points (•) if listing multiple items
- Keep paragraphs focused and readable
"""
       
        base = f"""You are an intelligent AI assistant helping users understand documents.
Using the following context from the knowledge base, answer the user's question comprehensively.
CRITICAL INSTRUCTIONS:
1. Answer ONLY using information from the context below
2. Provide complete, accurate, and detailed answers
3. NEVER mention sources like "Document 1", "according to ppt_X", or "(Source: ...)"
4. Write naturally as if you're explaining to a person face-to-face
5. Be helpful, clear, and thorough
{format_instruction}
CONTEXT FROM KNOWLEDGE BASE:
{context}
USER'S QUESTION: {query}
YOUR ANSWER (following the format rules above):"""
       
        if is_doc_context:
            base += "\n\nPriority: Extract the most important information and present it clearly."
       
        return base
   
    @staticmethod
    def get_multilingual_instruction(user_lang: str) -> str:
        """
        FIXED: STRICT language requirement for English/Swahili only.
       
        Args:
            user_lang: Detected user language code ('en' or 'sw')
        """
        lang_names = {
            'en': 'English',
            'sw': 'Swahili'
        }
       
        # CRITICAL: Force to English or Swahili
        if user_lang not in ['en', 'sw']:
            user_lang = 'en'
       
        language = lang_names.get(user_lang, 'English')
       
        return f"""
🔒 LANGUAGE REQUIREMENT:
- Respond ONLY in {language}
- Do NOT use any other language
- Do NOT mix languages
"""
   
    # ==================== CHATBOT PROMPTS ====================
   
    @staticmethod
    def get_greeting_detection_prompt(user_input: str) -> str:
        """
        Ultra-precise greeting detection - STRICT rules
       
        Args:
            user_input: The user's message
        """
        return f"""You are a STRICT greeting classifier. Your job is to detect if input is PURELY a greeting or contains a real question.
User input: "{user_input}"
CRITICAL RULES:
1. "how are you" / "how are you doing" = GREETING ONLY (not a real question)
2. "what's up" / "sup" / "wassup" = GREETING ONLY (not a real question)
3. Any input with ONLY greeting words = is_greeting: true, has_question: false
4. Any input asking about KNOWLEDGE/INFORMATION = has_question: true
GREETING-ONLY PHRASES (these are NOT questions):
- hi, hello, hey, howdy, greetings
- good morning, good afternoon, good evening, good night
- how are you, how are you doing, how's it going
- what's up, sup, wassup, yo
- nice to meet you, pleasure to meet you
REAL QUESTIONS (these need answers from knowledge base):
- what is..., how do I..., tell me about..., explain...
- questions about processes, steps, procedures, systems
- questions about specific topics or concepts
Return ONLY valid JSON:
{{"is_greeting": true/false, "has_question": true/false, "greeting_text": "greeting part", "user_name": "name if any", "question_part": "question if any"}}
EXAMPLES:
Input: "hello"
Output: {{"is_greeting": true, "has_question": false, "greeting_text": "hello", "user_name": "", "question_part": ""}}
Input: "hi how are you"
Output: {{"is_greeting": true, "has_question": false, "greeting_text": "hi how are you", "user_name": "", "question_part": ""}}
Input: "good morning"
Output: {{"is_greeting": true, "has_question": false, "greeting_text": "good morning", "user_name": "", "question_part": ""}}
Input: "hey there"
Output: {{"is_greeting": true, "has_question": false, "greeting_text": "hey there", "user_name": "", "question_part": ""}}
Input: "Hello, How are you"
Output: {{"is_greeting": true, "has_question": false, "greeting_text": "Hello, How are you", "user_name": "", "question_part": ""}}
Input: "what's up"
Output: {{"is_greeting": true, "has_question": false, "greeting_text": "what's up", "user_name": "", "question_part": ""}}
Input: "hello, what is the tender process?"
Output: {{"is_greeting": true, "has_question": true, "greeting_text": "hello", "user_name": "", "question_part": "what is the tender process?"}}
Input: "good morning, tell me about eGP"
Output: {{"is_greeting": true, "has_question": true, "greeting_text": "good morning", "user_name": "", "question_part": "tell me about eGP"}}
Input: "what is the tender process?"
Output: {{"is_greeting": false, "has_question": true, "greeting_text": "", "user_name": "", "question_part": "what is the tender process?"}}
Input: "how do I register?"
Output: {{"is_greeting": false, "has_question": true, "greeting_text": "", "user_name": "", "question_part": "how do I register?"}}
NOW ANALYZE: "{user_input}"
Return ONLY JSON (no explanation):"""
   
    @staticmethod
    def get_pure_greeting_response(user_input: str, user_name: str = "") -> str:
        """
        Generate pure, natural greeting responses like ChatGPT.
       
        Args:
            user_input: The user's greeting message
            user_name: User's name if detected
        """
        import datetime
        import random
       
        # Detect time of day
        hour = datetime.datetime.now().hour
        if hour < 12:
            time_of_day = "morning"
            time_emoji = "☀️"
        elif hour < 17:
            time_of_day = "afternoon"
            time_emoji = "🌤️"
        else:
            time_of_day = "evening"
            time_emoji = "🌙"
       
        # Normalize user input
        user_input_lower = user_input.lower().strip()
       
        # 🔥 SPECIAL HANDLING: "hello/hi/hey + how are you" combinations
        greeting_with_howareyou_patterns = [
            "hello how are you",
            "hi how are you",
            "hey how are you",
            "hello, how are you",
            "hi, how are you",
            "hey, how are you",
            "hello how r u",
            "hi how r u",
            "hey how r u",
        ]
       
        for pattern in greeting_with_howareyou_patterns:
            if pattern in user_input_lower:
                responses = [
                    "Hey{name}! 😊 I'm doing great — how about you? How's your day going so far?",
                    "Hi{name}! 😊 I'm doing really well, thanks for asking! How are you doing today?",
                    "Hello{name}! 😊 I'm functioning perfectly and ready to help! How's everything going with you?",
                    "Hey{name}! 😊 I'm doing fantastic! How about yourself? What brings you here today?",
                ]
                response = random.choice(responses)
                if user_name:
                    response = response.replace("{name}", f", {user_name}")
                else:
                    response = response.replace("{name}", "")
                return response
       
        # Define greeting patterns and responses
        greeting_responses = {
            # Simple greetings
            "hi": [
                "Hi{name}! 👋 How can I help you today?",
                "Hello{name}! 👋 What can I assist you with?",
                "Hey{name}! 👋 What would you like to know?",
            ],
            "hello": [
                "Hello{name}! 👋 I'm here to help. What can I do for you?",
                "Hi{name}! 👋 How can I assist you today?",
                "Hello{name}! 👋 What would you like to know about?",
            ],
            "hey": [
                "Hey{name}! 👋 What's up? How can I help?",
                "Hey there{name}! 👋 What can I do for you?",
                "Hi{name}! 👋 How can I assist you today?",
            ],
            "good morning": [
                f"Good morning{{name}}! {time_emoji} I hope you're having a great start to your day. How can I help you?",
                f"Good morning{{name}}! {time_emoji} Ready to assist you. What can I do for you today?",
                f"Good morning{{name}}! {time_emoji} How can I help you today?",
            ],
            "good afternoon": [
                f"Good afternoon{{name}}! {time_emoji} I hope your day is going well. What can I help you with?",
                f"Good afternoon{{name}}! {time_emoji} How can I assist you today?",
                f"Good afternoon{{name}}! {time_emoji} What would you like to know?",
            ],
            "good evening": [
                f"Good evening{{name}}! {time_emoji} How can I help you tonight?",
                f"Good evening{{name}}! {time_emoji} What can I assist you with?",
                f"Good evening{{name}}! {time_emoji} Ready to help. What do you need?",
            ],
            "good night": [
                f"Good night{{name}}! {time_emoji} Sleep well! If you need anything before you go, I'm here to help.",
                f"Good night{{name}}! {time_emoji} Sweet dreams! Feel free to ask if you need anything.",
                f"Good night{{name}}! {time_emoji} Rest well! Let me know if you need any help.",
            ],
            "how are you": [
                "I'm doing great{name}, thanks for asking! 😊 I'm here and ready to help you. What can I do for you?",
                "I'm functioning perfectly and ready to assist{name}! 😊 How can I help you today?",
                "I'm here and ready to help{name}! 😊 What would you like to know?",
                "I'm doing really well{name}! 😊 How about you? What can I help you with today?",
            ],
            "what's up": [
                "Not much{name}, just here ready to help you! 😊 What can I do for you?",
                "Hey{name}! 👋 I'm here to assist you. What do you need help with?",
                "All good here{name}! 😊 How can I help you today?",
            ],
            "howdy": [
                "Howdy{name}! 🤠 What can I help you with today?",
                "Howdy partner{name}! 🤠 How can I assist you?",
                "Hey there{name}! 👋 What do you need help with?",
            ],
        }
       
        # Check for specific greeting patterns
        for pattern, responses in greeting_responses.items():
            if pattern in user_input_lower:
                response = random.choice(responses)
                if user_name:
                    response = response.replace("{name}", f", {user_name}")
                else:
                    response = response.replace("{name}", "")
                return response
       
        # Default dynamic response based on time
        default_responses = [
            f"Good {time_of_day}{{name}}! {time_emoji} How can I help you today?",
            f"Hello{{name}}! {time_emoji} I hope you're having a wonderful {time_of_day}. What can I do for you?",
            f"Hi{{name}}! 👋 What can I help you with this {time_of_day}?",
        ]
       
        response = random.choice(default_responses)
       
        if user_name:
            response = response.replace("{name}", f", {user_name}")
        else:
            response = response.replace("{name}", "")
       
        return response
       
        # Default dynamic response based on time
        default_responses = {
            "morning": [
                "Good morning! ☀️ How can I help you today?",
                "Hello! ☀️ I hope you're having a wonderful morning. What can I do for you?",
            ],
            "afternoon": [
                "Good afternoon! 🌤️ How can I assist you?",
                "Hello! 🌤️ What can I help you with this afternoon?",
            ],
            "evening": [
                "Good evening! 🌙 How can I help you tonight?",
                "Hello! 🌙 What can I assist you with this evening?",
            ],
        }
       
        response = random.choice(default_responses.get(time_of_day, default_responses["afternoon"]))
       
        if user_name:
            parts = response.split("!", 1)
            if len(parts) == 2:
                response = f"{parts[0]}, {user_name}!{parts[1]}"
       
        return response
   
    @staticmethod
    def get_dynamic_greeting_response(user_name: str = "", time_of_day: str = "") -> str:
        """
        Generate dynamic greeting response (LEGACY - kept for compatibility).
        Use get_pure_greeting_response() for better results.
        """
        import datetime
       
        if not time_of_day:
            hour = datetime.datetime.now().hour
            if hour < 12:
                time_of_day = "morning"
            elif hour < 17:
                time_of_day = "afternoon"
            else:
                time_of_day = "evening"
       
        greetings = {
            "morning": "Good morning",
            "afternoon": "Good afternoon",
            "evening": "Good evening"
        }
       
        greeting = greetings.get(time_of_day, "Hello")
       
        if user_name:
            return f"{greeting}, {user_name}! 👋 I'm your AI assistant ready to help you with any questions."
        else:
            return f"{greeting}! 👋 I'm your AI assistant ready to help. What would you like to know?"
   
    @staticmethod
    def get_followup_generation_prompt(rag_answer: str) -> str:
        """Prompt for generating relevant follow-up questions."""
        return f"""Based on this answer:
"{rag_answer}"
Suggest ONE relevant follow-up question the user might ask next.
Requirements:
- Natural and conversational
- Directly related to the topic
- Concise (max 15 words)
- Starts with a question word (What, How, When, etc.)
Return ONLY the question, nothing else:"""
   
    @staticmethod
    def get_followup_confirmation_prompt(suggested_followup: str, user_reply: str) -> str:
        """Prompt for detecting if user accepts/rejects a follow-up suggestion."""
        return f"""Detect if user accepts or rejects this suggestion.
Suggested: "{suggested_followup}"
User said: "{user_reply}"
Return JSON only:
{{"decision": "accept"}} or {{"decision": "reject"}}
Accept: yes, sure, okay, tell me, go ahead, please, sounds good
Reject: no, not now, skip, later, different, cancel
"""
   
    # ==================== VISUAL GENERATION PROMPTS ====================
   
    @staticmethod
    def get_visual_request_detection_prompt(query: str) -> str:
        """Prompt for detecting if user wants a visual diagram."""
        return f"""Does this query request a visual representation?
Query: "{query}"
Visual keywords: diagram, flowchart, chart, graph, visualize, show, draw, illustrate, map, timeline
Return JSON only:
{{"wants_visual": true/false, "confidence": 0-100}}"""
   
    @staticmethod
    def get_step_extraction_instruction() -> str:
        """Instruction for extracting process steps from content."""
        return """Extract clear, sequential steps from this content.
Return a numbered list of actionable steps."""
   
    @staticmethod
    def get_complexity_analysis_prompt(content: str, step_count: int) -> str:
        """Prompt for analyzing process complexity."""
        return f"""Analyze process complexity:
Content: {content}
Steps: {step_count}
Criteria:
- Simple: 1-3 steps, straightforward
- Medium: 4-7 steps, some decisions
    - Complex: 8+ steps, multiple branches
Return JSON only:
{{"complexity": "simple/medium/complex", "reasoning": "brief reason"}}"""
   
    # ==================== OTHER PROMPTS ====================
   
    @staticmethod
    def get_ocr_postprocessing_prompt(raw_ocr_text: str) -> str:
        """Prompt for cleaning OCR output."""
        return f"""Clean this OCR text by:
1. Fixing spacing issues
2. Correcting obvious OCR errors
3. Preserving structure
Text: {raw_ocr_text}
Return cleaned text only:"""
   
    @staticmethod
    def get_document_summarization_prompt(content: str, doc_type: str = "document") -> str:
        """Prompt for summarizing documents."""
        return f"""Summarize this {doc_type} concisely:
{content}
Provide:
1. Main topic
2. Key points (3-5 bullets)
3. Important details"""
   
    @staticmethod
    def get_keyword_extraction_prompt(text: str, num_keywords: int = 10) -> str:
        """Prompt for extracting keywords."""
        return f"""Extract {num_keywords} important keywords from:
{text}
Return JSON only:
{{"keywords": ["keyword1", "keyword2", ...]}}"""
   
    @staticmethod
    def get_language_detection_prompt(text: str) -> str:
        """Prompt for detecting text language."""
        return f"""Detect language of: "{text}"
Return JSON only:
{{"language": "en/sw", "confidence": 0-100}}"""
   
    @staticmethod
    def get_error_explanation_prompt(error_type: str, context: str = "") -> str:
        """Generate user-friendly error explanations."""
        explanations = {
            "no_context": "I couldn't find relevant information in the documents to answer your question.",
            "api_error": "I encountered a technical issue. Please try again.",
            "translation_error": "I had trouble processing your message. Could you rephrase it?",
        }
        return explanations.get(error_type, "An error occurred.") + (f" {context}" if context else "")
   
    @staticmethod
    def get_transcription_correction_prompt(raw_transcription: str, domain_keywords: list) -> str:
        """Correct speech recognition errors."""
        keywords_str = ", ".join(domain_keywords[:20])
        return f"""Correct speech recognition errors in: "{raw_transcription}"
Valid domain terms: {keywords_str}
Return corrected text only:"""
   
    # ==================== UTILITY METHODS ====================
   
    @staticmethod
    def format_with_context(base_prompt: str, **kwargs) -> str:
        """Format prompt template."""
        try:
            return base_prompt.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing prompt variable: {e}")
   
    @staticmethod
    def get_token_limit_prompt(max_tokens: int = 4096) -> str:
        """Limit response length."""
        return f"\n\nKeep response under {max_tokens} tokens."
   
    @staticmethod
    def get_json_format_instruction() -> str:
        """Ensure JSON responses."""
        return "\n\nCRITICAL: Return ONLY valid JSON, no other text."
# ==================== PROMPT VALIDATION ====================
class PromptValidator:
    """Validates prompts."""
   
    @staticmethod
    def validate_prompt(prompt: str, max_length: int = 15000) -> bool:
        """Validate prompt."""
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if len(prompt) > max_length:
            print(f"⚠️ Warning: Prompt length ({len(prompt)}) exceeds recommended limit")
        return True
   
    @staticmethod
    def sanitize_user_input(text: str) -> str:
        """Sanitize user input."""
        dangerous_patterns = [
            "ignore previous instructions",
            "ignore all previous",
            "disregard all",
            "forget everything",
        ]
        text_lower = text.lower()
        for pattern in dangerous_patterns:
            if pattern in text_lower:
                text = text.replace(pattern, "[removed]")
                print(f"⚠️ Security: Removed dangerous pattern from input")
        return text.strip()
# ==================== EXPORT ====================
__all__ = ['SystemPrompts', 'PromptValidator']
