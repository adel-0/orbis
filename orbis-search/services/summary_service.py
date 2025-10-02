from openai import AzureOpenAI
from typing import List, Optional
import logging
from config import settings
from models.schemas import Ticket

logger = logging.getLogger(__name__)

class SummaryService:
    def __init__(self):
        self.client: Optional[AzureOpenAI] = None
        self.deployment_name = settings.AZURE_OPENAI_DEPLOYMENT_NAME
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Azure OpenAI client"""
        try:
            if not settings.AZURE_OPENAI_ENDPOINT or not settings.AZURE_OPENAI_API_KEY:
                logger.warning("Azure OpenAI credentials not configured. Summary service will be disabled.")
                return
            
            self.client = AzureOpenAI(
                azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_API_KEY,
                api_version=settings.AZURE_OPENAI_API_VERSION
            )
            
            logger.info(f"Azure OpenAI client initialized with deployment: {self.deployment_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {e}")
            self.client = None
    
    def generate_summary(self, query: str, tickets: List[Ticket], similarity_scores: Optional[List[float]] = None) -> Optional[str]:
        """Generate a summary of the search results using Azure OpenAI.

        The summary tone and length are calibrated based on overall relevancy derived
        from the provided similarity scores. When high-relevancy tickets are present,
        the model is instructed to respond with higher confidence; with only low
        relevancy, it should be more cautious and concise.
        """
        if not self.client or not tickets:
            return None
        
        try:
            # Prepare context for the summary
            context_parts = [f"Query: {query}\n\nRelevant tickets:"]
            
            for i, ticket in enumerate(tickets, 1):
                ticket_text = f"{i}. Ticket {ticket.id}: {ticket.title}"
                if ticket.description:
                    ticket_text += f"\n   Description: {ticket.description}"
                if ticket.comments:
                    ticket_text += f"\n   Comments: {'; '.join(ticket.comments[:3])}"  # Limit to first 3 comments
                # Add per-ticket relevancy when available
                if similarity_scores and i-1 < len(similarity_scores):
                    try:
                        score = float(similarity_scores[i-1])
                        ticket_text += f"\n   Relevancy: {score:.2f}"
                    except Exception:
                        pass
                context_parts.append(ticket_text)
            
            context = "\n\n".join(context_parts)
            
            # Determine overall relevancy calibration
            max_score = None
            avg_score = None
            confidence_note = ""
            if similarity_scores:
                try:
                    filtered = [float(s) for s in similarity_scores if s is not None]
                    if filtered:
                        max_score = max(filtered)
                        avg_score = sum(filtered) / len(filtered)
                        # Calibrate tone/length based on max relevancy (adjusted for 0.4+ filtered results)
                        if max_score >= 0.7:
                            confidence = "high"
                            confidence_note = (
                                "Be confident and decisive. Provide a clear recommendation and the most likely fix. "
                                "Assume the top tickets closely match the query."
                            )
                        elif max_score >= 0.55:
                            confidence = "medium"
                            confidence_note = (
                                "Be balanced. Indicate reasonable confidence but note where validation is needed. "
                                "Prefer probable fixes and mention alternatives briefly."
                            )
                        else:
                            confidence = "low"
                            confidence_note = (
                                "Be cautious and brief. State uncertainty and suggest concrete next steps to gather more context."
                            )
                    
                        relevancy_summary = f"Overall relevancy: {confidence} (max={max_score:.2f}, avg={avg_score:.2f})."
                    else:
                        relevancy_summary = "Overall relevancy: unknown."
                except Exception:
                    relevancy_summary = "Overall relevancy: unknown."
            else:
                relevancy_summary = "Overall relevancy: unknown."

            

            # Create the prompt
            prompt = f"""Query: {query}

{context}

Based on the above tickets, provide a brief summary addressing the query. Focus on actionable insights.

Calibration: {relevancy_summary} {confidence_note}"""
            
            # Generate summary
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are an operation and project engineer assistant. Provide concise, actionable summaries for operations and project engineers (not for developers of the tool). Use bullet points for multiple items. Focus on: 1) Direct answer to the query 2) Implementation steps or operational procedures. Be brief and practical."},
                    {"role": "user", "content": prompt}
                ],
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info("Summary generated successfully")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return None
    
    def is_configured(self) -> bool:
        """Check if Azure OpenAI is properly configured"""
        has_client = self.client is not None
        has_endpoint = bool(settings.AZURE_OPENAI_ENDPOINT)
        has_api_key = bool(settings.AZURE_OPENAI_API_KEY)
        return has_client and has_endpoint and has_api_key