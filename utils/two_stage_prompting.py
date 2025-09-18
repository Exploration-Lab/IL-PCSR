from typing import List
from pydantic import BaseModel, Field
#from langchain.pydantic_v1 import BaseModel, Field
from langchain.prompts import HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
import tiktoken
import re

##################################################################################CATEGRORY PROMPT ###########
class CategoryResponse(BaseModel):
    id : int = Field(description="ID of the category")

category_parser = PydanticOutputParser(pydantic_object=CategoryResponse)

category_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                """
You are a smart and diligent Indian legal assistant with expertise in analyzing legal douments.

**Context:**

You will be provided with:
- A legal document, which may be a court judgment, statute


**Your Task:**

1. **Analyze the legal document:**
   - Carefully understand the legal document.

2. **Identify to which categrory the document Belongs:**
   categories = [
    {"id": 1, "category": "Labour & Employment"},
    {"id": 2, "category": "Criminal"},
    {"id": 3, "category": "Income Tax"},
    {"id": 4, "category": "Motor Vehicle Accidents"},
    {"id": 5, "category": "Family & Marriage"},
    {"id": 6, "category": "Property & Land Disputes"},
    {"id": 7, "category": "Contract & Commercial"},
    {"id": 8, "category": "Constitutional"},
    {"id": 9, "category": "Intellectual Property Rights"},
    {"id": 10, "category": "Consumer Protection"},
    {"id": 11, "category": "Environmental"},
    {"id": 12, "category": "Company & Corporate"},
    {"id": 13, "category": "Service Matters"}
]

**Output Format:**

Return your answer in the following format:
""" + category_parser.get_format_instructions()
            )
        ),
        HumanMessagePromptTemplate.from_template(
            "Legal Document:\n{document}"
        ),
    ]
)


##############################################################################
class Score(BaseModel):
    id: str = Field(description="ID of the citation")
    score: float = Field(description="Score of the citation")

class ScoringResponse(BaseModel):
    #scores: List[Score] = Field(description="List of Scores")
    score : float = Field(description="Score") 

scoring_parser = PydanticOutputParser(pydantic_object=ScoringResponse)
##############################################################################

class StatuteListResponse(BaseModel):
    statutes: List[str] = Field(description="List of Statute Section IDs cited in the judgment")

classification_parser = PydanticOutputParser(pydantic_object=StatuteListResponse)

classification_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                """
You are a smart and diligent Indian legal assistant with expertise in analyzing court judgments and identifying cited statutory provisions.

**Context:**

You will be provided with:
- A **masked judgment**: All statute names and section numbers have been replaced with placeholders like [ACT], [SECTION], etc.
- A **list of statutory sections**: Each entry includes a unique ID, and the name of the corresponding statute.

**Your Task:**

1. **Analyze the Masked Judgment:**
   - Carefully interpret the masked judgment's legal reasoning and context.

2. **Understand the Statutory List:**
   - Review the provided statute list. 

3. **Determine Cited Statutes:**
   - Identify the statute sections that are cited  in the judgment.
   - Return only the relevant **IDs** of the cited statute sections.

**Output Format:**

Return your answer in the following format:
""" + classification_parser.get_format_instructions()
            )
        ),
        HumanMessagePromptTemplate.from_template(
            "MASKED JUDGMENT:\n{query}\n\nSTATUTE LIST:\n{statute}"
        ),
    ]
)


individual_scoring_chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                """
You are a smart and diligent Indian legal assistant. Your task is to determine whether a given masked judgment refers to a specific statutory section.

**Context:**

You will be provided with:
- A masked judgment: Statutory names and section numbers are replaced with placeholders such as [ACT], [SECTION], etc.
- The text of a specific statutory section.

**Your Task:**

1. **Analyze the Masked Judgment:**
   - Carefully read and interpret the judgment.
   - Identify legal issues, reasoning, and contextual clues that may indicate reference to statutory provisions, even if explicit names or numbers are masked.

2. **Review the Provided Section:**
   - Understand the content, scope, and purpose of the given statutory section.

3. **Determine Citation:**
   - Assess whether the masked judgment refers to or relies upon the provided section, based on context, legal reasoning, and any implicit references.

4. **Scoring:**
   - Assign a score of `1` if the judgment cites or refers to the given section (explicitly or implicitly).
   - Assign a score of `0` if the judgment does not cite or refer to the given section.

Output Format:
""" +scoring_parser.get_format_instructions()
                
            )
        ),
        HumanMessagePromptTemplate.from_template("QUERY JUDGEMENT:{query} \n\n STATUTE:{statute}"),
    ]
)

scoring_chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                """
                You are a highly capable Indian legal assistant with expertise in legal document analysis and statutory interpretation. Your task is to determine the likely relevance of statutory provisions to masked Indian legal judgments.

Context:

You will be provided with:
	•	A summary of query judgment, where statutory names and numbers are removed or replaced with placeholders like [ACT], [SECTION], etc.
	•	A list of 10 statutes, each accompanied by:
	•	A unique Statute ID
	•	The opening portion of the statute text, providing contextual information about its content.

Your Task:
	1.	Analyze the Masked Judgment:
	•	Carefully read and interpret the judgment.
	•	Identify the legal issues, reasoning, and contextual clues that imply reference to statutory provisions, even if they are masked.
	2.	Review the Statutes:
	•	For each of the 10 statutes provided, examine the beginning portion of the text.
	•	Determine whether the statute conceptually aligns with the legal context or issues raised in the judgment.
	3.	Score Each Statute:
	•	Assign a score of 1 if the statute is very likely referenced, applied, or forms part of the reasoning in the judgment.
	•	Assign a score of 0 if there is no clear alignment, or the relevance is uncertain, speculative, or ambiguous.

Output Format:
""" +scoring_parser.get_format_instructions()
                
            )
        ),
        HumanMessagePromptTemplate.from_template("QUERY JUDGEMENT:{query} \n\n STATUTE:{statute}"),
    ]
)


    
    
