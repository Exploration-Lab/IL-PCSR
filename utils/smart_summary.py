from typing import List
from pydantic import BaseModel, Field
#from langchain.pydantic_v1 import BaseModel, Field
from langchain.prompts import HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.output_parsers.boolean import BooleanOutputParser
import tiktoken
import re





bool_parser = BooleanOutputParser()

only_precs_chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                """
You are a indian legal AI assistant.

Your task is to decide whether the given precedent judgment is cited by the masked query judgment. The query text may include masked placeholders like [PRECEDENT], [ACT], or [SECTION].

Return YES if the query judgment cites the precedent judgment, otherwise return NO. 

Output Format:
'Respond with exactly one word: "YES" or "NO". '
'Do not include any explanation, punctuation, or additional text.'
"""
            )
        ),
        HumanMessagePromptTemplate.from_template(
            "MASKED QUERY JUDGEMENT:\n{query_txt} \n\n PRECEDENT JUDGEMENT:\n{precs_txt}"
            ),   
    ]
)

precs_secs_chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                """
You are a indian legal AI assistant.

Your task is to decide whether the given precedent judgment is cited by the masked query judgment. The query text may include masked placeholders like [PRECEDENT], [ACT], or [SECTION]. To assist your decision, statute cited in both the query and precedent are provided.

Return YES if the query judgment cites the precedent judgment, otherwise return NO. 

Output Format:
'Respond with exactly one word: "YES" or "NO". '
'Do not include any explanation, punctuation, or additional text.'
"""
             
            )
        ),
       HumanMessagePromptTemplate.from_template(
            "MASKED QUERY JUDGEMENT:\n{query_txt} \n QUERY STATUTES:{query_secs} \n\n PRECEDENT JUDGEMENT:\n{precs_txt} \n PRECEDENT STATUTES:{precs_secs}"
            ),   
    ]
)






# Let judgment J1 be cited by a set C= {J2, J3,. . . , Jn}of judgments. The title of J1 appears
# in each judgment of C. We propose that the set of text units, where each text unit cor-
# responds to the text surrounding the title of J1 that exists in each judgment of C, which
# we refer to as Preceding citation Anchor Text (PAT), could be utilized to improve the
# representation of J1. We explain the concept of PAT after explaining the concepts of Case
# Citation and Citation Anchor Text (CAT).

class Precedent(BaseModel):
    name: str = Field(description="Name")
    text: str = Field(description="Text")

class LLMResponse(BaseModel):
    precedents: List[Precedent] = Field(description="List of Precedents")

class ScoringResponse(BaseModel):
    score : float = Field(description="Score") 

llm_parser = PydanticOutputParser(pydantic_object=LLMResponse)
scoring_parser = PydanticOutputParser(pydantic_object=ScoringResponse)

iSPATU_chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                """
You are an intelligent Indian legal assistant with expertise in legal document analysis and precedent identification.

Objective:
Carefully analyze the given Indian legal judgment and extract all content related to cited precedents.

Instructions:
	1.	Read and comprehend the full judgment thoroughly.
	2.	Identify every precedent cited in the document.
	    •	Some citations may appear in anonymized form using the placeholder [PRECEDENT].
	3.	For each cited precedent extract:
        •	All paragraphs where the precedent is directly cited or discussed.
        •	All quoted material from the precedent
	4. Be expansive in your extraction.   
        •	Do not limit to just the citation sentence or immediate paragraph.
	    •	Include all logically connected paragraphs, even if they are non-adjacent.
        •   Multiple precedents may be discussed together in a single paragraph due to shared legal interpretations.
        •   Such paragraphs or quoted material should be included only once, even if they apply to multiple precedents.
     
Note: Due to the large size of the document, you will not receive the full text at once.
• You will be given small chunks of the document, with some overlapping content between them.
• Along with each chunk (except the first), you will also receive the cumulative output generated from all previous chunks.
• For the first chunk, no prior output will be provided.
• Perfrom the Instructions on the current chunk and append the output to the cumulative output.
• Ensure that the output is coherent and logically connected to the previous output.


Output Format:
"""
                + llm_parser.get_format_instructions()
            )
        ),
        HumanMessagePromptTemplate.from_template("Query Judgment Chunk:\n{document}  \n\n Cumulative Output:\n{previous_output}"),   
    ]
)

scoring_chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                """
Role: You are a legal intelligence model specializing in Indian case law and precedent analysis.

Objective:
Given a citation passage from a judgment and a full precedent document from a known precedent pool, determine whether the citation directly refers to or is derived from the given precedent.

Instructions:
	1.	Analyze Both Inputs:
	2.	Focus on Authoritative Sections:
        •   Determine whether the citation aligns directly with the court’s conclusions in the precedent.
	3.	Exclude Indirect References:
	    •	If the given precedent itself cites another precedent for the cited proposition, do not treat it as a direct match.
	4.	Scoring Criteria:
	    •	Score 1: The citation clearly and directly refers to a conclusion or holding of the given precedent.
	    •	Score 0: There is no clear direct reference, or the reference is indirect, ambiguous, or based on the precedent’s citation of other cases.
    

Output Format: 
""" +scoring_parser.get_format_instructions()
                
            )
        ),
        HumanMessagePromptTemplate.from_template("CITATION NAME:{cite_name} \n CIATTION TEXT:{cite_text}  \n\n PRECEDENT JUDGEMENT:\n{prec_text}"),
    ]
)




# class LegalReason(BaseModel):
#     title:str = Field(description="Title of the legal reason")
#     #precedents_names:List[str] = Field(description="List of precedents names which are cited for the legal reason")
#     reasons:List[str] = Field(description="Reasons for which the precedents are cited")
    
# class Response(BaseModel):
#     legal_reasons : List[LegalReason] = Field(description="List of legal reasons")
    
# class KeyPoint(BaseModel):
#     title:str = Field(description="Title of the key point")
#     paragraphs:List[str] = Field(description="paragraphs of the key point")

# class Reponse1(BaseModel):
#     key_points : List[KeyPoint] = Field(description="List of key points")

# parser = PydanticOutputParser(pydantic_object=Response)
# parser1 = PydanticOutputParser(pydantic_object=Reponse1)


# query_chat_template = ChatPromptTemplate.from_messages(
#     [
#         SystemMessage(
#             content=(
#                     """
#                     Extract reasons from a legal judgment (query) explaining why the judge cited specific precedents
#                     , to later match these reasons with findings from the cited precedents for retrieval tasks.
#                     Please process the given legal judgment and focus on the following instructions:

#                     Objective: Identify and extract all the legal reasons cited in the given judgment, 
#                     focusing on the legal principles, rules, or questions of law discussed or evaluated. 
#                     Exclude any specific factual context or case-specific details.

#                     Structure: Each reason should be phrased in a concise and neutral manner.
#                             Avoid including case-specific details (e.g., names, dates, or specific statutes cited).
#                             Ensure the reasons are comprehensive enough to match with similar principles from other precedents.

#                     Focus Areas: While extracting reasons, focus only the places where the precedents and cited text is present.
#                     """
#                     +parser.get_format_instructions()
                
#             )
#         ),
#         HumanMessagePromptTemplate.from_template("Query Judgement:{document}"),
#     ]
# )

# precedent_chat_template = ChatPromptTemplate.from_messages(
#     [
#         SystemMessage(
#             content=(
#                     """
#                     Summarize the key points from a provided case document that contributed to the final judgment.
#                     These summaries will later be used to identify the reasons why this case might be cited as a precedent.
#                     Please process the given legal precedent and focus on the following instructions:

#                     Objective:  Identify and extract the key legal findings, principles, or rules established in this precedent that could serve as the basis for its citation in other judgments.

#                     Structure: Each key points should be phrased in a concise and neutral manner.
#                                Avoid including case-specific details (e.g., names, dates, or specific statutes cited).
#                              Ensure the summaries comprehensively capture the reasons, enabling effective matching with those from the queries.

#                     Focus Areas: Prioritize the sections where legal principles are established, clarified, or interpreted, 
#                     focusing on the parts likely to be cited as precedents.
#                     """
#                     +parser1.get_format_instructions()
                
#             )
#         ),
#         HumanMessagePromptTemplate.from_template("Case Document:{document}"),
#     ]
# )


# class Incidents(BaseModel):
#     title:str = Field(description="Title of the Incidents")
#     paragraphs:List[str] = Field(description="paragraphs of the Incidents")

# class Response3(BaseModel):
#     incidents : List[Incidents] = Field(description="List of legal incidents")

# parser3 = PydanticOutputParser(pydantic_object=Response3)

# section_chat_template = ChatPromptTemplate.from_messages(
#     [
#         SystemMessage(
#             content=(
#                     """
#                     Extract legal incidents from a given judgment to understand why specific sections or articles of law
#                     were cited. These extracted incidents will later be matched with relevant sections and articles.

#                     Please process the given legal judgment and focus on the following instructions:

#                     Objective: Identify and extract all legal incidents referenced in the judgment, 
#                     focusing on the key facts and legal issues of the case.

#                     Structure: Phrase each incident concisely and neutrally.
#                     Exclude case-specific details (e.g., names, dates, case numbers).
#                     The extracted incidents should be rich in legal reasoning and sufficiently descriptive to enable accurate section/article matching.

#                     Focus Areas: Capture the core facts and issues underlying the case.
#                     """
#                     +parser3.get_format_instructions()
                
#             )
#         ),
#         HumanMessagePromptTemplate.from_template("Case Document:{document}"),
#     ]
# )

# def clean_string(input_string):
#     # Remove new lines, multiple spaces, and tabs
#     cleaned_string = re.sub(r'\s+', ' ', input_string.replace('\n', ' ').strip())
#     return cleaned_string