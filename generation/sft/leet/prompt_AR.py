## 추리논증 - 법규범 ## 
PROMPT_LEGAL_REASONING = """
You are solving a Korean-language multiple-choice inference question based on a passage from either the legal provision or norm. The input will be in Korean, and your output must also be written in Korean. Follow the step-by-step reasoning process below and output your final response in JSON format.

Step 1: Analyze the structure of the Legal privisions (norm) in the Passage 
- Classify each clause in a hierarchical manner, distinguishing principle (main rule), exception, proviso, and supplementary provisions.
- Identify the legal requirements (premises) and the legal effects (consequences) set out in each clause.
- A legal effect may impose a duty (mandatory rule), confer a right (discretionary rule), or restrict an existing right.
- Unless an existing right or duty is expressly presumed, begin by assuming no right or duty exists.
- Express causal link in a step-wise: “If condition A is satisfied → legal effect B ensues.”

Step 2: Analyze the case (facts)
- Identify the facts, timeline, actions, and parties stated in the case.
- Determine whether each fact satisfies, fails to satisfy, or leaves unclear the conditions identified in Step 1.
- Check whether any exceptions or provisos apply, follow the sequence (principle → exception → proviso), and structure the reasoning to show which legal effects arise or are excluded.

Step 3: Identify the Question Format and Task Type
- Clarify what the question is asking:
    ☐ Is it asking for all correct statements?
    ☐ Is it asking for the incorrect option?
    ☐ Is it asking for a single best choice?

Identify whether the question is about truth/falsity, logical possibility, or appropriateness.

Step 4: Evaluate the Truth of Each Option
- Identify the core claim in each choice.
- Determine whether the premise implied by the choice is explicitly stated, or whether it is the most reasonable inference based on the legal rule.
- For each option, make a clear judgment of whether it is True or False, and must consult the *[Reference] when making the judge*.
- Write in the style of an explanation, using a descriptive and logical tone that compares each option’s statement based on the key content identified in Step 1. Use declarative sentences ending with '-입니다.'

[Reference]
- If the sentence is a conditional statement("A then B"), evaluate the truth value of both the antecedent and the consequent to determine the overall truth of the statement. If A is not true, then B cannot be guaranteed.
- A conjunction ("A and B") is true only when both conditions are true.
- A disjunction ("A or B") is true if at least one of the conditions is true.
- A definition ("A means B") represents a necessary and sufficient condition, so it is true only if both "A implies B" and "B implies A" hold.

Clearly state whether each option is True or False.

Step 5: Determine the Final Answer
- Choose the correct answer based on the question type.
- Briefly explain why that answer is correct.

✅ Output Format (JSON)
Your final output must follow this exact JSON structure:
```json
{{
  "step1": "Write your explanation for Step 1 here in Korean.",
  "step2": "Write your explanation for Step 2 here in Korean.",
  "step3": "Write your explanation for Step 3 here in Korean.",
  "step4": "Write your explanation for Step 4 here in Korean.",
  "step5": "Write your explanation for Step 5 here in Korean.",
  "Final answer": "Write only the final answer here.\n- If the choices are in the form of ㄱ, ㄴ, ㄷ, etc., list only the correct ones separated by commas and spaces (e.g., 'ㄱ, ㄷ'). Multiple correct options are allowed, but you must select at least one.\n- If the choices are numbered (e.g., 1 to 5), write only a single correct number as an integer (e.g., '3'). Absolutely do NOT write multiple numbers like '①, ④'. Only one number must be selected.\n- If you are unsure, choose the most reasonable or likely correct answer based on the passage. Never leave the final answer blank."
}}

The input question is as follows:
<Question>
{input_question}
"""

## 추리논증 - 인문 ## 
PROMPT_HUMAN_REASONING = """
You are solving a Korean-language multiple-choice inference question based on a passage from the humanities and philosophy. The input will be in Korean, and your output must also be written in Korean. Follow the step-by-step reasoning process below and output your final response in JSON format.

Step 1: Understand the Key Concepts in the Passage
- Identify how key conceptual terms are defined in the passage, and internalize them by rephrasing in your own words.
- Structure abstract principles or theories, and logically categorize related concepts to explain the overall mechanism in simple terms.

Step 2: Analyzing Correspondence and Causal Relationships
- Identify how the expressions presented in the given case or choices correspond to the principles structured in Step 1.
- Determine whether the sentence in the choice is a conditional statement, a definition, or a conjunction/disjunction, and interpret it accordingly."
- Structure your reasoning as if you were visualizing the relationships.

Step 3: Identify the Question Format and Task Type
- Clarify what the question is asking:
    ☐ Is it asking for all correct statements?
    ☐ Is it asking for the incorrect option?
    ☐ Is it asking for a single best choice?

Identify whether the question is about truth/falsity, logical possibility, or appropriateness.

Step 4: Evaluate the Truth of Each Option
- For each option, make a clear judgment of whether it is True or False, and write an explanation based on the key content identified in Step 1.
- Write in the style of an explanation, using a descriptive and logical tone that compares each option’s statement with the corresponding content in the passage. Use declarative sentences ending with '-입니다.'

[Reference]
- If the sentence is a conditional statement("A then B"), evaluate the truth value of both the antecedent and the consequent to determine the overall truth of the statement. If A is not true, then B cannot be guaranteed.
- A conjunction ("A and B") is true only when both conditions are true.
- A disjunction ("A or B") is true if at least one of the conditions is true.
- A definition ("A means B") represents a necessary and sufficient condition, so it is true only if both "A implies B" and "B implies A" hold.
- Determine whether it aligns with the rules, logic, or examples from the passage.

Clearly state whether each option is True or False.

Step 5: Determine the Final Answer
- Choose the correct answer based on the question type.
- Briefly explain why that answer is correct.

✅ Output Format (JSON)
Your final output must follow this exact JSON structure:
```json
{{
  "step1": "Write your explanation for Step 1 here in Korean.",
  "step2": "Write your explanation for Step 2 here in Korean.",
  "step3": "Write your explanation for Step 3 here in Korean.",
  "step4": "Write your explanation for Step 4 here in Korean.",
  "step5": "Write your explanation for Step 5 here in Korean.",
  "Final answer": "Write only the final answer here.\n- If the choices are in the form of ㄱ, ㄴ, ㄷ, etc., list only the correct ones separated by commas and spaces (e.g., 'ㄱ, ㄷ'). 
                    Multiple correct options are allowed, but you must select at least one.\n- If the choices are numbered (e.g., 1 to 5), write **only a single correct number** as an integer (e.g., '3'). Absolutely do NOT write multiple numbers like '①, ④'. Only one number must be selected.\n- If you are unsure, choose the most reasonable or likely correct answer based on the passage. Never leave the final answer blank."
}}

The input question is as follows:
<Question>
{input_question}
"""

## 언어이해 - 경제/정치 (사회)) ## 
PROMPT_SOCIAL_REASONING = """
You are solving a Korean-language multiple-choice inference question based on a passage from the economy & political science. The input will be in Korean, and your output must also be written in Korean. Follow the step-by-step reasoning process below and output your final response in JSON format.

Step 1: Understand Key Concepts in the Passage
- Identify how economic or political concepts are defined in the introduction of the passage, and rephrase them in your own words to internalize the ideas.

Step 2: Recognize the Structure of the Passage and Analyze Economic/Political Phenomena Based on Its Perspective
- Determine whether the passage presents multiple viewpoints or develops a single unified theory.
- Then, analyze the economic or political phenomena using the new framework or perspective introduced in the passage.

Step 3: Identify Components That Can Be Modeled
- If the passage presents aspects of an economic or political system that can be modeled, organize and express them in a structured or schematic form.

Step 4: Identify the Question Format and Task Type
- Clarify what the question is asking:
    ☐ Is it asking for all correct statements?
    ☐ Is it asking for the incorrect option?
    ☐ Is it asking for a single best choice?

Identify whether the question is about truth/falsity, logical possibility, or appropriateness.

Step 5: Evaluate the Truth of Each Option
- Compare whether the given content aligns with the principles of the passage.
- Write in the style of an explanation, using a descriptive and logical tone that compares each option’s statement with the corresponding content in the passage. Use declarative sentences ending with '-입니다.'

Clearly state whether each option is True or False.

Step 6: Determine the Final Answer
- Choose the correct answer based on the question type.
- Briefly explain why that answer is correct.

✅ Output Format (JSON)
Your final output must follow this exact JSON structure:
```json
{{
  "step1": "Write your explanation for Step 1 here in Korean.",
  "step2": "Write your explanation for Step 2 here in Korean.",
  "step3": "Write your explanation for Step 3 here in Korean.",
  "step4": "Write your explanation for Step 4 here in Korean.",
  "step5": "Write your explanation for Step 5 here in Korean.",
  "Final answer": "Write only the final answer here.\n- If the choices are in the form of ㄱ, ㄴ, ㄷ, etc., list only the correct ones separated by commas and spaces (e.g., 'ㄱ, ㄷ'). 
                    Multiple correct options are allowed, but you must select at least one.\n- If the choices are numbered (e.g., 1 to 5), write **only a single correct number** as an integer (e.g., '3'). Absolutely do NOT write multiple numbers like '①, ④'. Only one number must be selected.\n- If you are unsure, choose the most reasonable or likely correct answer based on the passage. Never leave the final answer blank."
}}


The input question is as follows:
<Question>
{input_question}
"""

## 추리논증 - 과학기술 ## 
PROMPT_SCIENCE_REASONING = """
You are solving a Korean-language multiple-choice inference question based on a passage from the science & tech criticism. The input will be in Korean, and your output must also be written in Korean. Follow the step-by-step reasoning process below and output your final response in JSON format.

Step 1: Understand the Principle or Mechanism Described in the Passage
- Grasp how a simple model or rule functions and identify how rules and variables interact with each other.
- Briefly explain the core mechanism or general principle presented in the passage.

Step 2: Analyze Causal Relationships and Interactions Between Variables
- Determine what hypothesis the experiment aims to test, how the experiment was designed to verify it, and whether the results support the hypothesis.
- Organize the flow of concepts as if you're creating a visual structure.
- Identify and differentiate between the manipulated variable (independent variable), the measured outcome (dependent variable), and the controlled variables (control variables) kept constant during the experiment.

Step 3: Identify the Question Format and Task Type
- Clarify what the question is asking:
    ☐ Is it asking for all correct statements?
    ☐ Is it asking for the incorrect option?
    ☐ Is it asking for a single best choice?

Identify whether the question is about truth/falsity, logical possibility, or appropriateness.

Step 4: Evaluate the Truth of Each Option
- For each option, make a clear judgment of whether it is True or False, and write an explanation based on the key content identified in Step 1, 2.
- Write in the style of an explanation, using a descriptive and logical tone that compares each option’s statement with the corresponding content in the passage. Use declarative sentences ending with '-입니다.'

Clearly state whether each option is True or False.

Step 5: Determine the Final Answer
- Choose the correct answer based on the question type.
- Briefly explain why that answer is correct.

✅ Output Format (JSON)
Your final output must follow this exact JSON structure:
```json
{{
  "step1": "Write your explanation for Step 1 here in Korean.",
  "step2": "Write your explanation for Step 2 here in Korean.",
  "step3": "Write your explanation for Step 3 here in Korean.",
  "step4": "Write your explanation for Step 4 here in Korean.",
  "step5": "Write your explanation for Step 5 here in Korean.",
  "Final answer": "Write only the final answer here.\n- If the choices are in the form of ㄱ, ㄴ, ㄷ, etc., list only the correct ones separated by commas and spaces (e.g., 'ㄱ, ㄷ'). 
                    Multiple correct options are allowed, but you must select at least one.\n- If the choices are numbered (e.g., 1 to 5), write **only a single correct number** as an integer (e.g., '3'). Absolutely do NOT write multiple numbers like '①, ④'. Only one number must be selected.\n- If you are unsure, choose the most reasonable or likely correct answer based on the passage. Never leave the final answer blank."
}}


The input question is as follows:
<Question>
{input_question}
"""

## 추리논증 - 논증 평가/문제 해결 ## 
PROMPT_SOLVING_REASONING = """
You are solving a Korean-language multiple-choice inference question based on a passage from the argument. The input will be in Korean, and your output must also be written in Korean. Follow the step-by-step reasoning process below and output your final response in JSON format.

Step 1: Identify the Author’s Main Argument
- Determine whether the focus is on strengthening or weakening the content, the reasoning, or the experiment.
- In the case of content or reasoning, clearly distinguish between the author’s main claim (conclusion) and the supporting reasons (premises).
- In the case of experiments, read the experimental results first, and then, if necessary, refer to the experimental background, which contains the principles needed to interpret the experiment.

Step 2: Analyze the <보기> example
- When an example different from the one in the passage is presented and a judgment is made, identify the similarities and differences between the passage's case and the <보기> example.
- Then assess whether the judgment aligns with the passage or acts as a counterexample.

Step 3: Identify the Question Format and Task Type
- Clarify what the question is asking:
    ☐ Is it asking for all correct statements?
    ☐ Is it asking for the incorrect option?
    ☐ Is it asking for a single best choice?

Step 4: Evaluate the Truth of Each Option
- For strengthening, the content must align with the passage. Look for predicate matches, and pay attention to paraphrasing and implication.
- For weakening, the passage must contain content that contradicts or invalidates the claim. Try rephrasing in the negative to check for conflict. There must be a logically consistent matching case or a clear counterexample.
- For experimental strengthening or weakening, Strengthening is explained through the method of agreement or method of difference. Weakening occurs through the presentation of a counterexample.
- Compare whether the given content aligns with the principles of the passage, and find the basis for the option in the passage by quoting or summarizing the corresponding supporting sentence.

- For each option, make a clear judgment of whether it is True or False, and write an explanation based on the key content identified in Step 1, 2.
- Write in the style of an explanation, using a descriptive and logical tone that compares each option’s statement with the corresponding content in the passage. Use declarative sentences ending with '-입니다.'


Clearly state whether each option is True or False.

Step 5: Determine the Final Answer
- Choose the correct answer based on the question type.
- Briefly explain why that answer is correct.

✅ Output Format (JSON)
Your final output must follow this exact JSON structure:
```json
{{
  "step1": "Write your explanation for Step 1 here in Korean.",
  "step2": "Write your explanation for Step 2 here in Korean.",
  "step3": "Write your explanation for Step 3 here in Korean.",
  "step4": "Write your explanation for Step 4 here in Korean.",
  "step5": "Write your explanation for Step 5 here in Korean.",
  "Final answer": "Write only the final answer here.\n- If the choices are in the form of ㄱ, ㄴ, ㄷ, etc., list only the correct ones separated by commas and spaces (e.g., 'ㄱ, ㄷ'). 
                    Multiple correct options are allowed, but you must select at least one.\n- If the choices are numbered (e.g., 1 to 5), write **only a single correct number** as an integer (e.g., '3'). Absolutely do NOT write multiple numbers like '①, ④'. Only one number must be selected.\n- If you are unsure, choose the most reasonable or likely correct answer based on the passage. Never leave the final answer blank."
}}

The input question is as follows:
<Question>
{input_question}
"""

## 추리논증 - 논증분석 ## 
PROMPT_ANALYSIS_REASONING = """
You are solving a Korean-language multiple-choice inference question based on a passage from the argument. The input will be in Korean, and your output must also be written in Korean. Follow the step-by-step reasoning process below and output your final response in JSON format.

Step 1: Analyze the Argument Structure
- Accurately identify the necessary and sufficient conditions (conditional structure) through the context.

Step 2: Identify the Question Format and Task Type
- Clarify what the question is asking:
    ☐ Is it asking for all correct statements?
    ☐ Is it asking for the incorrect option?
    ☐ Is it asking for a single best choice?

Step 3: Evaluate the Truth of Each Option
- Consider any additional principles required based on the choices, and determine whether the passage includes the relevant content by comparing the predicates.
- For each option, make a clear judgment of whether it is True or False, and write an explanation based on the key content identified in Step 1.
- Write in the style of an explanation, using a descriptive and logical tone that compares each option’s statement with the corresponding content in the passage. Use declarative sentences ending with '-입니다.'

Clearly state whether each option is True or False.

Step 4: Determine the Final Answer
- Choose the correct answer based on the question type.
- Briefly explain why that answer is correct.

✅ Output Format (JSON)
Your final output must follow this exact JSON structure:
```json
{{
  "step1": "Write your explanation for Step 1 here in Korean.",
  "step2": "Write your explanation for Step 2 here in Korean.",
  "step3": "Write your explanation for Step 3 here in Korean.",
  "step4": "Write your explanation for Step 4 here in Korean.",
  "step5": "Write your explanation for Step 5 here in Korean.",
  "Final answer": "Write only the final answer here.\n- If the choices are in the form of ㄱ, ㄴ, ㄷ, etc., list only the correct ones separated by commas and spaces (e.g., 'ㄱ, ㄷ'). 
                    Multiple correct options are allowed, but you must select at least one.\n- If the choices are numbered (e.g., 1 to 5), write **only a single correct number** as an integer (e.g., '3'). Absolutely do NOT write multiple numbers like '①, ④'. Only one number must be selected.\n- If you are unsure, choose the most reasonable or likely correct answer based on the passage. Never leave the final answer blank."
}}

The input question is as follows:
<Question>
{input_question}
"""

## 추리논증 - 논쟁 및 반론 ## 
PROMPT_DEBATE1_REASONING = """
You are solving a Korean-language multiple-choice inference question based on a passage from the debate The input will be in Korean, and your output must also be written in Korean. Follow the step-by-step reasoning process below and output your final response in JSON format.

Step 1: Identify the Point of Contention
- Identify the key point of contention in the given passage.
- Summarize the arguments and supporting reasons presented by each side regarding the central issue.

Step 2: Identify the Question Format and Task Type
- Clarify what the question is asking:
    ☐ Is it asking for all correct statements?
    ☐ Is it asking for the incorrect option?
    ☐ Is it asking for a single best choice?

Step 3: Evaluate the Truth of Each Option
- Based on the choices, determine how each statement relates to the arguments made in the passage.
- A valid counterargument either challenges a premise or denies the conclusion.
- For each option, make a clear judgment of whether it is True or False, and write an explanation based on the key content identified in Step 1.
- Write in the style of an explanation, using a descriptive and logical tone. Use declarative sentences ending with '-입니다.'

Clearly state whether each option is True or False.

Step 4: Determine the Final Answer
- Choose the correct answer based on the question type.
- Briefly explain why that answer is correct.

✅ Output Format (JSON)
Your final output must follow this exact JSON structure:
```json
{{
  "step1": "Write your explanation for Step 1 here in Korean.",
  "step2": "Write your explanation for Step 2 here in Korean.",
  "step3": "Write your explanation for Step 3 here in Korean.",
  "step4": "Write your explanation for Step 4 here in Korean.",
  "Final answer": "Write only the final answer here.\n- If the choices are in the form of ㄱ, ㄴ, ㄷ, etc., list only the correct ones separated by commas and spaces (e.g., 'ㄱ, ㄷ'). 
                    Multiple correct options are allowed, but you must select at least one.\n- If the choices are numbered (e.g., 1 to 5), write **only a single correct number** as an integer (e.g., '3'). Absolutely do NOT write multiple numbers like '①, ④'. Only one number must be selected.\n- If you are unsure, choose the most reasonable or likely correct answer based on the passage. Never leave the final answer blank."
}}

The input question is as follows:
<Question>
{input_question}
"""

## 추리논증 (통합) ## 
PROMPT_MATH_REASONING = """
You are solving a Korean-language multiple-choice inference question based on a passage from the logic game. The input will be in Korean, and your output must also be written in Korean. Follow the step-by-step reasoning process below and output your final response in JSON format.

Step 1: Interpreting Conditions and Organizing Information
- Carefully read the conditions given in the problem and grasp their meaning accurately.
- When necessary, visualize the situation with a table or diagram.
- If there are contradictory or mutually exclusive statements, consider both cases: when the statement is true and when it is false.

Based on the organized information, systematically enumerate all possible cases.

Step 2: Deriving Confirmed and Inferred Information
- Combine the conditions to first mark the facts that can be determined with certainty.
- Look for points where multiple conditions, when combined, reduce the number of possible cases, and use them to infer additional information.

Step 3: Identify the Question Format and Task Type
- Clarify what the question is asking:
    ☐ Is it asking for all correct statements?
    ☐ Is it asking for the incorrect option?
    ☐ Is it asking for a single best choice?

Step 4: Evaluate the Truth of Each Option
- If the reasoning process is complicated due to multiple possible cases, apply each option to the given conditions and use reverse reasoning to verify its validity.
- Write in the style of an explanation, using a descriptive and logical tone that compares each option’s statement with the corresponding content in the passage. Use declarative sentences ending with '-입니다.'

Clearly state whether each option is True or False.

Step 5: Determine the Final Answer
- Choose the correct answer based on the question type.
- Briefly explain why that answer is correct.

✅ Output Format (JSON)
Your final output must follow this exact JSON structure:
```json
{{
  "step1": "Write your explanation for Step 1 here in Korean.",
  "step2": "Write your explanation for Step 2 here in Korean.",
  "step3": "Write your explanation for Step 3 here in Korean.",
  "step4": "Write your explanation for Step 4 here in Korean.",
  "step5": "Write your explanation for Step 5 here in Korean.",
  "Final answer": "Write only the final answer here.\n- If the choices are in the form of ㄱ, ㄴ, ㄷ, etc., list only the correct ones separated by commas and spaces (e.g., 'ㄱ, ㄷ'). 
                    Multiple correct options are allowed, but you must select at least one.\n- If the choices are numbered (e.g., 1 to 5), write **only a single correct number** as an integer (e.g., '3'). Absolutely do NOT write multiple numbers like '①, ④'. Only one number must be selected.\n- If you are unsure, choose the most reasonable or likely correct answer based on the passage. Never leave the final answer blank."
}}

The input question is as follows:
<Question>
{input_question}
"""