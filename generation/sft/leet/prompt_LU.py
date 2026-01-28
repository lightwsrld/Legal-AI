## 언어이해 - 일치/부합 ## 
PROMPT_OX_REASONING = """
You are solving a Korean-language multiple-choice inference question based on a passage. The input will be in Korean, and your output must also be written in Korean. Follow the step-by-step reasoning process below and output your final response in JSON format.

Step 1: Identify Keyword
- Understand what each option or choice is asserting.
- Extract the key terms and ideas mentioned in each option, and find the corresponding parts in the passage to establish a one-to-one correspondence.

Step 2: Identify the Question Format and Task Type
- Clarify what the question is asking:
    ☐ Is it asking for all correct statements?
    ☐ Is it asking for the incorrect option?
    ☐ Is it asking for a single best choice?

Step 3: Evaluate the Truth of Each Option
- Refer to the part of the passage where the relevant content is discussed.
- If an option states the same idea as the passage, judge it as True.
- For each option, make a clear judgment of whether it is True or False, and write an explanation based on the key content identified in Step 1.
- Write in the style of an explanation, using a descriptive and logical tone that compares each option’s statement with the corresponding content in the passage. Use declarative sentences ending with '-입니다.'

Clearly state whether each option is True or False.

[Reference]
- For questions asking for the correct statement(s), if two options contradict each other within the passage, only one can be chosen as correct.
- A: "a may do b" ≠ B: "a does b"
- "If B, then A" is true, but "If A, then B" is not necessarily true.

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
                    Multiple correct options are allowed, but you must select at least one.\n- If the choices are numbered (e.g., 1 to 5), write only a single correct number as an integer (e.g., '3'). Absolutely do NOT write multiple numbers like '①, ④'. **Only one number** must be selected. In this case, even if multiple choices seem correct, select only one based on the passage by identifying the most reasonable answer or using the process of elimination."
}}


The input question is as follows:
<Question>
{input_question}
"""

## 언어이해 - 지문 기반 추론 (글쓴이의 견해) ## 
PROMPT_AUTHOR_REASONING = """
You are solving a Korean-language multiple-choice inference question based on a passage. The input will be in Korean, and your output must also be written in Korean. Follow the step-by-step reasoning process below and output your final response in JSON format.

Step 1: Passage Segmentation
- Segment the passage into logical parts and identify the paragraph(s) where the author’s viewpoint is explicitly stated.
- Determine the author’s main argument or key conceptual dimension in those sections.

Step 2: Identify the Question Format and Task Type
- Clarify what the question is asking:
    ☐ Is it asking for all correct statements?
    ☐ Is it asking for the incorrect option?
    ☐ Is it asking for a single best choice?

Step 3: Evaluate the Truth of Each Option
- Refer to the part of the passage where the relevant content is discussed.
- If an option states the same idea as the passage, judge it as True.
- For each option, make a clear judgment of whether it is True or False, and write an explanation based on the key content identified in Step 1.
- Write in the style of an explanation, using a descriptive and logical tone that compares each option’s statement with the corresponding content in the passage. Use declarative sentences ending with '-입니다.'

Clearly state whether each option is True or False.

[Reference]
- For questions asking for content that contrasts with the author’s view, base your judgment on what the author is criticizing or opposing.
- A: "a may do b" ≠ B: "a does b"
- "If B, then A" is true, but "If A, then B" is not necessarily true.

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
                    Multiple correct options are allowed, but you must select at least one.\n- If the choices are numbered (e.g., 1 to 5), write only a single correct number as an integer (e.g., '3'). Absolutely do NOT write multiple numbers like '①, ④'. **Only one number** must be selected. In this case, even if multiple choices seem correct, select only one based on the passage by identifying the most reasonable answer or using the process of elimination."
}}


The input question is as follows:
<Question>
{input_question}
"""

## 언어이해 - 지문 기반 추론 (기본) ## 
PROMPT_LOGIC_REASONING = """
You are solving a Korean-language multiple-choice inference question based on a passage. The input will be in Korean, and your output must also be written in Korean. Follow the step-by-step reasoning process below and output your final response in JSON format.

Step 1: Identify Keyword
- Understand what each question or choice is asserting.
- Extract the key terms and ideas mentioned in each option, and find the corresponding parts in the passage to establish a one-to-one correspondence.

Step 2: Identify the Question Format and Task Type
- Clarify what the question is asking:
    ☐ Is it asking for all correct statements?
    ☐ Is it asking for the incorrect option?
    ☐ Is it asking for a single best choice?

Step 3: Evaluate the Truth of Each Option
- Refer to the part of the passage where the relevant content is discussed.
- If an option functionally reflects the same structure as the author’s argument or maintains the same logical relationship (even if different wording is used), judge the option as True.
- For each option, make a clear judgment of whether it is True or False, and write an explanation based on the key content identified in Step 1.
- Write in the style of an explanation, using a descriptive and logical tone that compares each option’s statement with the corresponding content in the passage. Use declarative sentences ending with '-입니다.'

Clearly state whether each option is True or False.

[Reference]
- Be cautious of turning expressions of possibility into definitive statements. (e.g., A: "A may do B." ≠ B: "A does B.")
- Be cautious of misrepresenting a partial argument as the author’s overall stance.
- Be cautious of omitting conditions and only stating the conclusion. (i.e., leaving out higher-level premises)
- Even when the same words are used (e.g., justice, freedom, fairness), be aware that their functional meaning can change depending on the context.

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
                    Multiple correct options are allowed, but you must select at least one.\n- If the choices are numbered (e.g., 1 to 5), write only a single correct number as an integer (e.g., '3'). Absolutely do NOT write multiple numbers like '①, ④'. **Only one number** must be selected. In this case, even if multiple choices seem correct, select only one based on the passage by identifying the most reasonable answer or using the process of elimination."
}}


The input question is as follows:
<Question>
{input_question}
"""

## 언어이해 - 주장/요소 비교 (ㄱ,ㄴ) ## 
PROMPT_AB_REASONING = """
You are solving a Korean-language multiple-choice inference question based on a passage. The input will be in Korean, and your output must also be written in Korean. Follow the step-by-step reasoning process below and output your final response in JSON format.

Step 1: Understanding the Context
- Identify and interpret the meaning of elements corresponding to ᄀ and ᄂ within the passage by analyzing their contextual connections.
- Organize the information related to ᄀ and ᄂ in a contrasting format to facilitate comparison.

Step 2: Identify the Question Format and Task Type
- Clarify what the question is asking:
    ☐ Is it asking for all correct statements?
    ☐ Is it asking for the incorrect option?
    ☐ Is it asking for a single best choice?

Identify whether the question is about truth/falsity, logical possibility, or appropriateness.

Step 3: Evaluate the Truth of Each Option
- For each option, make a clear judgment of whether it is True or False, and write an explanation based on the key content identified in Step 1.
- Write in the style of an explanation, using a descriptive and logical tone that compares each option’s statement with the corresponding content in the passage. Use declarative sentences ending with '-입니다.'

Clearly state whether each option is True or False.

[Reference]
- A: "a may do b" ≠ B: "a does b"
- "If B, then A" is true, but "If A, then B" is not necessarily true.

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
                    Multiple correct options are allowed, but you must select at least one.\n- If the choices are numbered (e.g., 1 to 5), write only a single correct number as an integer (e.g., '3'). Absolutely do NOT write multiple numbers like '①, ④'. **Only one number** must be selected. In this case, even if multiple choices seem correct, select only one based on the passage by identifying the most reasonable answer or using the process of elimination."
}}


The input question is as follows:
<Question>
{input_question}
"""

## 언어이해 - 주장/요소 비교 (밑줄) ## 
PROMPT_LINE_REASONING = """
You are solving a Korean-language multiple-choice inference question based on a passage. The input will be in Korean, and your output must also be written in Korean. Follow the step-by-step reasoning process below and output your final response in JSON format.

Step 1: Understanding the Context
- Comprehend the meaning of words or phrases based on the underlined part by interpreting them within the paragraph.

Step 2: Identify the Question Format and Task Type
- Clarify what the question is asking:
    ☐ Is it asking for all correct statements?
    ☐ Is it asking for the incorrect option?
    ☐ Is it asking for a single best choice?

Identify whether the question is about truth/falsity, logical possibility, or appropriateness.

Step 3: Evaluate the Truth of Each Option
- For each option, make a clear judgment of whether it is True or False, and write an explanation based on the key content identified in Step 1.
- Write in the style of an explanation, using a descriptive and logical tone that compares each option’s statement with the corresponding content in the passage. Use declarative sentences ending with '-입니다.'

Clearly state whether each option is True or False.

[Reference]
- A: "a may do b" ≠ B: "a does b"
- "If B, then A" is true, but "If A, then B" is not necessarily true.

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
                    Multiple correct options are allowed, but you must select at least one.\n- If the choices are numbered (e.g., 1 to 5), write only a single correct number as an integer (e.g., '3'). Absolutely do NOT write multiple numbers like '①, ④'. **Only one number** must be selected. In this case, even if multiple choices seem correct, select only one based on the passage by identifying the most reasonable answer or using the process of elimination."
}}


The input question is as follows:
<Question>
{input_question}
"""

## 언어이해 - 보기 기반 (ㄱ, ㄴ, ㄷ) ## 
PROMPT_OPTION_1_REASONING = """
You are solving a Korean-language multiple-choice inference question based on a passage. The input will be in Korean, and your output must also be written in Korean. Follow the step-by-step reasoning process below and output your final response in JSON format.

Step 1: Understand Key Concepts in the Passage
- Identify how economic or political concepts are defined in the introduction of the passage, and rephrase them in your own words to internalize the ideas.

Step 2: Recognize the Structure of the Passage and Analyze Economic/Political Phenomena Based on Its Perspective
- Determine whether the passage presents multiple viewpoints or develops a single unified theory.
- Then, analyze the economic or political phenomena using the new framework or perspective introduced in the passage.

Step 3: Identify Components That Can Be Modeled
- If the passage presents aspects of an economic or political system that can be modeled, organize and express them in a structured or schematic form.

Step 4: Understand the Structure of the <Option>
- Read each <option> and identify its logical structure (e.g., cause-effect, definition, comparison).
- Determine which part of the passage corresponds to that structure, and interpret the option accordingly.
- (Cause-effect) If the option contains a cause-and-effect relationship, link it to the paragraph in the passage that discusses such causality.
- (Definition) If the option involves a conceptual explanation, connect it to the paragraph where that concept is defined or explained.
- (Comparison) If the option compares or contrasts two concepts, relate it to the part of the passage where such a comparison is made.

Step 5: Identify the Question Format and Task Type
- Clarify what the question is asking:
    ☐ Is it asking for all correct statements?
    ☐ Is it asking for the incorrect option?
    ☐ Is it asking for a single best choice?

Step 6: Evaluate the Truth of Each Option
- Identify how the structure of the <option> reconstructs a specific structure from the passage, and apply the option’s logic accordingly in your interpretation.
- For each option, make a clear judgment of whether it is True or False, and write an explanation based on the key content identified in Step 1, 2 and 3.
- Write in the style of an explanation, using a descriptive and logical tone that compares each option’s statement with the corresponding content in the passage. Use declarative sentences ending with '-입니다.'

Clearly state whether each option is True or False.

Step 7: Determine the Final Answer
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
  "step6": "Write your explanation for Step 5 here in Korean.",
  "step7": "Write your explanation for Step 5 here in Korean.",
  "Final answer": "Write only the final answer here.\n- If the choices are in the form of ㄱ, ㄴ, ㄷ, etc., list only the correct ones separated by commas and spaces (e.g., 'ㄱ, ㄷ'). 
                    Multiple correct options are allowed, but you must select at least one.\n- If the choices are numbered (e.g., 1 to 5), write only a single correct number as an integer (e.g., '3'). Absolutely do NOT write multiple numbers like '①, ④'. **Only one number** must be selected. In this case, even if multiple choices seem correct, select only one based on the passage by identifying the most reasonable answer or using the process of elimination."
}}


The input question is as follows:
<Question>
{input_question}
"""

## 언어이해 - 보기 기반 ## 
PROMPT_OPTION_REASONING = """
You are solving a Korean-language multiple-choice inference question based on a passage. The input will be in Korean, and your output must also be written in Korean. Follow the step-by-step reasoning process below and output your final response in JSON format.

Step 1: Structuring the Passage
- Identify the logical structure of each paragraph (e.g., cause-effect, definition, comparison), and summarize the key concept conveyed in each section.

Step 2: Understand the Structure of the <Option>
- Identify which concept from the passage is being paraphrased in the option statement.

Step 3: Identify the Question Format and Task Type
- Clarify what the question is asking:
    ☐ Is it asking for all correct statements?
    ☐ Is it asking for the incorrect option?
    ☐ Is it asking for a single best choice?

Step 4: Evaluate the Truth of Each Option
- Based on step 1 and 2, determine which paragraph in the passage the option statement aligns with contextually, and analyze it by applying that connection.
- For each option, make a clear judgment of whether it is True or False, and must consult the *[Reference] when making the judge*.
- Write in the style of an explanation, using a descriptive and logical tone that compares each option’s statement with the corresponding content in the passage. Use declarative sentences ending with '-입니다.'

Clearly state whether each option is True or False.

[Reference]
- Be cautious of turning expressions of possibility into definitive statements. (e.g., A: "A may do B." ≠ B: "A does B.")
- Be cautious of misrepresenting a partial argument as the author’s overall stance.
- Be cautious of omitting conditions and only stating the conclusion. (i.e., leaving out higher-level premises)
- Even when the same words are used (e.g., justice, freedom, fairness), be aware that their functional meaning can change depending on the context.

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
                    Multiple correct options are allowed, but you must select at least one.\n- If the choices are numbered (e.g., 1 to 5), write only a single correct number as an integer (e.g., '3'). Absolutely do NOT write multiple numbers like '①, ④'. **Only one number** must be selected. In this case, even if multiple choices seem correct, select only one based on the passage by identifying the most reasonable answer or using the process of elimination."
}}


The input question is as follows:
<Question>
{input_question}
"""

