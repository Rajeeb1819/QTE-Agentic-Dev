# Databricks notebook source
# MAGIC %run ../reusable/ado

# COMMAND ----------

# MAGIC %run ../reusable/jira

# COMMAND ----------

# MAGIC %run ../reusable/llm

# COMMAND ----------

# MAGIC %run ../reusable/github

# COMMAND ----------

# MAGIC %run ../reusable/log

# COMMAND ----------

# MAGIC %run ../reusable/database

# COMMAND ----------
import json, re, datetime
import traceback 

from databricks.sdk.runtime import *


def generate_manual_tests_from_requirement(tool, domain_or_org, project_key, requirement_id, requested_on, test_type):
    start_timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
    if tool == 'JIRA':
        read_requirement_flag, read_response, requirement_id, summary, description, acceptance_criteria, parent_key, parent_summary,  parent_description, missing_fields = read_requirement_details(domain_or_org, project_key, requirement_id)   
    elif tool == 'ADO':
        read_requirement_flag, read_response, requirement_id, summary, description, acceptance_criteria, parent_key, parent_summary,  parent_description, missing_fields = ado_read_requirement_details(domain_or_org, project_key, requirement_id)
    end_timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]

    if read_requirement_flag: 
        add_request_process_log(requested_on, "read_requirement", start_timestamp, end_timestamp, "Passed", "")
    else: 
        add_request_process_log(requested_on, "read_requirement", start_timestamp, end_timestamp, "Failed", str(read_response))
        return False, "Read requirement stage failed"
    
    if test_type.lower() == "bdd":
        input_data, prompt_template = prompt_for_gherkin_style(requirement_id, summary, description, acceptance_criteria)
    else:
        input_data, prompt_template = prepare_prompt_from_user_requirements(requirement_id, summary, description, acceptance_criteria)
    message = prepare_meassge(input_data, prompt_template)

    start_timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
    llm_flag, llm_response = azure_openai_request(message, 0)
    end_timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]

    if llm_flag: 
        testcase_response = llm_response.json()['choices'][0]['message']['content']
        add_llm_log(requested_on, str(message), start_timestamp, end_timestamp, str(llm_response.json()['choices'][0]['message']), str(llm_response.status_code), "First Try")
        add_request_process_log(requested_on, "llm_calls", start_timestamp, end_timestamp, "Passed", "")
    else: 
        add_llm_log(requested_on, str(message), start_timestamp, end_timestamp, str(llm_response), llm_response.get("Status Code", None), f"{llm_response}")
        add_request_process_log(requested_on, "llm_calls", start_timestamp, end_timestamp, "Failed", str(llm_response))
        write_comment(domain_or_org, requirement_id, f"Unable to create Manual Test Case for requirement due to Azure OpenAI response issue, Contact GenAI InsighQA Admin : {requirement_id}")
        return False, "LLM call stage failed"

    if test_type.lower() == "bdd":
        test_details = parse_the_data(testcase_response, tool)
        create, message = create_manual_testcase_bdd_format(test_details, tool, project_key, domain_or_org, requirement_id, requested_on) 
        if not create: return False, "Unable to create manual testcase, contact GenAI InsightQA for error resolution"
        return create, message
            
    
    start_timestamp_1 = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
    post_processing_flag_1, testcase_data_raw = remove_double_slash_from_raw_data(testcase_response)
    end_timestamp_1 = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
    if not post_processing_flag_1:
        add_request_process_log(requested_on, "post_processing", start_timestamp_1, end_timestamp_1, "Failed", str(testcase_data_raw))
        return False, "Post processing 1 stage failed"
    
    start_timestamp_2 = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
    post_processing_flag_2, testcase_details = create_test_data_dictionary(testcase_data_raw)
    end_timestamp_2 = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
    if post_processing_flag_2:
        add_request_process_log(requested_on, "post_processing", start_timestamp_2, end_timestamp_2, "Passed", "")
    else:
        add_request_process_log(requested_on, "post_processing", start_timestamp_2, end_timestamp_2, "failed", str(testcase_details))
        return False, "Post processing 2 stage failed"

    create, message = generate_manual_testcase_in_jira_ado(testcase_details, domain_or_org, project_key, parent_key, requirement_id, tool, requested_on)
    if not create: return False, "Unable to create manual testcase, contact GenAI InsightQA for error resolution"
    return create, message

def prepare_meassge(input_data, prompt_template):
    prompt = prompt_template.format(input_data = input_data)
    message = [{"role": "system", "content": "You are an expert Tester."}, {"role": "user", "content": prompt}]
    return message

def remove_double_slash_from_raw_data(testcases_response:str):
    '''
    To remove the '\n\n' form the prompt response.
    '''
    try:
        testcase_double_slash = re.split(r'\n{2,}', testcases_response)
        data_into_list = []
        temp = ""
            
        for line in testcase_double_slash:
            if line.strip(): 
                if line.lower().startswith("test case") and temp: 
                    data_into_list.append(temp.strip())
                    temp = line
                elif line.startswith("Note"):
                    continue
                else:
                    temp += f"\n{line}"  
        
        if temp: 
            data_into_list.append(temp.strip())
        return True, data_into_list
    except Exception as e:
        return False, {"message": f"MANUAL_TEST_GENERATION_1 : {e}"}
def split_list_into_action_expected_result(testcase_data_list:list, element:str):
    '''
    Split the details into action:list and expected result:list
    '''
    try:
        if element in testcase_data_list:
            index = testcase_data_list.index(element)
            action = testcase_data_list[:index]
            expected_result = testcase_data_list[index+1:]
            return action, expected_result
        else:
            return [], testcase_data_list
    except Exception as e:
        return False, {"message": f"MANUAL_TEST_GENERATION_2 : {e}"}
    
def create_test_data_dictionary(testcase_data:list):
    '''
    Create list[dict] --> [{"Test Summary":"", "Test Actions":[], "Expected Result":[]}, ...]
    '''
    try:
        test_case = []
        for i in range(0, len(testcase_data)):
            test_case.append(testcase_data[i].split("\n"))

        test_case_details_list = []
        for data in test_case:
            test_case_details = {}
            for each_data in data:
                if ": " in each_data:
                    test_case_summary = each_data.split(":")[1].strip()
                    test_case_details["Test Summary"] = test_case_summary  
                action, expected_result = split_list_into_action_expected_result(data, 'Expected Results:')
                if action == False:
                    return action, expected_result
                test_case_details["Test Actions"] = action[2:] 
                test_case_details["Expected Result"] = expected_result
            test_case_details_list.append(test_case_details)
        return True, test_case_details_list
    except Exception as e:
        return False, {"message": f"MANUAL_TEST_GENERATION_3 : {e}"}

def create_data_for_api(test_case_details:dict) -> list:
    '''
    Create data payload for Azure and Jira APIs.
    [{"action":"", "result":""}]
    '''
    single_testcase_data = [{"action": re.sub(r'^[0-9]', '', test_case_details["Test Actions"][i]).lstrip('.').lstrip(), "result": re.sub(r'^[0-9]', '', test_case_details["Expected Result"][i]).lstrip('.').lstrip()} for i in range(len(test_case_details["Test Actions"]))]
    return single_testcase_data

def generate_manual_testcase_in_jira_ado(test_case_details_list:list, domain_or_org, project_key, parent_key, requirement_id, tool, requested_on):
    '''
    Creating the manual testcases when action == expected result.
    '''
    test_id_list = ""

    for test_case in test_case_details_list:
        logger.debug(f"Test Actions : {len(test_case['Test Actions'])} - Expected Result : {len(test_case['Expected Result'])}")
        if len(test_case["Test Actions"]) == len(test_case["Expected Result"]):
            single_testcase_data = create_data_for_api(test_case)
            flag, test_id_list = call_jira_ado_test_creation_api(single_testcase_data, test_case["Test Summary"], domain_or_org, project_key, parent_key, requirement_id, tool, test_id_list, requested_on)
        else:
            flag, test_id_list = create_testcase_for_failed_scenario(test_case, domain_or_org, project_key, requirement_id, parent_key, tool, test_id_list, requested_on)
    
    review_comment_start_time = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
    if tool == "JIRA":
        flag, response = write_comment(domain_or_org, requirement_id, f"Below are the manual testcase created for this requirement : {test_id_list}")
    elif tool == "ADO":
        flag, response = ado_write_comment_with_format(domain_or_org, project_key, requirement_id, f"Below are the manual testcase created for this requirement : {test_id_list}")
    review_comment_end_time = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
    if flag:
        add_request_process_log(requested_on, "api_calls_review_comment", review_comment_start_time, review_comment_end_time, "Passed", "")
    else: add_request_process_log(requested_on, "api_calls_review_comment", review_comment_start_time, review_comment_end_time, "Failed", str(response))
    return True, ""

def call_jira_ado_test_creation_api(single_testcase_data:list, summary, domain_or_org, project_key, parent_key, requirement_id, tool, test_id_list, requested_on):
    '''
    Call the function from Jira and ADO for creating the Manual Testcase in Jira Xray and Azure DevOps.
    '''
    create_test_start_timestamp =  datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
    if tool.lower() == "jira":
        testcase_flag, testcase_id = create_manual_testcase(project_key, "", summary, json.dumps(single_testcase_data))
    else:
        testcase_flag, testcase_id = create_manual_testcase_ado(domain_or_org, project_key, "", summary, single_testcase_data, requirement_id)
    create_test_end_timestamp =  datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]

    if testcase_flag: 
        test_id_list = f"{test_id_list}\n{testcase_id}"
        if tool.lower() == "jira":
            link_test_with_requirement(domain_or_org, project_key, requirement_id, testcase_id)
        add_request_process_log(requested_on, "api_calls_create_testcase", create_test_start_timestamp, create_test_end_timestamp, "Passed", testcase_id)
    else: 
        add_request_process_log(requested_on, "api_calls_create_testcase", create_test_start_timestamp, create_test_end_timestamp, "Failed", str(testcase_id))
        return False, "Test case creation stage failed"
    return True, test_id_list
def create_testcase_for_failed_scenario(test_case:dict, domain_or_org, project_key, requirement_id, parent_key, tool, test_id_list, requested_on):
    '''
    Creating testcase where 'action != expected result'.
    Making additional LLM calls.
    '''
    input_data, prompt_template = prepare_prompt_from_testcase(test_case["Test Summary"], "\n".join(test_case["Test Actions"]), "\n".join(test_case["Expected Result"]), requirement_id)
    message = prepare_meassge(input_data, prompt_template)

    llm_start_timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
    llm_flag, llm_response = azure_openai_request(message, 0)
    llm_end_timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]

    if llm_flag: 
        testcase_response = llm_response.json()['choices'][0]['message']['content']
        add_llm_log(requested_on, str(message), llm_start_timestamp, llm_end_timestamp, str(testcase_response), str(llm_response.status_code), "")
    else: 
        add_llm_log(requested_on, str(message), llm_start_timestamp, llm_end_timestamp, llm_response, llm_response.get("Status Code", None), f"Failed_testcase_retry : {llm_response}")
        write_comment(domain_or_org, requirement_id, f"Unable to create Manual Test Case for requirement due to Azure OpenAI response issue, Contact GenAI InsighQA Admin : {requirement_id}")
        return False, "LLM call stage failed for failed scenario"
    
    post_processing_start_timestamp_1 = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
    post_processing_flag_1, testcase_data_raw = remove_double_slash_from_raw_data(testcase_response)
    post_processing_end_timestamp_1 = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
    if not post_processing_flag_1:
        add_request_process_log(requested_on, "post_processing", post_processing_start_timestamp_1, post_processing_end_timestamp_1, "Failed", str(testcase_data_raw))
        return False, "Post processing 1 stage failed for failed scenario"
    
    post_processing_start_timestamp_2 = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
    post_processing_flag_2, testcase_details = create_test_data_dictionary(testcase_data_raw)
    post_processing_end_timestamp_2 = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
    if post_processing_flag_2:
        add_request_process_log(requested_on, "post_processing", post_processing_start_timestamp_2, post_processing_end_timestamp_2, "Passed", "")
    else:
        add_request_process_log(requested_on, "post_processing", post_processing_start_timestamp_2, post_processing_end_timestamp_2, "failed", str(testcase_details))
        return False, "Post processing 2 stage failed for failed scenario"


    for test_case_foreach in testcase_details:
        if len(test_case_foreach["Test Actions"]) == len(test_case_foreach["Expected Result"]):
            single_testcase_data = create_data_for_api(test_case_foreach)
            flag, test_id_list = call_jira_ado_test_creation_api(single_testcase_data, test_case_foreach["Test Summary"], domain_or_org, project_key, parent_key, requirement_id, tool, test_id_list, requested_on)
        # else:
        #     logger.error("Encountered issue while generating each testcases : create_testcase_for_failed_scenario()")
        #     if tool == "JIRA":
        #         write_comment(domain_or_org, requirement_id, f"Encountered some issue while generating the testcases. Please reach out to QTE GenAI Team for error resolution.")
        #     elif tool == "ADO":
        #         ado_write_comment_with_format(domain_or_org, project_key, requirement_id, f"Encountered some issue while generating the testcases. Please reach out to QTE GenAI Team for error resolution.")
    return flag, test_id_list

def prepare_prompt_from_user_requirements(requirement_id, summary, description, acceptance_criteria):
    '''
    Preparing the prompt to create manual testcases by using requirement_id, summary, description, acceptance_criteria.
    passed scenarios
    '''
    # Create input data for generating test cases
    input_data = f"USER_STORY_ID: {requirement_id}\n" \
                    f"FEATURE_DESCRIPTION: {summary}\n" \
                    f"USER_STORY: {description}\n" \
                    f"ACCEPTANCE_CRITERIA: {acceptance_criteria}"
    
    prompt_template = """Complete the test case generation task in a step by step manner as per the instructions. Generate the functional test cases based on the User_Story. Each test case should include the following:
        User_Story: {input_data}
        Test Case Generation Task:
        Here are the requirements for generating test cases:
        - Read and understand the USER_STORY, ACCEPTANCE_CRITERIA from the User_Story.
        - Start the test cases with the title "TESTCASES FOR USER STORY" and add the USER_STORY_ID here.
        - Generate test cases for each USER_STORY and ACCEPTANCE_CRITERIA, covering positive and negative scenarios.
        - Take one ACCEPTANCE_CRITERIA at a time. Review the genertated test case, test steps and expected results they should be as per the instructions.
        - Generate test cases based on each ACCEPTANCE_CRITERIA from the User_Story, do not leave any gaps.
        - Include edge cases to test system boundaries.
        - Ensure that there are no gaps between functional requirements in USER_STORY detail and test case coverage.
        - For each test case, provide the precise, step by step navigation.
        - Cover all cases.
        - Output should provide test cases but not include any code or any other information.
        - The output should contain Test case number with description and test steps for every test case with expected results for every correponding test step.
        - The format of the output should be Test Case: Description about the test case.
        - Following the description,on the next line,Test Steps should be printed then the corresponding expected result for every test step.
        - Make sure for every test case, the number of test steps must be equal to the expected results.
        - Number of test steps should be same as the number of Expected results.
        - Every test step should have corresponding expected results.
        - Every expected result should have corresponding test step.
        - Expected results should also be numbered according to the test steps.
        - Output will be invalid if number of test step is not equal to the number of expected results.
        - It is complusary to follow all instructions.   
        """
    return input_data, prompt_template

def prepare_prompt_from_testcase(test_summary, test_action, test_expected_result, requirement_id):
    '''
    Preparing the prompt to create manual testcases by using test_summary, test_action, test_expected_result, requirement_id.
    for failed scenario
    '''
    # Create input data for generating test cases
    input_data = f"Test Case Summary: {test_summary}\n" \
                    f"Test Case Actions: \n{test_action}\n" \
                    f"Test Case Expected Result: \n{test_expected_result}\n" \
                    f"User story id: {requirement_id}"
    
    prompt_template = """In the User_Story there is Test Case, Test Steps and Expected Results. But the number of test steps and expected results are not equal.For a test case to be successful number of test steps and expected results should be equal.Please go through the given User_Story and generate Test steps and Expected results as instructed.
        Below are the following detail:
        User_Story: {input_data}
        Test Case Generation Task:
        Here are the requirements for generating test steps and expected results:
        - Read and understand the Test Case, Test Steps from the User_Story.
        - Start the Test cases with the title "TESTCASES FOR USER STORY" and add the USER_STORY_ID here.
        - Generate expected results for the given test case and test steps.
        - Take Test steps and review the genertated expected results they should be as per the instructions.
        - Do not leave gap while creating expected results.
        - Output should not include any code or any other information.
        - It is very important to count and check if number of test step is equal to the expected results.
        - The output should contain Test case number with description and test steps for test case with expected results.
        - The format of the output should be Test Case: Description about the test case.
        - Following the description,on the next line,Under the heading Test Steps, test steps should be printed then on the next line.
        - Under the heading Expected Results, Expected Results should be printed then on the next line.
        There should be expected result for every correponding test step.
        - Make number of test steps must be equal to the expected results.
        - Every test step should have corresponding expected results.
        - Expected results should also be numbered according to the test steps.
        - Do not generate test cases.
        - Output will be invalid if number of test step is not equal to the number of expected results.
        - It is complusary to follow all instructions.
        """
    return input_data, prompt_template

def prompt_for_gherkin_style(requirement_id, summary, description, acceptance_criteria):
    input_data = (
        f"USER_STORY_ID: {requirement_id}\n"
        f"USER_STORY_SUMMARY: {summary}\n"
        f"USER_STORY_DESCRIPTION: {description}\n"
        f"ACCEPTANCE_CRITERIA: {acceptance_criteria}"
    )
    prompt_template = """In the User_Story, there is a need to generate test cases and test steps in Gherkin Behaviour driven development format. Below are the instructions for creating the test cases and test steps:
            The basic syntax of Gherkin includes:

            · Feature: A high-level description of a software feature.
            · Scenario: A specific example or use case that illustrates a feature.
            · Given: The initial system state or context.
            · When: The action (usually a user action) that triggers a system change.
            · Then: The expected outcome.
            · And/But: Any extra steps or conditions.
            · Background: To define common context or preconditions that are shared across multiple scenarios within a feature.
            · Scenario Outline: To run a single scenario multiple times with different sets of data.
            · Examples: A table that provides the data required to execute a Scenario Outline.

            Testers do not have to use every element of this syntax but Scenario, Given, When, and Then are compulsory.
            User_Story: {input_data}
            1. Read and understand the User_Story to identify the requirements for test cases and test steps.
            2. List the test case summary, test step & expected results for each test case in Gherkin format to describe the action or event as per the Gherkin Behaviour driven development syntax.
            3. Each test step should be written in a clear and concise manner, focusing on the specific action being performed.
            4. Repeat the process for each test case in the User_Story.
            5. The output should be in a clear and organized format, following the Gherkin Behaviour driven development syntax.
            6. Do not include any code or unnecessary information in the output.
            7. It is important to follow all the instructions and ensure that the test steps and expected results are accurately generated.
            8. Output format shuold be as follows: Feature: Test case description/summary followed by Test steps followed by Expected results.
            9. The Test cases, Test steps & Expected results should be strictly in Gherkin Behaviour driven development format.
            10. Do not include any unnecessary information in the output.
            """
    return input_data, prompt_template

def parse_the_data(testcase_response: str, tool: str) -> list:
    print("Parse bdd data in")
    lines = testcase_response.strip().splitlines()
    feature_title = ""
    output = []
    scenario_lines = []
    in_scenario = False
    current_summary = ""
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Feature:"):
            feature_title = stripped
 
        # Scenario or Scenario Outline begins
        elif stripped.startswith("Scenario"):
 
            # Save previous scenario block
            if scenario_lines:
                if tool.lower() == "jira":
                    output.append({
                        "Test Summary": current_summary,
                        "Scenario": "\n".join(scenario_lines)
                    })
                else:
                    output.append({
                        "Test Summary": current_summary,
                        "Test Actions": scenario_lines,
                        "Expected Result" : len(scenario_lines) * [""]
                    })
                scenario_lines = []
 
            in_scenario = True
            scenario_lines = [feature_title, stripped]
            current_summary = re.sub(r'^Scenario(?: Outline)?:\s*', '', stripped)
 
        # Collect rest of scenario steps
        elif in_scenario:
            if stripped == "" and not scenario_lines:
                continue  # skip leading blank lines
            scenario_lines.append(line)
 
    # Catch last scenario block
    if tool.lower() == "jira":
        if scenario_lines:
            output.append({
                "Test Summary": current_summary,
                "Scenario": "\n".join(scenario_lines)
            })
    else: 
        if scenario_lines:
            output.append({
                "Test Summary": current_summary,
                "Test Actions": scenario_lines,
                "Expected Result" : len(scenario_lines) * [""]
            })
 
    return output

def create_manual_testcase_bdd_format(testcase_details:list, tool, project_key, domain_or_org, requirement_id, requested_on):
    print("create_manual_testcase_bdd_format in")
    start_timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
    test_id_list = ""
    print(testcase_details)
    for data in testcase_details:
        print(f"Data : {data}")
        if "Test Summary" in data:
            test_summary = data.get("Test Summary").strip()
            
            if tool.lower() == "jira":
                test_scenario = data.get("Scenario").strip()
                testcase_flag, testcase_id = create_bdd_testcase(test_scenario, test_summary, project_key)
            else:
                test_step = create_data_for_api(data)
                testcase_flag, testcase_id = create_manual_testcase_ado(
                    domain_or_org, project_key, "", test_summary, test_step, requirement_id
                )
            
            
            if testcase_flag: 
                test_id_list = f"{test_id_list}\n{testcase_id}"
                if tool.lower() == "jira":
                    link_test_with_requirement(domain_or_org, project_key, requirement_id, testcase_id)
    
    end_timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]

    if test_id_list:
        add_request_process_log(requested_on, "api_calls_create_testcase", start_timestamp, end_timestamp, "Passed", testcase_id) 
    else:
        add_request_process_log(requested_on, "api_calls_create_testcase", start_timestamp, end_timestamp, "Failed", str(testcase_id))
        return False, "Test case creation stage failed"
    
    
    review_comment_start_time = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
    if tool == "JIRA":
        flag, response = write_comment(domain_or_org, requirement_id, f"Below are the manual testcase created for this requirement : {test_id_list}")
    elif tool == "ADO":
        flag, response = ado_write_comment_with_format(domain_or_org, project_key, requirement_id, f"Below are the manual testcase created for this requirement : {test_id_list}")
    review_comment_end_time = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
    if flag:
        add_request_process_log(requested_on, "api_calls_review_comment", review_comment_start_time, review_comment_end_time, "Passed", "")
    else: add_request_process_log(requested_on, "api_calls_review_comment", review_comment_start_time, review_comment_end_time, "Failed", str(response))
    print("create_manual_testcase_bdd_format out")
    return True, ""
