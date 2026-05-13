# Databricks notebook source
# MAGIC %run ../reusable/jira

# COMMAND ----------

# MAGIC %run ../reusable/ado

# COMMAND ----------

# MAGIC %run ../reusable/llm

# COMMAND ----------

# MAGIC %run ../reusable/github

# COMMAND ----------

# MAGIC %run ../reusable/log

# COMMAND ----------

# MAGIC %run ../reusable/database

# COMMAND ----------

from databricks.sdk.runtime import *
import re, datetime

def generate_selenium_scripts(tool, domain_or_org, project, requirement_id, testcase_id, test_scripting_language, github_repo, requested_on, test_type):
    global report_testscriptc, testscript, prompt_template_2
    read_testcase_start_timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
    try:
        if tool == 'JIRA':
            flag, test_description, testcases = read_testcase_details(domain_or_org, project, testcase_id)
            print(f"Test Description : {test_description}\n Testcase : {testcases}")
        elif tool == 'ADO':
            flag, test_description, testcases = read_testcase_details_ado(domain_or_org, project, testcase_id)
        read_testcase_end_timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
    
        if flag:
            add_request_process_log(requested_on, "read_test_requirement", read_testcase_start_timestamp, read_testcase_end_timestamp, "Passed", "")
        else:
            add_request_process_log(requested_on, "read_test_requirement", read_testcase_start_timestamp, read_testcase_end_timestamp, "Failed", str(test_description))
            return False, "Read Requirement stage Failed"

        if test_type.lower() == "bdd":
            prompt_template = create_bdd_template(testcase_id, test_description, testcases, test_scripting_language)
            prompt = escape_single_braces(prompt_template).format(testcases=testcases)
            message = [{"role": "user", "content": prompt}]
        
            llm_start_timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
            flag, response = azure_openai_request(message, 0)
            llm_end_timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
        
            if flag:
                testscript = response.json()['choices'][0]['message']['content']
                add_llm_log(requested_on, str(message), llm_start_timestamp, llm_end_timestamp, str(response.text), str(response.status_code), "")
                add_request_process_log(requested_on, "llm_calls_1", llm_start_timestamp, llm_end_timestamp, "Passed", "")
            else: 
                add_llm_log(requested_on, str(message), llm_start_timestamp, llm_end_timestamp, response, response.get("Status Code", None), f"Failed_testcase_retry : {response}")
                add_request_process_log(requested_on, "llm_calls_1", llm_start_timestamp, llm_end_timestamp, "Failed", str(response))
                write_comment(domain_or_org, testcase_id, f"Unable to create Automated Test Script, Contact GenAI InsightQA Admin : {testcase_id}")
                return False, "LLM call stage failed"
            feature_file, report_testscript = extract_files(testscript)
        else:
            prompt_template = create_template_1(testcase_id, test_description, testcases, test_scripting_language)
            prompt = prompt_template.format(testcases=testcases)
            message = [{"role": "user", "content": prompt}]
        
            llm_start_timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
            flag, response = azure_openai_request(message, 0)
            llm_end_timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
        
            if flag:
                testscript = response.json()['choices'][0]['message']['content']
                add_llm_log(requested_on, str(message), llm_start_timestamp, llm_end_timestamp, str(response.text), str(response.status_code), "")
                add_request_process_log(requested_on, "llm_calls_1", llm_start_timestamp, llm_end_timestamp, "Passed", "")
            else: 
                add_llm_log(requested_on, str(message), llm_start_timestamp, llm_end_timestamp, response, response.get("Status Code", None), f"Failed_testcase_retry : {response}")
                add_request_process_log(requested_on, "llm_calls_1", llm_start_timestamp, llm_end_timestamp, "Failed", str(response))
                write_comment(domain_or_org, testcase_id, f"Unable to create Automated Test Script, Contact GenAI InsightQA Admin : {testcase_id}")
                return False, "LLM call stage failed"
        
            prompt_template_2 = create_template_2(test_scripting_language, testscript)    
            prompt = escape_single_braces(prompt_template_2).format(testscript=testscript)
            message = [{"role": "user", "content": prompt}]
        
            llm_start_timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
            flag, response = azure_openai_request(message, 0)
            llm_end_timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
        
            if flag:
                report_testscript = response.json()['choices'][0]['message']['content']
                add_llm_log(requested_on, str(message), llm_start_timestamp, llm_end_timestamp, str(response.text), str(response.status_code), "")
                add_request_process_log(requested_on, "llm_calls_2", llm_start_timestamp, llm_end_timestamp, "Passed", "")
            else: 
                add_llm_log(requested_on, str(message), llm_start_timestamp, llm_end_timestamp, response, response.get("Status Code", None), f"Failed_testcase_retry : {response}")
                add_request_process_log(requested_on, "llm_calls_2", llm_start_timestamp, llm_end_timestamp, "Failed", str(response))
                write_comment(domain_or_org, testcase_id, f"Unable to create Automated Test Script, Contact GenAI InsightQA Admin : {testcase_id}")
                return False, "LLM call stage failed"

            validate_start_timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
            validate_flag = validate_structure(report_testscript, test_scripting_language)
            validate_end_timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
            print("validate_flag", validate_flag)
            if validate_flag:
                add_request_process_log(requested_on, "automation_script_validation", validate_start_timestamp, validate_end_timestamp, "Passed", "")
            else:
                add_request_process_log(requested_on, "automation_script_validation", validate_start_timestamp, validate_end_timestamp, "Failed", "Validation Failed")
                write_comment(domain_or_org, testcase_id, f"Unable to create Automated Test Script for requirement ,Please trigger again or Contact GenAI InsighQA Admin : {testcase_id}")
                return False, "Test Script validation failed"
        
        if test_scripting_language.lower() == "java":
            extention = "java"
        elif test_scripting_language.lower() == "python":
            extention = "py"  
        else:
            extention = "cs"
            
        filename = f"{testcase_id}.{extention}".replace("-", "_")
        if test_type.lower() == "bdd":
            feature_filename = f"{testcase_id}.feature"
    
        if requirement_id:
            branch_name = (str(requirement_id) + "_testcase").replace("-", "_")
        else:
            flag, name = get_linked_issue(domain_or_org, testcase_id)
            if not flag:
                return flag, name
            branch_name = (str(name) + "_testcase").replace("-", "_")
        start_time = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
        flag, response = commit_files_to_branch(github_repo, branch_name, filename, report_testscript, "Added by GenAI")
        if test_type.lower() == "bdd":
            flag, response = commit_files_to_branch(github_repo, branch_name, feature_filename, feature_file, "Added by GenAI")
        end_time = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
    
        if flag:
            add_request_process_log(requested_on, "api_calls_generate_script", start_time, end_time, "passed", "")
        else:
            add_request_process_log(requested_on, "api_calls_generate_script", start_time, end_time, "Failed", str(response))
            flag, response = write_comment(domain_or_org, testcase_id, f"Unable to create Automated Test Script for requirement , Contact GenAI InsighQA Admin : {testcase_id}")
            return False, "API call to generate script failed"
    
        comment_start_time = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
        if tool == "JIRA":
            flag, response = write_comment_with_link(domain_or_org, project, testcase_id, "Automation Script committed successfully\nAutomation Script GitHub Link : ", f"https://github.com/{github_repo}/blob/{branch_name}/{filename}")
        elif tool == "ADO":
            flag, response = ado_write_comment_with_link(domain_or_org, project, testcase_id, "Automation Script committed successfully", f"https://github.com/{github_repo}/blob/{branch_name}/{filename}")
        
        comment_end_time = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
    
        if flag:
            add_request_process_log(requested_on, "api_calls_review_comment", comment_start_time, comment_end_time, "Passed", "")
        else: 
            add_request_process_log(requested_on, "api_calls_review_comment", comment_start_time, comment_end_time, "Failed", str(response))
            return False, "API calls to add review comment stage failed"   
        return True, ""
    except Exception as e:
        if tool == "JIRA":
            write_comment(domain_or_org, requirement_id, "Unable to generate Automation Test Case for requirement, contact GenAI InsighQA Admin : " + str(requirement_id))
            print(f"5 flag name : {response} error: {e}")
        elif tool == "ADO":
            ado_write_comment_with_format(domain_or_org, project, requirement_id, "Unable to generate Automation Test Case for requirement, contact GenAI InsighQA Admin : " + str(requirement_id))
        return False, "Unable to create Automated Testscript, contact GenAI InsightQA for error resolution"

def create_template_1(testcase_id, test_description, testcases, test_scripting_language):
    if test_scripting_language.lower() == "java":
        prompt_template = """You are tasked with generating Selenium test scripts.Read and understand the Test_Case to Generate Selenium test script for the given test case.
            Selenium script must follow below requirements:
            Note: Do not provide any comments and suggestions.Only provide selenium scripts.
            Include all these at the start of the script:
            - package com.gsk.torchbearer.sanity_tests;
            - import com.gsk.torchbearer.lib.BaseTest;
            - import com.gsk.torchbearer.model.TestEvidence;
            - import com.gsk.torchbearer.model.TestStatus;
            - import org.testng.annotations.Test;
            - import org.openqa.selenium.By;
            - '@Test(groups = {{"Web"}})' just before class name declaration
            
            1. Language = Java ,  Testing Framework =TestNG
            2. Generate Selenium test script functions for the input testcase.
            3. Class Name to be like "public class  TC01%s extends BaseTest".\
            4. Do not use 'System.setProperty("webdriver.chrome.driver", "<<Enter path to chromedriver>>")' this property code in the output.\
            5. Provide descriptive comment for each function with same approach.\
            6. Use getReport().TestData.Description = %s;\
            7. Use getReport().TestData.Url = ""; for placing the url.\
            8. Provide placeholders for URL , Locators and Assertions like "<< Enter Application URL >>" , ("<<Enter locator value>>"),("<<Add Text Value>>").\
            9. Include TryAssert(() -> Assert.assertTrue(getDriver().getCurrentUrl().contains("<< Enter value>>"))); to verifying web url for that test step,if applicable to that test case.\
            10. getReport() should be used only twice as stated above as getReport().TestData.Description = %s; and getReport().TestData.Url = ""; for placing the url.Do not use it multiple times on your own.
            11. @Test before function name should not be there.
            12. Following all instructions is mandatory.

            Output format: Please provide the output as Java code for the Selenium test script. No comment/suggestions.
            Test_Case : '''%s'''.
        """ % (testcase_id, test_description, test_description, testcases)
        
    elif test_scripting_language.lower() == "python":
        prompt_template = """You are tasked with generating Selenium test scripts.Read and understand the test case to Generate Selenium test script for the input test case.
            Selenium script must follow below requirements:
            Include all these at the start of the script:
            -import pytest
            -from selenium.webdriver.common.by import By
            -from lib.BaseTest import BaseTest
            -from lib.model.TestEvidence import TestEvidence
            -from lib.model.TestStatus import TestStatus
            -from lib.model.TestType import TestType
            -"@pytest.mark.Web"
            
            1. Language = Python ,  Testing Framework = PyTest
            2. Generate Selenium test script functions for the input test case for all the steps.                                
            3. Class Name to be like "class TestTc01%s(BaseTest):" .
            4. Provide descriptive comment for each function with same approach.
            5. Include "BaseTest.report.TestData.Description = %s" .
            6. Include "BaseTest.report.TestData.Url = "" "  for placing url  .\
            7. Use driver.find_element(By.XPATH,"<<Enter locator value>>") wherever required.
            8. Use assert "<< Enter Application URL >>" in self.driver.current_url
            9. Provide placeholders for URL , Locators and Assertions like "<< Enter Application URL >>" , ("<<Enter locator value>>"),("<<Add Text Value>>").
            10.Following all instructions is mandatory.\
            
            Output format: Please provide the output as Python code for the Selenium test script. No comment/suggestions.
            Test_Case : '''%s'''.
        """ % (testcase_id,test_description,testcases)
    else:
        prompt_template = """ You are tasked with generating Selenium test scripts. Read and understand the Test_Case to Generate Selenium test script for the given test case.
            Selenium script must follow the below requirements:
            Note: Do not provide any comments and suggestions. Only provide Selenium scripts.
            Include all these at the start of the script:
            - using OpenQA.Selenium;
            - using NUnit.Framework;
            - using torchbearer_sdk;
            - using torchbearer.lib.model;
                        1. Language = C#, Testing Framework = NUnit
            2. Generate Selenium test script functions for the input test case.
            3. Class Name to be like "public class TC_%s: BaseTest".
            4. Provide descriptive comments for each function with the same approach.
            5. Include "Report.TestData.Description = %s" .
            6. Include "Report.TestData.Url = "" "  for placing url  .\
            7. Use Driver.FindElement(By.XPath("<<Enter locator value>>")) wherever
                required.
            8. Use TestContext.WriteLine() to print the test step details.
            9. Use Assert.AreEqual() to verify the expected and actual results.
            10. Provide placeholders for URL, Locators, and Assertions like "<< Enter Application URL >>", ("<< Enter locator value >>"),                      ("<< Add Text Value >>").
            11. Include [Test, TestOf(nameof(TestType.Web))] attribute before each test function.
            12. Uses appropriate waits and synchronization
            13. Following all instructions is mandatory.
            For each alternative implementation, explain:
            - Why this approach might be preferred
            - When to use it over other options
            - Any tradeoffs or limitations
            Format the output with:
            a. Test Case Description
            b. Primary Implementation (most straightforward)
            c. Alternative Implementations (with explanations)
            d. Complete boilerplate code (namespaces, driver setup, etc.)
            Include these Selenium C# capabilities in alternatives where applicable:
            - Different locator strategies (ID, XPath, CSS, etc.)
            - Explicit vs implicit waits
            - Page Object Model pattern
            - Actions API for complex interactions
            - JavaScript execution
            - Screenshot on failure
            - Data-driven testing approaches
            - Browser configuration options
            - Handling iframes/alerts/windows
            - Custom expected conditions
            Output format: Please provide the output as C# code for the Selenium test script. No comment/suggestions.
                        Test_Case : '''%s'''.
        """ % (testcase_id, test_description, testcases)
        
    return prompt_template

def create_template_2(test_scripting_language, testscript):
    if test_scripting_language.lower() == "java":
        prompt_template = """Selenium Java Testscript  : '''%s'''
            Note: Do not provide any comments and suggestions.Only provide selenium scripts.
            
            1. You have to rewrite this Selenium Java Testscript to Include a call to reusable function "AddEvidence(new TestEvidence())"
            
            
            2. getReport().AddEvidence(new TestEvidence() includes the following variables within curly brackets these should be included in output testscript on separate lines.
                - "Expected" = which contains expected result of testcase step in words
                - "Actual" = which contains actual result of testcase step in words
                - "StepStatus" = TestStatus.Passed           
                - "Details" = which contains testcase goal
                - "Screenshot" =  GetScreenshot(); 
                - "StepName" = which contains the testcase step
                - "TestType" = com.gsk.torchbearer.model.TestType.Web;

            3. For example the sample output for getReport().AddEvidence(new TestEvidence()
                getReport().AddEvidence(new TestEvidence()
                
                    Expected = which contains expected result of testcase step in words
                    Actual = which contains actual result of testcase step in words
                    StepStatus = TestStatus.Passed
                    Details = which contains testcase goal
                    Screenshot = GetScreenshot();
                    StepName = which contains the testcase step
                    TestType = com.gsk.torchbearer.model.TestType.Web;
                );
            Code should have double curly brackets after getReport().AddEvidence(new TestEvidence()
            4. getReport() should be called only the number of times specified here.Do not add getReport() function calls on your own.
            5. Add getReport().AddEvidence(new TestEvidence() structure should be repeated for every step.
        """ % (testscript)
    
    elif test_scripting_language.lower() == "python":
        prompt_template = """Selenium Python Testscript  : '''%s'''
            Note: Do not provide any comments and suggestions.Only provide selenium scripts.
            1. You have to rewrite this Selenium Python Testscript to Include a call to reusable function "BaseTest.report.add_evidence(evidence)"
            
                
            2.  Where evidence = TestEvidence()
                evidence.TestType = Specifies the type of testcase step ; TestType.Web for testing web application.
                evidence.Expected = contains the expected result of testcase step in words
                evidence.Actual = contains the actual result of testcase step in words
                evidence.StepStatus = TestStatus.Passed
                evidence.Details = which contains testcase goal
                evidence.Screenshot = BaseTest.get_screenshot()
                evidence.StepName = contains the testcase step name with complete details
                BaseTest.report.add_evidence(evidence)
                This structure of evidence = TestEvidence() should be repeated for every step. 
            3. Following all instructions is mandatory.    

        """ % (testscript)
    else:
        prompt_template = """Selenium C# Testscript: %s
            Note: Do not provide or add any comments and suggestions or add Selenium C# Testscript/quotes. Only provide Selenium scripts.
            1. You have to rewrite this Selenium C# Testscript to include a call to a reusable function "AddEvidence(new TestEvidence())" for every step in the script.

            2. Report.AddEvidence(new TestEvidence()) includes the following variables within curly brackets that should be included in the output test script on separate lines:
                - "Expected" = which contains the expected result of the test case step in words
                - "Actual" = which contains the actual result of the test case step in words
                - "StepStatus" = TestStatus.Passed          
                - "Details" = which contains the test case goal
                - "Screenshot" =  GetScreenshot();
                - "StepName" = which contains the test case step
                - "TestType" = TestType.Web;
            3. For example, the sample output for AddEvidence(new TestEvidence()):
                Report.AddEvidence(new TestEvidence()
                    Expected = which contains the expected result of the test case step in words
                    Actual = which contains the actual result of the test case step in words
                    StepStatus = TestStatus.Passed
                    Details = which contains the test case goal
                    Screenshot = GetScreenshot();
                    StepName = which contains the test case step
                    TestType = TestType.Web;
                );
            Code should have double curly brackets after AddEvidence(new TestEvidence()).
            4. It is mandatory to add Report.AddEvidence(new TestEvidence()) structure for every step or every line in Test Script.

            5. Count number of steps and Report.AddEvidence(new TestEvidence()) in Test Script. Number of steps should be equal to number of Report.AddEvidence(new TestEvidence()).
            
        """ %(testscript)
    return prompt_template
def create_bdd_template(testcase_id, test_description, testcases, test_scripting_language):
    if test_scripting_language.lower() == 'java':
        prompt = f"""Given a functional scenario as input,
            My scenario input: {testcases}

            write a cucumber feature file with both positive and negative test cases, using proper Gherkin syntax.

            Follow the below format for generating the Cucumber Feature file.
            Generate the Cucumber feature file in Gherkin Syntax starting with the line:
            ### Cucumber feature file
            The feature file should include:

            A clear feature: title and a business-readable description
            A background: section with common steps.
            Minimum of 3 scenario: blocks covering both positive and negative cases.
            Proper use of Given/When/Then/And keywords
            DO NOT include ``` at beginning and end of feature file.

            Also, generate the corresponding Selenium TestNG step definition file in Java. Ensure the implementation follows clean code practices, uses parameterization, and is suitable for enterprise- grade UI automation frameworks. The Format should follow the below steps:
            1. Map gherkin steps to Java methods using appropriate annotations like @Given, @When, @Then
            2. Ensure each method contains basic implemetation 
            3. Use readable method names derived from step text
            4. Include necessary imports and class/module declaration. It is mandatory to include exactly the following import statements at the beginning of the script:
                - package com.gsk.torchbearer.sanity_tests;
                - import com.gsk.torchbearer.lib.BaseTest;
                - import com.gsk.torchbearer.model.TestEvidence;
                - import com.gsk.torchbearer.model.TestStatus;
                - import org.testng.annotations.Test;
                - import org.openqa.selenium.By; 
            5. Ensure all methods are clean, readable, and prepared for reuse across scenarios.
            6. Start file with a comment: ### Step Definition File.

            7. Include a call to reusable function "AddEvidence(new TestEvidence())"

            8. getReport().AddEvidence(new TestEvidence() includes the following variables within curly brackets these should be included in output testscript on separate lines.
                            - "Expected" = which contains expected result of testcase step in words
                            - "Actual" = which contains actual result of testcase step in words
                            - "StepStatus" = TestStatus.Passed           
                            - "Details" = which contains testcase goal
                            - "Screenshot" =  GetScreenshot(); 
                            - "StepName" = which contains the testcase step
                            - "TestType" = com.gsk.torchbearer.model.TestType.Web;
            
            9. For example the sample output for getReport().AddEvidence(new TestEvidence()
                            getReport().AddEvidence(new TestEvidence()
                                Expected = which contains expected result of testcase step in words
                                Actual = which contains actual result of testcase step in words
                                StepStatus = TestStatus.Passed
                                Details = which contains testcase goal
                                Screenshot = GetScreenshot();
                                StepName = which contains the testcase step
                                TestType = com.gsk.torchbearer.model.TestType.Web;
                            );

            10. getReport() should be called only the number of times specified here.Do not add getReport() function calls on your own.
            11. Add getReport().AddEvidence(new TestEvidence() structure should be repeated for every step.
            12. Provide placeholders for URL , Locators and Assertions like "<< Enter Application URL >>", ("<<Enter locator value>>"), ("<<Add Text Value>>").\
            13. Class Name to be like "public class  TC01{testcase_id} extends BaseTest".\
            14. Provide descriptive comment for each function with same approach.\
            15. Use getReport().TestData.Description = {test_description};\
            16. Use getReport().TestData.Url = ""; for placing the url.\
            17. It is mandatory to include exactly the following import statements at the beginning of the script:
                - package com.gsk.torchbearer.sanity_tests;
                - import com.gsk.torchbearer.lib.BaseTest;
                - import com.gsk.torchbearer.model.TestEvidence;
                - import com.gsk.torchbearer.model.TestStatus;
                - import org.testng.annotations.Test;
                - import org.openqa.selenium.By; 
                - '@Test(groups = {{"Web"}})' just before class name declaration
            18. Map gherkin steps to Java methods using appropriate annotations like @Given, @When, @Then. 
            Note: Do not provide any comments and suggestions. Only provide Selenium scripts.
            """
            elif test_scripting_language.lower() == "python":
        prompt = f"""Given a functional scenario as input,
            My scenario input: {testcases}

            write a cucumber feature file with both positive and negative test cases, using proper Gherkin syntax.

            Follow the below format for generating the Cucumber Feature file.
            Generate the Cucumber feature file in Gherkin Syntax starting with the line:
            ### Cucumber feature file
            The feature file should include:

            A clear feature: title and a business-readable description
            A background: section with common steps.
            Minimum of 3 scenario: blocks covering both positive and negative cases.
            Proper use of Given/When/Then/And keywords
            DO NOT include ''' at beginning and end of feature file.

            Also, generate the corresponding Selenium PyTest step definition file in python. Ensure the implementation follows clean code practices, uses parameterization, and is suitable for enterprise- grade UI automation frameworks. The Format should follow the below steps:
            1. Map gherkin steps to Python methods using appropriate annotations like @Given, @When, @Then.
            2. Ensure each method contains basic implemetation 
            3. Use readable methid names derived from step text
            4. Include necessary imports ans class/module declaration.
            5. Ensure all methods are clean, readable, and prepared for reuse across scenarios.
            6. Start file with a comment: ### Step Definition File.
            7. Include a call to reusable function "BaseTest.report.add_evidence(evidence)"
            8.  Where evidence = TestEvidence()
                            evidence.TestType = Specifies the type of testcase step ; TestType.Web for testing web application.
                            evidence.Expected = contains the expected result of testcase step in words
                            evidence.Actual = contains the actual result of testcase step in words
                            evidence.StepStatus = TestStatus.Passed
                            evidence.Details = which contains testcase goal
                            evidence.Screenshot = BaseTest.get_screenshot()
                            evidence.StepName = contains the testcase step name with complete details
                            BaseTest.report.add_evidence(evidence)
                            This structure of evidence = TestEvidence() should be repeated for every step. 
            9. Following all instructions is mandatory.
            10. Class Name to be like "class TestTc01{testcase_id}(BaseTest):" .
            11. Provide descriptive comment for each function with same approach.
            12. Include "BaseTest.report.TestData.Description = {test_description}" .
            13. Include "BaseTest.report.TestData.Url = "" "  for placing url  .\
            14. Use driver.find_element(By.XPATH,"<<Enter locator value>>") wherever required.
            15. Provide placeholders for URL , Locators and Assertions like "<< Enter Application URL >>" , ("<<Enter locator value>>"),("<<Add Text Value>>").
            16. It is mandatory to include exactly the following import statements at the beginning of the script:
                -import pytest
                -from selenium.webdriver.common.by import By
                -from lib.BaseTest import BaseTest
                -from lib.model.TestEvidence import TestEvidence
                -from lib.model.TestStatus import TestStatus
                -from lib.model.TestType import TestType
                -"@pytest.mark.Web"
            17. Map gherkin steps to Python methods using appropriate annotations like @Given, @When, @Then.
            Note: Do not provide any comments and suggestions. Please follow this strictly. Only provide Selenium scripts.
            """
    else:
        prompt = f"""Given a functional scenario as input,
            My scenario input: {testcases}

            write a cucumber feature file with both positive and negative test cases, using proper Gherkin syntax.

            Follow the below format for generating the Cucumber Feature file.
            Generate the Cucumber feature file in Gherkin Syntax starting with the line:
            ### Cucumber feature file
            The feature file should include:

            A clear feature: title and a business-readable description
            A background: section with common steps.
            Minimum of 3 scenario: blocks covering both positive and negative cases.
            Proper use of Given/When/Then/And keywords
            DO NOT include ''' at beginning and end of feature file.

            Also, generate the corresponding Selenium NUnit step definition file in C#. Ensure the implementation follows clean code practices, uses parameterization, and is suitable for enterprise- grade UI automation frameworks. The Format should follow the below steps:
            1. Map gherkin steps to C# methods using appropriate annotations like @Given, @When, @Then.
            2. Ensure each method contains basic implemetation 
            3. Use readable methid names derived from step text
            4. Include necessary imports ans class/module declaration.
            5. Ensure all methods are clean, readable, and prepared for reuse across scenarios.
            6. Start file with a comment: ### Step Definition File.

            7. Include a call to a reusable function "AddEvidence(new TestEvidence())" for every step in the script.
            
            8. Report.AddEvidence(new TestEvidence()) includes the following variables within curly brackets that should be included in the output test script on separate lines:
                            - "Expected" = which contains the expected result of the test case step in words
                            - "Actual" = which contains the actual result of the test case step in words
                            - "StepStatus" = TestStatus.Passed          
                            - "Details" = which contains the test case goal
                            - "Screenshot" =  GetScreenshot();
                            - "StepName" = which contains the test case step
                            - "TestType" = TestType.Web;
            9. For example, the sample output for AddEvidence(new TestEvidence()):
                            Report.AddEvidence(new TestEvidence()
                                Expected = which contains the expected result of the test case step in words
                                Actual = which contains the actual result of the test case step in words
                                StepStatus = TestStatus.Passed
                                Details = which contains the test case goal
                                Screenshot = GetScreenshot();
                                StepName = which contains the test case step
                                TestType = TestType.Web;
                            );
                       Code should have double curly brackets after AddEvidence(new TestEvidence()).
            4. It is mandatory to add Report.AddEvidence(new TestEvidence()) structure for every step or every line in Test Script.
            5. Generate Selenium test script functions for the input test case.
            6. Class Name to be like "public class TC_{testcase_id}: BaseTest".
            7. Provide descriptive comments for each function with the same approach.
            8. Include "Report.TestData.Description = {test_description}" .
            9. Include "Report.TestData.Url = "" "  for placing url  .\
            10. Use Driver.FindElement(By.XPath("<<Enter locator value>>")) wherever
                required.
            11. Use TestContext.WriteLine() to print the test step details.
            12. Use Assert.AreEqual() to verify the expected and actual results.
            13. Provide placeholders for URL, Locators, and Assertions like "<< Enter Application URL >>", ("<< Enter locator value >>"),                      ("<< Add Text Value >>").
            14. Include [Test, TestOf(nameof(TestType.Web))] attribute before each test function.
            15. Uses appropriate waits and synchronization
            16. Following all instructions is mandatory.
            17. It is mandatory to include all these statement at the start of the script:
                - using OpenQA.Selenium;
                - using NUnit.Framework;
                - using torchbearer_sdk;
                - using torchbearer.lib.model;
            18. Map gherkin steps to C# methods using appropriate annotations like @Given, @When, @Then.
            Note: Do not provide any comments and suggestions. Please follow this strictly. Only provide Selenium scripts.
            """                
    return prompt

def escape_single_braces(string):
    return re.sub(r'(?<!})}(?!})', '}}', re.sub(r'(?<!{){(?!{)', '{{', string))

def get_linked_issue(domain_or_org, testcase_id):
    flag, response = read_jira_issue(domain_or_org, testcase_id)
    if not flag:
        return flag, response

    branch_name = "GIQ_Automation"

    field_issuelinks = response["fields"]["issuelinks"]
    if len(field_issuelinks) == 1:
        issue_type = field_issuelinks[0]["type"]
        if issue_type["outward"] =='tests':
            branch_name = field_issuelinks[0]["outwardIssue"]["key"]
    return True, branch_name

def validate_structure(script, language):
    using_pattern = re.compile(r'\busing\s+\w+')
    from_pattern = re.compile(r'\bfrom\s+\w+')
    import_pattern = re.compile(r'\bimport\s+\w+')
    package_pattern = re.compile(r'\bpackage\s+\w+')
    import_pattern = re.compile(r'\bimport\s+\w+')

    if language.lower() == "c#":
        if not using_pattern.search(script):
            print("Error: Missing using declaration in C#.")
            return False
    elif language.lower() == "python":
        if not from_pattern.search(script) or not import_pattern.search(script):
            print("Error: Missing either from/import declaration in Python.")
            return False
    else:
        if not package_pattern.search(script) or not import_pattern.search(script):
            print("Error: Missing either package/import declaration in Java.")
            return False
    return True

def extract_files(content):
    feature_file = []
    step_definition_file = []
    is_feature_section = False
    is_step_definition_section = False

    for line in content.splitlines():
        if line.strip() == "### Cucumber feature file":
            is_feature_section = True
            is_step_definition_section = False
            continue
        if line.strip() == "### Step Definition File":
            is_feature_section = False
            is_step_definition_section = True
            continue

        if is_feature_section:
            feature_file.append(line)
        elif is_step_definition_section:
            step_definition_file.append(line)

    return "\n".join(feature_file).replace("```gherkin", "").replace("```", "").strip(), "\n".join(step_definition_file).replace("```java", "").replace("```", "").strip()
