# Databricks notebook source
# MAGIC %run ../reusable/jira

# COMMAND ----------

# MAGIC %run ../reusable/llm

# COMMAND ----------

# MAGIC %run ../reusable/github

# COMMAND ----------

# MAGIC %run ../reusable/ado

# COMMAND ----------

# MAGIC %run ../reusable/log

# COMMAND ----------

# MAGIC %run ../reusable/database

# COMMAND ----------

from databricks.sdk.runtime import *
import datetime, logging
import json, sys
import pytz
import traceback


def review_requirement(tool, domain_or_org, project_key, requirement_id, requested_on):
    try:
        read_requirement_start_timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
        if tool == 'JIRA':
            read_requirement_flag, read_response, requirement_id, summary, description, acceptance_criteria, parent_key, parent_summary, parent_description, missing_fields = read_requirement_details(
                domain_or_org, project_key, requirement_id)
        elif tool == 'ADO':
            read_requirement_flag, read_response, requirement_id, summary, description, acceptance_criteria, parent_key, parent_summary, parent_description, missing_fields = ado_read_requirement_details(
                domain_or_org, project_key, requirement_id)
        read_requirement_end_timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]

        if read_requirement_flag:
            add_request_process_log(requested_on, "read_requirement", read_requirement_start_timestamp,
                                    read_requirement_end_timestamp, "Passed", "")
        else:
            add_request_process_log(requested_on, "read_requirement", read_requirement_start_timestamp,
                                    read_requirement_end_timestamp, "Failed",
                                    str(read_response)); return False, "Read Requirement stage Failed"

        if missing_fields:
            counter = 1
            missing_field_list = []
            if "Description" in missing_fields and "Acceptance Criteria" in missing_fields:
                for field in missing_fields:
                    missing_field_list.append(f"{counter}. {field}\n")
                    counter += 1
                if tool.lower() == 'jira':
                    write_comment(domain_or_org, requirement_id,
                                  f"Unable to process the request as following mandatory details are missing in the User Requirement : \n{''.join(missing_field_list)}Please provide the required missing field values and trigger again.")
                else:
                    ado_write_comment_with_format(domain_or_org, project_key, requirement_id,
                                                  f"Unable to process the request as following mandatory details are missing in the User Requirement : <br>{''.join(missing_field_list)}<br>Please provide the required missing field values and trigger again.")
                return False

        prompt_raw = get_prompt("default", "default", "Requirement Review", "requirement_review")

        prompt = (prompt_raw.replace("{requirement_id}", str(requirement_id))
                    .replace("{parent_summary}",parent_summary)
                    .replace("{parent_description}", parent_description)
                    .replace("{summary}", summary)
                    .replace("{description}",description)
                    .replace("{acceptance_criteria}", acceptance_criteria))

        message = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]

        llm_start_timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
        llm_flag, llm_response = azure_openai_request(message, 0)
        llm_end_timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]

        if llm_flag:
            review_response = llm_response.json()['choices'][0]['message']['content'] + "|"
            add_llm_log(requested_on, str(message), llm_start_timestamp, llm_end_timestamp, str(review_response),
                        str(llm_response.status_code), "")
            add_request_process_log(requested_on, "llm_calls", llm_start_timestamp, llm_end_timestamp, "Passed", "")
        else:
            add_llm_log(requested_on, str(message), llm_start_timestamp, llm_end_timestamp, str(llm_response),
                        llm_response.get("Status Code", None), str(llm_response))
            add_request_process_log(requested_on, "llm_calls", llm_start_timestamp, llm_end_timestamp, "Failed",
                                    str(llm_response))
            return False, "LLM call stage failed"

        post_processing_start_timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
        post_processing_flag, message = True, ""
        try:
            prompt_response_list = review_response.split("|")[1:-1]
            parsed_data_dictionary = {}
            for data in prompt_response_list:
                if data and ":" in data: data_list = data.split(":", 1)
                key, value = data_list[0].strip(), data_list[1].strip()
                parsed_data_dictionary[key] = value
        except Exception as e:
            post_processing_flag, message = False, {"message": f"Post processing : {e}"}
        post_processing_end_timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]

        if post_processing_flag:
            add_request_process_log(requested_on, "post_processing", post_processing_start_timestamp,
                                    post_processing_end_timestamp, "Passed", "")
        else:
            add_request_process_log(requested_on, "post_processing", post_processing_start_timestamp,
                                    post_processing_end_timestamp, "failed", str(message))
            return False, "Post processing stage failed"

        logger.debug(f"Requirement Review Data : {parsed_data_dictionary}")

        scores = json.loads(('''{"Measurability": %s, "Testability": %s, "Clarity": %s, "Completeness": %s}''') % (
            parsed_data_dictionary["Measurability quality score"], parsed_data_dictionary["Testability quality score"],
            parsed_data_dictionary["Clarity quality score"], parsed_data_dictionary["Completeness quality score"]))

        suggestion = f"\n User Story Description : {parsed_data_dictionary.get('Suggestion user story', None)} \n\n User Story Acceptance Criteria :\n {parsed_data_dictionary.get('Suggestion acceptance criteria', None)}"
        review_comment = parsed_data_dictionary['Review Comments'].split(". ")
        if review_comment and 'None' in review_comment: review_comment.pop(); review_comment.append("No review comment")

        # api_calls_review_comment
        review_comment_start_time = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]
        if tool == "JIRA":
            flag, response = add_jira_comment(domain_or_org, requirement_id,
                                              format_review_comment(review_comment, scores, suggestion, missing_fields))
        elif tool == "ADO":
            flag, response = ado_write_comment_with_format(domain_or_org, project_key, requirement_id,
                                                           format_ado_review_comment(review_comment, scores, suggestion,
                                                                                     missing_fields))
        review_comment_end_time = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S_%f")[:-3]

        if flag:
            add_request_process_log(requested_on, "api_calls_review_comment", review_comment_start_time,
                                    review_comment_end_time, "Passed", "")
        else:
            add_request_process_log(requested_on, "api_calls_review_comment", review_comment_start_time,
                                    review_comment_end_time, "Failed", str(response))
            return False, "Write comment stage failed"

        return True, "
    except Exception as e:
        print(traceback.format_exc())
        return False, "Unable to review requirement, contact GenAI InsightQA for error resolution"
