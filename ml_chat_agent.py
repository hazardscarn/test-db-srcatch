import sqlite3
import os
from sqldb import SqlDatabase
import google.generativeai as genai
import google.ai.generativelanguage as glm
import pandas as pd
import numpy as np
import re


gmodel=genai.GenerativeModel(model_name="gemini-1.5-flash-001")

sql_agent=SqlDatabase(db_name="telecom_churn", db_path="sqldatabase")


def remove_sql_and_backticks(input_text):
    """
    Removes 'sql' and '```' from the input string.

    Parameters
    ----------
        input_text : str
            the input string

    Returns
    -------
        str
            the modified string
    """
    modified_text = re.sub(r'```|sql', '', input_text)
    modified_text = re.sub(r'\\\\', '', input_text)
    return modified_text




def generate_sql(user_question:str):
    """
    Generates the SQL query based on the user question
    Use this function to create a SQL query to retrive any data user have asked for or to create an answer to user question.

    Parameters
    ----------
        user_question : str
            the user question
    Returns
    -------
        str
            the result sql query generated
    """

    context_prompt = f"""
            You are a Sqlite SQL guru. Write a SQL comformant query for Sqlite that answers the following question while using the provided context to correctly refer to the SQlite tables and the needed column names.

            Guidelines:
            - Join as minimal tables as possible.
            - When joining tables ensure all join columns are the same data_type.
            - Analyze the database and the table schema provided as parameters and undestand the relations (column and table relations).
            - Don't include any comments in code.
            - **Remove ```sql and ``` from the output and generate the SQL in single line.**
            - Tables should be refered to using a fully qualified name with enclosed in ticks (`) e.g. `project_id.owner.table_name`.
            - Use all the non-aggregated columns from the "SELECT" statement while framing "GROUP BY" block.
            - Use ONLY the column names (column_name) mentioned in Table Schema. DO NOT USE any other column names outside of this.
            - Associate column_name mentioned in Table Schema only to the table_name specified under Table Schema.
            - Table names are case sensitive. DO NOT uppercase or lowercase the table names.
            - Always enclose subqueries and union queries in brackets.
            - Refer to the examples provided below, if given.
            - If the question is not clear, answer with : {"Sorry. No information available for this question."}
            - If the question have task to create summary, create the appropritae summary logic
            - Only answer questions relevant to the tables or columns listed in the table schema If a non-related question comes, answer exactly with : {"Sorry. No information available for this question."}
            - When question is about average feature contribution to churn, use customer_shap_contributions columns only in the output unless it's necessary

            Below are descriptions for the tables in the database:

            customer_data : 
                - Contains all available customer data points and their churn prediction
                - Donot use this data for SHAP contribution except for adding filters
            data_dictionary: 
                - Contains the description of all the columns in the customer_data table
            customer_shap_contributions : 
                - Contains the individual feature contribution towards every customers churn prediction in SHAP scale.
                - High positive SHAP value indicates higher probability of churn and vice versa.
                - Can be used to explain a churn prediction of a customer
            customer_counterfactuals : 
                - Contains the counterfactuals generated for every customer. Can be used to provide recommendations to a customer to reduce churn
            
            global_shap_summary: Contains the global shap summary across Feature and it's subgroups. This helps to identify the most important features in the model and top contributors to churn prediction.
                                Remember to use *Group*, Feature and Importance Rank here for any analysis.
            

            Here are some examples of user-question and SQL queries:
            Q: What is the average monthly charges for customers who have churned?
            A: SELECT AVG(monthly_charges) from customer_data where churn=1;

            Q:Tell me about customerID 3114822?
            A: SELECT * from customer_data where customerID=3114822;

            Q:Tell me SHAP contribution of customerID 3114822?
            A: SELECT * from customer_shap_contributions where customerID=3114822;

            Q:Tell me some action recommendations for customerID 3114822?
            A: SELECT * from customer_counterfactuals where customerID=3114822;

            Q:What are the top contributors to churn?
            A:SELECT `global_shap_summary`.Feature,`global_shap_summary`.`Group`, `global_shap_summary`.`Probability Change (%)`,`global_shap_summary`.`Importance Rank`
            FROM `global_shap_summary` ORDER BY `global_shap_summary`.`Importance Rank` ASC

            Q:Give a summary of shap contribution of age and revenue customers in service city hou
            A:  SELECT AVG(`customer_shap_contributions`.agehh1),
                AVG(`customer_shap_contributions`.revenue_per_minute)
                FROM `customer_shap_contributions` 
                JOIN `customer_data` ON
                `customer_shap_contributions`.customerid = `customer_data`.customerid 
                WHERE `customer_data`.service_city = 'hou'

            question:
            {user_question}

            Table Schema:
            Tables:

                customer_data
                customer_shap_contributions
                customer_counterfactuals
                data_dictionary
                global_shap_summary


                Schemas:


                customer_data:
                Name: childreninhh, Type: TEXT
                Name: handsetrefurbished, Type: TEXT
                Name: handsetwebcapable, Type: TEXT
                Name: truckowner, Type: TEXT
                Name: rvowner, Type: TEXT
                Name: homeownership, Type: TEXT
                Name: buysviamailorder, Type: TEXT
                Name: respondstomailoffers, Type: TEXT
                Name: optoutmailings, Type: TEXT
                Name: nonustravel, Type: TEXT
                Name: ownscomputer, Type: TEXT
                Name: hascreditcard, Type: TEXT
                Name: newcellphoneuser, Type: TEXT
                Name: notnewcellphoneuser, Type: TEXT
                Name: ownsmotorcycle, Type: TEXT
                Name: madecalltoretentionteam, Type: TEXT
                Name: creditrating, Type: TEXT
                Name: prizmcode, Type: TEXT
                Name: occupation, Type: TEXT
                Name: maritalstatus, Type: TEXT
                Name: service_city, Type: TEXT
                Name: monthlyrevenue, Type: REAL
                Name: monthlyminutes, Type: REAL
                Name: totalrecurringcharge, Type: REAL
                Name: directorassistedcalls, Type: REAL
                Name: overageminutes, Type: REAL
                Name: roamingcalls, Type: REAL
                Name: percchangeminutes, Type: REAL
                Name: percchangerevenues, Type: REAL
                Name: droppedcalls, Type: REAL
                Name: blockedcalls, Type: REAL
                Name: unansweredcalls, Type: REAL
                Name: customercarecalls, Type: REAL
                Name: threewaycalls, Type: REAL
                Name: receivedcalls, Type: REAL
                Name: outboundcalls, Type: REAL
                Name: inboundcalls, Type: REAL
                Name: peakcallsinout, Type: REAL
                Name: offpeakcallsinout, Type: REAL
                Name: droppedblockedcalls, Type: REAL
                Name: callforwardingcalls, Type: REAL
                Name: callwaitingcalls, Type: REAL
                Name: monthsinservice, Type: REAL
                Name: uniquesubs, Type: REAL
                Name: activesubs, Type: REAL
                Name: handsets, Type: REAL
                Name: handsetmodels, Type: REAL
                Name: currentequipmentdays, Type: REAL
                Name: agehh1, Type: REAL
                Name: retentioncalls, Type: REAL
                Name: retentionoffersaccepted, Type: REAL
                Name: referralsmadebysubscriber, Type: REAL
                Name: adjustmentstocreditrating, Type: REAL
                Name: revenue_per_minute, Type: REAL
                Name: total_calls, Type: REAL
                Name: avg_call_duration, Type: REAL
                Name: service_tenure, Type: REAL
                Name: customer_support_interaction, Type: REAL
                Name: handsetprice, Type: REAL
                Name: incomegroup, Type: REAL
                Name: customerid, Type: REAL
                Name: churn, Type: REAL
                Name: prediction, Type: REAL

                    Distinct Values for Categorical Variables:

                    childreninhh: ['no', 'yes']
                    handsetrefurbished: ['no', 'yes']
                    handsetwebcapable: ['no', 'yes']
                    truckowner: ['yes', 'no']
                    rvowner: ['yes', 'no']
                    homeownership: ['known', 'unknown']
                    buysviamailorder: ['no', 'yes']
                    respondstomailoffers: ['yes', 'no']
                    optoutmailings: ['yes', 'no']
                    nonustravel: ['no', 'yes']
                    ownscomputer: ['no', 'yes']
                    hascreditcard: ['no', 'yes']
                    newcellphoneuser: ['no', 'yes']
                    notnewcellphoneuser: ['no', 'yes']
                    ownsmotorcycle: ['no', 'yes']
                    madecalltoretentionteam: ['no', 'yes']
                    creditrating: ['2-high', '3-good', '4-medium', '5-low', '1-highest', '6-verylow', '7-lowest']
                    prizmcode: ['rural', 'other', 'suburban', 'town']
                    occupation: ['professional', 'other', 'homemaker', 'crafts', 'self', 'clerical', 'retired', 'student']
                    maritalstatus: ['yes', 'no', 'unknown']


                customer_shap_contributions:
                Name: childreninhh, Type: REAL
                Name: handsetrefurbished, Type: REAL
                Name: handsetwebcapable, Type: REAL
                Name: truckowner, Type: REAL
                Name: rvowner, Type: REAL
                Name: homeownership, Type: REAL
                Name: buysviamailorder, Type: REAL
                Name: respondstomailoffers, Type: REAL
                Name: optoutmailings, Type: REAL
                Name: nonustravel, Type: REAL
                Name: ownscomputer, Type: REAL
                Name: hascreditcard, Type: REAL
                Name: newcellphoneuser, Type: REAL
                Name: notnewcellphoneuser, Type: REAL
                Name: ownsmotorcycle, Type: REAL
                Name: madecalltoretentionteam, Type: REAL
                Name: creditrating, Type: REAL
                Name: prizmcode, Type: REAL
                Name: occupation, Type: REAL
                Name: maritalstatus, Type: REAL
                Name: service_city, Type: REAL
                Name: monthlyrevenue, Type: REAL
                Name: monthlyminutes, Type: REAL
                Name: totalrecurringcharge, Type: REAL
                Name: directorassistedcalls, Type: REAL
                Name: overageminutes, Type: REAL
                Name: roamingcalls, Type: REAL
                Name: percchangeminutes, Type: REAL
                Name: percchangerevenues, Type: REAL
                Name: droppedcalls, Type: REAL
                Name: blockedcalls, Type: REAL
                Name: unansweredcalls, Type: REAL
                Name: customercarecalls, Type: REAL
                Name: threewaycalls, Type: REAL
                Name: receivedcalls, Type: REAL
                Name: outboundcalls, Type: REAL
                Name: inboundcalls, Type: REAL
                Name: peakcallsinout, Type: REAL
                Name: offpeakcallsinout, Type: REAL
                Name: droppedblockedcalls, Type: REAL
                Name: callforwardingcalls, Type: REAL
                Name: callwaitingcalls, Type: REAL
                Name: monthsinservice, Type: REAL
                Name: uniquesubs, Type: REAL
                Name: activesubs, Type: REAL
                Name: handsets, Type: REAL
                Name: handsetmodels, Type: REAL
                Name: currentequipmentdays, Type: REAL
                Name: agehh1, Type: REAL
                Name: retentioncalls, Type: REAL
                Name: retentionoffersaccepted, Type: REAL
                Name: referralsmadebysubscriber, Type: REAL
                Name: adjustmentstocreditrating, Type: REAL
                Name: revenue_per_minute, Type: REAL
                Name: total_calls, Type: REAL
                Name: avg_call_duration, Type: REAL
                Name: service_tenure, Type: REAL
                Name: customer_support_interaction, Type: REAL
                Name: handsetprice, Type: REAL
                Name: incomegroup, Type: REAL
                Name: customerid, Type: REAL

                    Distinct Values for Categorical Variables:



                customer_counterfactuals:
                Name: customerid, Type: REAL
                Name: changes, Type: TEXT

                    Distinct Values for Categorical Variables:



                data_dictionary:
                Name: Column Name, Type: TEXT
                Name: Data Type, Type: TEXT
                Name: Description, Type: TEXT

                    Distinct Values for Categorical Variables:

                    Data Type: ['object', 'float64', 'int64']


                global_shap_summary:
                Name: Group, Type: TEXT
                Name: SHAP Value, Type: REAL
                Name: Adjusted Probability, Type: REAL
                Name: Probability Change (%), Type: REAL
                Name: Feature, Type: TEXT
                Name: Feature Importance, Type: REAL
                Name: Importance Rank, Type: REAL

                    Distinct Values for Categorical Variables:



            """
    context_query=gmodel.generate_content(context_prompt, stream=False)
    print(remove_sql_and_backticks(str(context_query.candidates[0].content.parts[0].text)))

    return str(context_query.candidates[0].content.parts[0].text)


def execute_sql(sql_query:str):
    """
    Executes the provided SQL query using the sql_agent and returns the result as a dictionary.

    The function takes a SQL query as input, executes it using the sql_agent, and returns the result as a dictionary.

    Parameters
    ----------
    sql_query : str
        The SQL query to be executed.

    Returns
    -------
    dict
        The result of the SQL query, converted to a dictionary.
    """

    print(sql_query)
    sql_query=remove_sql_and_backticks(sql_query)
    sql_query=sql_query.replace("\\", "")
    print(sql_query)
    rdf=sql_agent.execute_query(query=sql_query)
    print(rdf.to_dict())
    #return f"""retrived answer is : {rdf.to_dict()}"""
	return rdf
    


def global_shap_summary(df_data,df_shap_data):
    """
        Calculates the global SHAP summary for the given data and SHAP values.
        This can be used to idenitfy patterns, top churn contributors and feature importance for the subset of data.
        Requires two inputs:
        ----------
        df_data : pandas.DataFrame
            The customer data of the subset of interest as pandas DataFrame.
        df_shap_data : pandas.DataFrame
            The SHAP feature contribution data for the same customers as a DataFrame.
        
        It is important to have noth input data to be of same size and same order of customers.

    """



    def sigmoid(x):
        """ Sigmoid function to convert log-odds to probabilities. """
        return 1 / (1 + np.exp(-x))
    
    base_value=1.0    
    base_probability = sigmoid(base_value)
    results = []
    feature_importances = {}
    # Process each feature
    common_columns = df_data.columns.intersection(df_shap_data.columns)


    # Calculate feature importances
    for feature in common_columns:
        feature_shap_values = df_shap_data[feature]
        feature_importances[feature] = np.mean(np.abs(feature_shap_values))

    importance_df = pd.DataFrame(list(feature_importances.items()), columns=['Feature', 'Importance'])
    importance_df.sort_values('Importance', ascending=False, inplace=True)
    importance_df['Rank'] = range(1, len(importance_df) + 1)
    importance_ranks = importance_df.set_index('Feature')['Rank'].to_dict()

    for feature in common_columns:
        feature_values = df_data[feature]
        feature_shap_values = df_shap_data[feature]
        df = pd.DataFrame({feature: feature_values, 'SHAP Value': feature_shap_values})
        numeric_features = df_data.select_dtypes(include=['number']).columns

        if feature in numeric_features:
            df['Group'] = pd.qcut(df[feature], 10, duplicates='drop')
        else:
            df['Group'] = df[feature]

        group_avg = df.groupby('Group',observed=True)['SHAP Value'].mean().reset_index()
        group_avg['Adjusted Probability'] = sigmoid(base_value + group_avg['SHAP Value'])
        group_avg['Probability Change (%)'] = (group_avg['Adjusted Probability'] - base_probability) * 100
        group_avg['Feature'] = feature
        group_avg['Feature Importance'] = feature_importances[feature]
        group_avg['Importance Rank'] = importance_ranks[feature]
        results.append(group_avg)
    
    result_df = pd.concat(results, ignore_index=True)
    result_df.sort_values(['Importance Rank', 'Probability Change (%)'], ascending=[True, False], inplace=True)
    return result_df


##Create the Chat Agent
gen_model = genai.GenerativeModel(
    model_name="gemini-1.5-flash-001",
    tools=[generate_sql,execute_sql] # list of all available tools
)
chat = gen_model.start_chat(enable_automatic_function_calling=True)

#print(gen_model._tools.to_proto())


##Check SQL Tool
#df=generate_sql("""What is the averge churn for customer in service city hou""")
#df

response = chat.send_message(""""What is the averge churn for customer in service city hou?""")


# for content in chat.history:
#     part = content.parts[0]
#     print(content.role, "->", type(part).to_dict(part))
#     print('-'*80)