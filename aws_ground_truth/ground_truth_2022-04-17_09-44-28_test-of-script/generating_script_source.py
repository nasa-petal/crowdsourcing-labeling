"""
1. get collection of papers
2. create AWS GT job with labels and instructions and selecting the vendor
3. add papers to the job
"""

import datetime
import inspect
import json
import os
import pathlib
import shutil
import sys

import boto3
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

####################
# Constants
####################
aws_profile = 'GRC-VE00-PeTaL-Lab-mturk-script'
role_arn = 'arn:aws:iam::056730517754:role/ground-truth-sagemaker-execution-role'
# work_team_arn = 'arn:aws:sagemaker:us-east-1:056730517754:workteam/private-crowd/ground-truth-petal-test'
work_team_arn = 'arn:aws:sagemaker:us-east-1:056730517754:workteam/private-crowd/ground-truth-just-herb'

labeling_job_name = 'ground-truth-petal-quadrant-test-4'

level1_v3_taxonomy_labels = [
    'Attach',
    'Move',
    'Protect from harm',
    'Manage mechanical forces',
    'Sustain ecological community',
    'Chemically assemble or break down',
    'Modify or convert energy',
    'Physically assemble/disassemble',
    'Sense, send, or process information',
    'Process resources',
    'Change size or color',
    'Paper is not biomimicry relevant',
    'None of the given labels apply',
    'I am not sure / Skip'
]
num_clusters = 7
num_labeled_papers_to_label = 7
num_not_labeled_papers_to_label = 7
num_workers_per_paper = 3
task_time_limit_in_seconds = 10 * 60
task_availability_in_seconds = 10 * 24 * 60 * 60

ground_truth_manifest_filename = 'ground-truth-quadrant-pilot-1.manifest'
label_category_configuration_file = 'label_category_configuration.json'
s3_bucket_name = 'ground-truth-petal-test'
custom_worker_task_template = 'custom-worker-task-template.html'
comment = 'test of script'

def get_list_of_abstracts_from_papers_using_kmeans(df_papers, num_clusters, num_papers_desired,
                                                   vectorizer):
    # get abstracts from each cluster, sorted by petal ID and take the first
    # num_labeled/num_clusters
    abstracts = df_papers.abstract.tolist()
    X_abstracts = vectorizer.fit_transform(abstracts)
    model_abstracts = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100,
                                     n_init=1)
    model_abstracts.fit(X_abstracts)
    model_clusters = model_abstracts.labels_.tolist()
    petalIDs = df_papers.petalID.tolist()
    level1s = df_papers.level1.tolist()
    papers_dict = {'abstract': abstracts, 'cluster': model_clusters,
                           'petalID': petalIDs,
                           'level1': level1s
                           }

    df_with_clusters = pd.DataFrame(papers_dict,
                                                  index=[model_clusters],
                                                  columns=['abstract', 'cluster', 'petalID',
                                                           'level1'])
    print(df_with_clusters['cluster'].value_counts())

    df_grouped_by_cluster = df_with_clusters.groupby(df_with_clusters['cluster'])

    # gather up papers for labeling
    num_papers_per_cluster_to_label = num_papers_desired // num_clusters
    papers_to_be_labeled = []
    for group in df_grouped_by_cluster.groups.keys():
        df_cluster = df_grouped_by_cluster.get_group(group)
        for paper in df_cluster.head(num_papers_per_cluster_to_label).itertuples():
            papers_to_be_labeled.append(paper)

    return papers_to_be_labeled


####################
# make a directory for all the outputs
####################
timestamp = datetime.datetime.now()
timestamp = timestamp.strftime("%Y-%m-%d_%H-%M-%S")

comment_no_spaces = comment.replace(' ', '-')
dirname = f"ground_truth_{timestamp}_{comment_no_spaces}"
os.mkdir(dirname)

# for reproducibility, save this script
with open(pathlib.Path(dirname).joinpath("generating_script_source.py"),"w") as f:
    f.write(inspect.getsource(sys.modules[__name__]))

definition_dir = pathlib.Path(dirname).joinpath("definition")
os.mkdir(definition_dir)
results_dir = pathlib.Path(dirname).joinpath("results")
os.mkdir(results_dir)


####################
# Get all the AWS sessions and clients needed
####################
session = boto3.Session(profile_name=aws_profile)  # This profile was created using AWS command
                                                   # line tools using access keys
sagemaker = session.client(
    service_name='sagemaker',
    region_name='us-east-1',
)

s3 = session.client(
    service_name='s3',
    region_name='us-east-1',
)

####################
# read in golden.json from github
####################
df = pd.read_json("https://raw.githubusercontent.com/nasa-petal/data-collection-and-prep/main/golden.json")
# df = pd.read_json("golden.json")  # If needed because networking not working

####################
# Do k-means clustering of abstracts
####################
vectorizer = TfidfVectorizer(stop_words='english')
# split into labeled and unlabeled
# Take lowest petal ID numbered documents from each batch and put into the source file
df_labeled_papers = df[df['level1'].astype(bool)]
labeled_papers_to_be_labeled = get_list_of_abstracts_from_papers_using_kmeans(df_labeled_papers,
                                                                              num_clusters,
                                                                              num_labeled_papers_to_label,
                                                                              vectorizer)

df_not_labeled_papers = df[~df['level1'].astype(bool)]
not_labeled_papers_to_be_labeled = get_list_of_abstracts_from_papers_using_kmeans(df_not_labeled_papers,
                                                                              num_clusters,
                                                                              num_not_labeled_papers_to_label,
                                                                              vectorizer)

papers_to_be_labeled = labeled_papers_to_be_labeled + not_labeled_papers_to_be_labeled

####################
# Write manifest file of abstracts to label and upload to S3
####################
with open(pathlib.Path(definition_dir).joinpath(ground_truth_manifest_filename),"w") as f:
    for paper in papers_to_be_labeled:
        item = {'source': paper.abstract}
        f.write(json.dumps(item) + "\n")

s3.upload_file(
    str(pathlib.Path(definition_dir).joinpath(ground_truth_manifest_filename)),
    s3_bucket_name,
    ground_truth_manifest_filename
)

####################
# Write the label category configuration file and upload to S3
####################
label_category_config = {}
label_category_config['document-version'] = 'taxonomy-v3-2022-04-15'
labels_list_of_dicts = []
for label in level1_v3_taxonomy_labels:
    labels_list_of_dicts.append({'label': label})
label_category_config['labels'] = labels_list_of_dicts

with open(pathlib.Path(definition_dir).joinpath(label_category_configuration_file),"w") as f:
    f.write(json.dumps(label_category_config))
s3.upload_file(
    str(pathlib.Path(definition_dir).joinpath(label_category_configuration_file)),
    s3_bucket_name,
    label_category_configuration_file
)

####################
# Upload the template to S3
####################
shutil.copy(custom_worker_task_template, str(pathlib.Path(definition_dir)))
s3.upload_file(
    str(pathlib.Path(definition_dir).joinpath(custom_worker_task_template)),
    s3_bucket_name,
    custom_worker_task_template
)

####################
# make the labeling job in AWS Ground Truth
####################
labeling_job_arguments = {
    'LabelingJobName': labeling_job_name,
    'LabelAttributeName': 'label',
    'InputConfig': {
        'DataSource': {
            'S3DataSource': {
                'ManifestS3Uri': f's3://{s3_bucket_name}/{ground_truth_manifest_filename}'
            }
        },
        'DataAttributes': {
            "ContentClassifiers": [
                "FreeOfPersonallyIdentifiableInformation",
                "FreeOfAdultContent"
            ]
        }
    },
    'OutputConfig': {
        'S3OutputPath': f's3://{s3_bucket_name}',
        # 'KmsKeyId': 'string'
    },
    'RoleArn': role_arn,
    'LabelCategoryConfigS3Uri': f's3://{s3_bucket_name}/{label_category_configuration_file}',
    'StoppingConditions': {
        'MaxHumanLabeledObjectCount': 30,
        'MaxPercentageOfInputDatasetLabeled': 100
    },
    'HumanTaskConfig': {
        'WorkteamArn': work_team_arn,
        'UiConfig': {
            'UiTemplateS3Uri': f's3://{s3_bucket_name}/files-for-jobs/{custom_worker_task_template}'
        },
        'PreHumanTaskLambdaArn': 'arn:aws:lambda:us-east-1:432418664414:function:PRE-TextMultiClassMultiLabel',  # From the docs
        'TaskKeywords': [
            'Text Classification',
        ],
        'TaskTitle': 'Multi-label text classification task of journal papers related to biomimicry',
        'TaskDescription': 'Select all labels that apply to the text shown',
        'NumberOfHumanWorkersPerDataObject': num_workers_per_paper,
        'TaskTimeLimitInSeconds': task_time_limit_in_seconds,
        'TaskAvailabilityLifetimeInSeconds': task_availability_in_seconds,
        'MaxConcurrentTaskCount': 10,
        'AnnotationConsolidationConfig': {
            'AnnotationConsolidationLambdaArn': 'arn:aws:lambda:us-east-1:432418664414:function:ACS-TextMultiClassMultiLabel'  # from the docs
        },
    },
    'Tags': [
        {
            'Key': 'string',
            'Value': 'string'
        },
    ]
}


####################
## Save info about the job into definitions folder
####################
with open(pathlib.Path(definition_dir).joinpath('labeling_job_arguments.json'),"w") as f:
    f.write(json.dumps(labeling_job_arguments))

response = sagemaker.create_labeling_job(**labeling_job_arguments)





####################
# Links used to help with the writing of this code
####################

# https://docs.aws.amazon.com/sagemaker/latest/dg/sms-text-classification-multilabel.html

# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.create_labeling_job

# https://docs.weka.io/additional-protocols/s3/s3-examples-using-boto3

# https://www.stackvidhya.com/difference-between-boto3-session-resource-client/

# https://pythonprogramminglanguage.com/kmeans-text-clustering/

# https://www.tutorialspoint.com/how-to-append-two-dataframes-in-pandas

# https://www.stackvidhya.com/create-empty-dataframe-pandas/

# https://datatofish.com/first-n-rows-pandas-dataframe/

# https://stackoverflow.com/questions/23195250/create-empty-dataframe-with-same-dimensions-as-another#comment112234830_34568092

# sorted_df = df.sort_values(by=['Column_name'], ascending=True)

# https://stackoverflow.com/questions/13784192/creating-an-empty-pandas-dataframe-then-filling-it




#
# df_labeled_papers = df[df['level1'].astype(bool)]
# labeled_abstracts = df_labeled_papers.abstract.tolist()
# X_labeled_abstracts = vectorizer.fit_transform(labeled_abstracts)
# model_labeled_abstracts = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1)
# model_labeled_abstracts.fit(X_labeled_abstracts)
# model_labeled_papers_clusters = model_labeled_abstracts.labels_.tolist()
# petalIDs = df_labeled_papers.petalID.tolist()
# level1s = df_labeled_papers.level1.tolist()
# labeled_papers_dict = { 'abstract': labeled_abstracts, 'cluster': model_labeled_papers_clusters,
#                         'petalID': petalIDs,
#                         'level1': level1s
#                         }
#
# df_labeled_papers_with_cluster = pd.DataFrame(labeled_papers_dict, index = [model_labeled_papers_clusters] ,
#                                      columns = ['abstract', 'cluster', 'petalID',
#                                                 'level1'])
# print(df_labeled_papers_with_cluster['cluster'].value_counts())
#
# df_labeled_papers_grouped_by_cluster = df_labeled_papers_with_cluster.groupby(df_labeled_papers_with_cluster['cluster']) #groupby cluster for aggregation purposes
#
#
# ### gather up papers for labeling
# num_labeled_papers_per_cluster_to_label = num_labeled_papers_to_label // num_clusters
# labeled_papers_to_be_labeled = []
# for group in df_labeled_papers_grouped_by_cluster.groups.keys():
#     df_cluster = df_labeled_papers_grouped_by_cluster.get_group(group)
#     for paper in df_cluster.head(num_labeled_papers_per_cluster_to_label).itertuples():
#         labeled_papers_to_be_labeled.append(paper)
#
# df_not_labeled_papers = df[~ df['level1'].astype(bool)]
# not_labeled_abstracts = df_not_labeled_papers.abstract.tolist()
# X_not_labeled_abstracts = vectorizer.fit_transform(not_labeled_abstracts)
# model_not_labeled_abstracts = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1)
# model_not_labeled_abstracts.fit(X_not_labeled_abstracts)
# model_not_labeled_papers_clusters = model_not_labeled_abstracts.labels_.tolist()
# petalIDs = df_not_labeled_papers.petalID.tolist()
# level1s = df_not_labeled_papers.level1.tolist()
# not_labeled_papers_dict = { 'abstract': not_labeled_abstracts, 'cluster': model_not_labeled_papers_clusters,
#                             'petalID': petalIDs,
#                             'level1': level1s}
#
# df_not_labeled_papers_with_cluster = pd.DataFrame(not_labeled_papers_dict, index = [model_not_labeled_papers_clusters] ,
#                                      columns = ['abstract', 'cluster', 'petalID', 'level1'])
#
# df_not_labeled_papers_by_cluster = df_not_labeled_papers_with_cluster.groupby(df_not_labeled_papers_with_cluster['cluster']) #groupby cluster for aggregation purposes
#
# ### gather up papers for labeling
# num_not_labeled_papers_per_cluster_to_label = num_not_labeled_papers_to_label // num_clusters
# not_labeled_papers_to_be_labeled = []
# for group in df_not_labeled_papers_by_cluster.groups.keys():
#     df_cluster = df_not_labeled_papers_by_cluster.get_group(group)
#     for paper in df_cluster.head(num_not_labeled_papers_per_cluster_to_label).itertuples():
#         not_labeled_papers_to_be_labeled.append(paper)
#
#
# print(df_not_labeled_papers_by_cluster['cluster'].value_counts())

#
# response = sagemaker.create_labeling_job(
#     LabelingJobName=labeling_job_name,
#     LabelAttributeName='label',
#     InputConfig={
#         'DataSource': {
#             'S3DataSource': {
#                 'ManifestS3Uri': f's3://{s3_bucket_name}/{ground_truth_manifest_filename}'
#             }
#         },
#         'DataAttributes': {
#             "ContentClassifiers": [
#                 "FreeOfPersonallyIdentifiableInformation",
#                 "FreeOfAdultContent"
#             ]
#         }
#     },
#     OutputConfig={
#         'S3OutputPath': f's3://{s3_bucket_name}',
#         # 'KmsKeyId': 'string'
#     },
#     RoleArn=role_arn,
#     LabelCategoryConfigS3Uri=f's3://{s3_bucket_name}/{label_category_configuration_file}',
#     StoppingConditions={
#         'MaxHumanLabeledObjectCount': 30,
#         'MaxPercentageOfInputDatasetLabeled': 100
#     },
#     HumanTaskConfig={
#         'WorkteamArn': work_team_arn,
#         'UiConfig': {
#             'UiTemplateS3Uri': f's3://{s3_bucket_name}/files-for-jobs/{custom_worker_task_template}'
#         },
#         'PreHumanTaskLambdaArn': 'arn:aws:lambda:us-east-1:432418664414:function:PRE-TextMultiClassMultiLabel',  # From the docs
#         'TaskKeywords': [
#             'Text Classification',
#         ],
#         'TaskTitle': 'Multi-label text classification task of journal papers related to biomimicry',
#         'TaskDescription': 'Select all labels that apply to the text shown',
#         'NumberOfHumanWorkersPerDataObject': 3,
#         'TaskTimeLimitInSeconds': 10 * 60,
#         'TaskAvailabilityLifetimeInSeconds': 10 * 24 * 60 * 60,
#         'MaxConcurrentTaskCount': 10,
#         'AnnotationConsolidationConfig': {
#             'AnnotationConsolidationLambdaArn': 'arn:aws:lambda:us-east-1:432418664414:function:ACS-TextMultiClassMultiLabel'  # from the docs
#         },
#     },
#     Tags=[
#         {
#             'Key': 'string',
#             'Value': 'string'
#         },
#     ]
# )
