'''
Get collection of papers
create AWS GT job with labels and instructions and selecting the vendor
add papers to the job
'''

# https://docs.aws.amazon.com/sagemaker/latest/dg/sms-text-classification-multilabel.html

# https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sagemaker.html#SageMaker.Client.create_labeling_job

# https://docs.weka.io/additional-protocols/s3/s3-examples-using-boto3

# https://www.stackvidhya.com/difference-between-boto3-session-resource-client/

import boto3

# client = boto3.client('sagemaker')

aws_profile = 'GRC-VE00-PeTaL-Lab-mturk-script'

session = boto3.Session(profile_name=aws_profile)  # This profile was created using AWS command
# line tools. Creating that involved using access keys
sagemaker = session.client(
    service_name='sagemaker',
    region_name='us-east-1',
)

print(sagemaker)

print(sagemaker.list_workteams())

s3 = session.client(
    service_name='s3',
    region_name='us-east-1',
)

import boto3  # pip install boto3

# Let's use Amazon S3
# s3 = boto3.client("s3")
# print(dir(s3))
# for bucket in s3.list_buckets():
#     print(bucket)

# arn:aws:s3:::ground-truth-petal-test
# https://stackoverflow.com/questions/67876666/uploading-files-to-aws-s3-bucket-folder-in-python-causes-regex-error
# s3.upload_file(
#     "golden.json",
#     # "arn:aws:s3:::ground-truth-petal-test",
#     "ground-truth-petal-test",
#     "golden.json"
# )
# labels
#
# Assemble/break down structure
# Attach
# Chemically assemble/break down
# Manage mechanical forces
# Manipulate solids, liquids, gases, or energy
# Modify color/camouflage
# Modify size/shape/material properties
# Modify/convert energy
# Move on/through solids, liquids, or gases
# Protect from living or non-living threats
# Sense, send, or process information
# Sustain ecological community


### Make ManifestS3Uri file and upload to S3
# import boto3
# s3 = boto3.resource('s3')
# bucket = s3.Bucket('s3://ground-truth-petal-test/')


# read in golden.json from github !
import pandas as pd
# df = pd.read_json("https://raw.githubusercontent.com/nasa-petal/data-collection-and-prep/main/golden.json")
df = pd.read_json("golden.json")

# Do k-means clustering of abstracts
# https://pythonprogramminglanguage.com/kmeans-text-clustering/
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# Params
num_clusters = 7
num_labeled_papers_to_label = 30
num_not_labeled_papers_to_label = 70
ground_truth_manifest_filename = 'ground-truth-quadrant-pilot-1.manifest'
label_category_configuration_file = 'label_category_configuration.json'
labeling_job_name = 'ground-truth-petal-quadrant-test-3'


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score

vectorizer = TfidfVectorizer(stop_words='english')


df_labeled_papers = df[df['level1'].astype(bool)]
labeled_abstracts = df_labeled_papers.abstract.tolist()
X_labeled_abstracts = vectorizer.fit_transform(labeled_abstracts)
model_labeled_abstracts = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1)
model_labeled_abstracts.fit(X_labeled_abstracts)
model_labeled_papers_clusters = model_labeled_abstracts.labels_.tolist()
petalIDs = df_labeled_papers.petalID.tolist()
level1s = df_labeled_papers.level1.tolist()
labeled_papers_dict = { 'abstract': labeled_abstracts, 'cluster': model_labeled_papers_clusters,
                        'petalID': petalIDs,
                        'level1': level1s
                        }

df_labeled_papers_with_cluster = pd.DataFrame(labeled_papers_dict, index = [model_labeled_papers_clusters] ,
                                     columns = ['abstract', 'cluster', 'petalID',
                                                'level1'])
print(df_labeled_papers_with_cluster['cluster'].value_counts())

df_labeled_papers_grouped_by_cluster = df_labeled_papers_with_cluster.groupby(df_labeled_papers_with_cluster['cluster']) #groupby cluster for aggregation purposes


### gather up papers for labeling
num_labeled_papers_per_cluster_to_label = num_labeled_papers_to_label // num_clusters
labeled_papers_to_be_labeled = []
for group in df_labeled_papers_grouped_by_cluster.groups.keys():
    df_cluster = df_labeled_papers_grouped_by_cluster.get_group(group)
    for paper in df_cluster.head(num_labeled_papers_per_cluster_to_label).itertuples():
        labeled_papers_to_be_labeled.append(paper)

df_not_labeled_papers = df[~ df['level1'].astype(bool)]
not_labeled_abstracts = df_not_labeled_papers.abstract.tolist()
X_not_labeled_abstracts = vectorizer.fit_transform(not_labeled_abstracts)
model_not_labeled_abstracts = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1)
model_not_labeled_abstracts.fit(X_not_labeled_abstracts)
model_not_labeled_papers_clusters = model_not_labeled_abstracts.labels_.tolist()
petalIDs = df_not_labeled_papers.petalID.tolist()
level1s = df_not_labeled_papers.level1.tolist()
not_labeled_papers_dict = { 'abstract': not_labeled_abstracts, 'cluster': model_not_labeled_papers_clusters,
                            'petalID': petalIDs,
                            'level1': level1s}

df_not_labeled_papers_with_cluster = pd.DataFrame(not_labeled_papers_dict, index = [model_not_labeled_papers_clusters] ,
                                     columns = ['abstract', 'cluster', 'petalID', 'level1'])

df_not_labeled_papers_by_cluster = df_not_labeled_papers_with_cluster.groupby(df_not_labeled_papers_with_cluster['cluster']) #groupby cluster for aggregation purposes

### gather up papers for labeling
num_not_labeled_papers_per_cluster_to_label = num_not_labeled_papers_to_label // num_clusters
not_labeled_papers_to_be_labeled = []
for group in df_not_labeled_papers_by_cluster.groups.keys():
    df_cluster = df_not_labeled_papers_by_cluster.get_group(group)
    for paper in df_cluster.head(num_not_labeled_papers_per_cluster_to_label).itertuples():
        not_labeled_papers_to_be_labeled.append(paper)


print(df_not_labeled_papers_by_cluster['cluster'].value_counts())


papers_to_be_labeled = labeled_papers_to_be_labeled + not_labeled_papers_to_be_labeled


# get abstracts from each cluster, sorted by petal ID and take the first num_labeled/num_clusters


# split into labeled and unlabeled
# Take lowest petal ID numbered documents from each batch and put into the source file

# https://www.tutorialspoint.com/how-to-append-two-dataframes-in-pandas
# https://www.stackvidhya.com/create-empty-dataframe-pandas/
# https://datatofish.com/first-n-rows-pandas-dataframe/
# https://stackoverflow.com/questions/23195250/create-empty-dataframe-with-same-dimensions-as-another#comment112234830_34568092
# sorted_df = df.sort_values(by=['Column_name'], ascending=True)
# https://stackoverflow.com/questions/13784192/creating-an-empty-pandas-dataframe-then-filling-it



import json
with open(ground_truth_manifest_filename, 'w') as f:
    for paper in papers_to_be_labeled:
        item = {'source': paper.abstract}
        f.write(json.dumps(item) + "\n")

s3.upload_file(
    ground_truth_manifest_filename,
    # "arn:aws:s3:::ground-truth-petal-test",
    "ground-truth-petal-test",
    ground_truth_manifest_filename
)


labels = [
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
# Make the label category configuration file
label_category_config = {}
label_category_config['document-version'] = 'taxonomy-v3-2022-04-15'
labels_list_of_dicts = []
for label in labels:
    labels_list_of_dicts.append({'label': label})
label_category_config['labels'] = labels_list_of_dicts

with open(label_category_configuration_file, 'w') as f:
    f.write(json.dumps(label_category_config))
s3.upload_file(
    label_category_configuration_file,
    # "arn:aws:s3:::ground-truth-petal-test",
    "ground-truth-petal-test",
    label_category_configuration_file
)
#
#
# with open('source.json', 'w') as data:
#     for s in [
#               'frogs that build foam nests floating on water face the problems of '
#               'over-dispersion of the secretions used and eggs being dangerously',
#               'recent evidence suggests that bats can detect the geomagnetic field,'
#               'but the way in which this is used by them for navigation to a home roost',
#               ]:
#         data.write(s + '\n')
# with open('source.json', 'r') as data:
#     bucket.upload_fileobj(data, 'ground-truth-api-test.manifest')


# with open("source.json", "rb") as f:
#     s3.upload_fileobj(f, "s3://ground-truth-petal-test/", "ground-truth-api-test.manifest")

response = sagemaker.create_labeling_job(
    LabelingJobName=labeling_job_name,
    LabelAttributeName='label',
    InputConfig={
        'DataSource': {
            'S3DataSource': {
                'ManifestS3Uri': f's3://ground-truth-petal-test/{ground_truth_manifest_filename}'
            }
        },
        'DataAttributes': {
            "ContentClassifiers": [
                "FreeOfPersonallyIdentifiableInformation",
                "FreeOfAdultContent"
            ]
        }
    },
    OutputConfig={
        'S3OutputPath': 's3://ground-truth-petal-test',
        # 'KmsKeyId': 'string'
    },
    # RoleArn='arn:aws:iam::*:role/*,
    RoleArn='arn:aws:iam::056730517754:role/ground-truth-sagemaker-execution-role',

    LabelCategoryConfigS3Uri=f's3://ground-truth-petal-test/{label_category_configuration_file}',
    StoppingConditions={
        'MaxHumanLabeledObjectCount': 30,
        'MaxPercentageOfInputDatasetLabeled': 100
    },
    HumanTaskConfig={
        # 'WorkteamArn': 'arn:aws:sagemaker:us-east-1:056730517754:workteam/private-crowd/ground-truth-petal-test',
        'WorkteamArn': 'arn:aws:sagemaker:us-east-1:056730517754:workteam/private-crowd/ground-truth-just-herb',
        'UiConfig': {
            'UiTemplateS3Uri': 's3://ground-truth-petal-test/files-for-jobs/custom-worker-task-template.html'
        },
        'PreHumanTaskLambdaArn': 'arn:aws:lambda:us-east-1:432418664414:function:PRE-TextMultiClassMultiLabel',
        'TaskKeywords': [
            'Text Classification',
        ],
        'TaskTitle': 'Multi-label text classification task of journal papers related to biomimicry',
        'TaskDescription': 'Select all labels that apply to the text shown',
        'NumberOfHumanWorkersPerDataObject': 3,
        'TaskTimeLimitInSeconds': 10 * 60 ,
        'TaskAvailabilityLifetimeInSeconds': 10 * 24 * 60 * 60 ,
        'MaxConcurrentTaskCount': 10,
        'AnnotationConsolidationConfig': {
            'AnnotationConsolidationLambdaArn': 'arn:aws:lambda:us-east-1:432418664414:function:ACS-TextMultiClassMultiLabel'
        },
    },
    Tags=[
        {
            'Key': 'string',
            'Value': 'string'
        },
    ]
)


## Save info about the job that was created into a folder