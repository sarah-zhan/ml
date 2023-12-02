# Lambda
# import requests

# url = 'http://localhost:8080/2015-03-31/functions/function/invocations' #local
#url = "https://qxqypeg9z3.execute-api.us-east-2.amazonaws.com/test/predict" #aws
# data = {
#     'url': 'https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg'
# }

# result = requests.post(url, json=data).json()
# print(result)


# API Gateway; avoid having credientials errors
import boto3

client = boto3.client('apigateway')
response = client.test_invoke_method(
    restApiId='qxqypeg9z3',
    resourceId='3jq5dco7bi',
    httpMethod='POST',
    pathWithQueryString='/test/predict',
    body='{"url": "https://habrastorage.org/webt/rt/d9/dh/rtd9dhsmhwrdezeldzoqgijdg8a.jpeg"}'
)
print(response['body'])


