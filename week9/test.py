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


