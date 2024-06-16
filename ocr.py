import os

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
import time

from dotenv import load_dotenv
load_dotenv()

# set `<your-endpoint>` and `<your-key>` variables with the values from the Azure portal
endpoint = os.getenv('ENDPOINT')
key = os.getenv('KEY')


def process_image():
    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

    # Get an image with text
    read_image_url = "image2.jpeg"

    with open(read_image_url, "rb") as image_data:
        read_response = computervision_client.read_in_stream(image_data, raw=True)

    read_operation_location = read_response.headers["Operation-Location"]
    # Grab the ID from the URL
    operation_id = read_operation_location.split("/")[-1]

    # Call the "GET" API and wait for it to retrieve the results
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status not in ['notStarted', 'running']:
            break
        time.sleep(1)

    ingredients_string = ""
    # Print the detected text, line by line
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                ingredients_string += line.text

    return ingredients_string
