import requests
import base64
import json

def test_extract_question():
    # Read the test image and convert to base64
    with open('/home/devbox/project/test/test_image.png', 'rb') as image_file:
        image_data = image_file.read()
    image_base64 = base64.b64encode(image_data).decode('utf-8')

    # Prepare the request payload with just the image data
    payload = {
        "data": {
            "image": image_base64
        }
    }

    # Make the request to the endpoint
    response = requests.post(
        'https://wnuuxrmfbply.sealosgzg.site/api/extract_question',
        headers={
            "Content-Type": "application/json"
        },
        data=json.dumps(payload)
    )

    # Print the response
    print("\nResponse Status Code:", response.status_code)
    print("\nResponse Content:", response.content.decode('utf-8'))

    # Parse the response
    try:
        result = response.json()
        print("\nExtracted Text:", result.get('extractedText', 'No text extracted'))
    except json.JSONDecodeError:
        print("\nError: Could not parse JSON response")

if __name__ == '__main__':
    test_extract_question()
