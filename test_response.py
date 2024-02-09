import requests

# Set the URL of your Flask server
url = "http://localhost:5000/chat"

# Prepare the JSON data for the request
json_data = {
    "query": "Who are you",
    "context": "Your context here",
    "init": True  # Set to True if it's the initial request
}

# Make the POST request
response = requests.post(url, json=json_data)

# Print the response from the server
print(response.json())
