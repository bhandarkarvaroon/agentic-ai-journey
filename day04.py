import requests

def get_github_user(username: str) -> dict:
    try:
        response = requests.get(f"https://api.github.com/users/{username}")
        response.raise_for_status()  # raises an error if status is 4xx or 5xx
        return response.json()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            print(f"User '{username}' not found on GitHub")
        else:
            print(f"HTTP error: {e}")
    except requests.exceptions.ConnectionError:
        print("No internet connection")
    except Exception as e:
        print(f"Something went wrong: {e}")
    return {}

user = get_github_user("torvalds")
if user:
    print(user["name"])
    print(user["public_repos"])

# Now test with a bad username
print("\n--- Testing bad username ---")
bad_user = get_github_user("this_user_does_not_exist_xyz123")
print(bad_user)
