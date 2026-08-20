import asyncio
import httpx

async def get_github_user(client: httpx.AsyncClient, username: str) -> dict:
    try:
        response = await client.get(f"https://api.github.com/users/{username}")
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            print(f"User '{username}' not found on GitHub")
        else:
            print(f"HTTP error: {e}")
    except httpx.ConnectError:
        print("No internet connection")
    return {}
    

async def main() -> None:
    username = input("Enter GitHub username: ")
    print(f"You entered: {username}")

    async with httpx.AsyncClient() as client:
        task = [get_github_user(client, username)]
        results = await asyncio.gather(*task)

    for user in results:
        bio = user['bio'] or "No bio available"
        location = user['location'] or "Location not specified"
        print(f"{user['name']} — {user['public_repos']} repos — {user['followers']} followers — {location} — {bio}")

    with open("profilesummary.txt","w") as f:
        f.write(f"{user['name']} — {user['public_repos']} repos — {user['followers']} followers — {location} — {bio}")

asyncio.run(main())