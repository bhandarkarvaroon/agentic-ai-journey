import asyncio
import httpx

async def get_github_user(client: httpx.AsyncClient, username: str) -> dict:
    response = await client.get(f"https://api.github.com/users/{username}")
    response.raise_for_status()
    return response.json()

async def main() -> None:
    usernames = ["torvalds", "gvanrossum", "dhh", "bhandarkarvaroon"]

    async with httpx.AsyncClient() as client:
        tasks = [get_github_user(client, u) for u in usernames]
        results = await asyncio.gather(*tasks)

    for user in results:
        print(f"{user['name']} — {user['public_repos']} repos")

asyncio.run(main())
