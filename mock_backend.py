import asyncio


class MockBackend:
    async def generate(self, request):
        last_user_message = None
        for message in reversed(request.messages):
            if message.role == "user":
                last_user_message = message.content
                break

        if last_user_message is None:
            raise ValueError("No user message found in the request.")

        return {
            "model": request.model,
            "content": f"Mock response: {last_user_message}",
            "role": "assistant",
        }

    async def generate_batch(self, requests):
        await asyncio.sleep(0.01)
        return [await self.generate(request) for request in requests]
