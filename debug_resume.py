import asyncio

from xarness.tui.app import AgentApp
from xarness.tui.model_screen import ModelPickerScreen
from xarness.tui.resume_screen import ResumeScreen
from xarness.tui.widgets import ChatInput

import sys
sys.path.insert(0, "tests")
from test_tui_screens import make_app  # noqa


async def main() -> None:
    app = make_app([])
    async with app.run_test() as pilot:
        await pilot.press("/", "r", "e", "s", "u", "m", "e", "tab", "enter")
        for i in range(20):
            await pilot.pause()
            if isinstance(app.screen, ResumeScreen):
                break
        print("resume flow -> screen:", app.screen)

    app2 = make_app([], config_path=None, profile_names=["alpha", "beta"])
    async with app2.run_test() as pilot:
        await pilot.press("/", "m", "o", "d", "e", "l", "tab", "enter")
        for i in range(20):
            await pilot.pause()
            if isinstance(app2.screen, ModelPickerScreen):
                break
        print("model flow -> screen:", app2.screen)


asyncio.run(main())