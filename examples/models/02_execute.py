"""在线模型 → Planner → Runtime → 独立 WRS 节点 → 真实 UR7E 模型。"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from wrs_agent.bindings import load_bindings
from wrs_agent.planner import ModelPlanner
from wrs_agent.planner.providers.llm import LLMClient, LLMConfig, LLMError
from wrs_agent.processes import LocalStack
from wrs_agent.runtime import Runtime
from wrs_agent.schemas import GoalRequest, new_id

CONFIG = Path(__file__).resolve().parents[2] / "configs/actions.toml"
GOAL = "先移动到命名姿态 B，然后在世界坐标系向上移动 2 厘米。"


async def main():
    # LLM_* 环境变量选择协议、端点与模型；本脚本的 Runtime 使用在线模型，不启动另一份 Agent。
    try:
        model = LLMClient(LLMConfig.from_env(), live_model=True)
    except LLMError as exc:
        raise SystemExit(f"模型配置错误：{exc.error.message}") from None
    try:
        async with LocalStack(backend="wrs", bindings=CONFIG) as stack:
            runtime = Runtime(
                stack.system.clients,
                load_bindings(CONFIG)[1],
                ModelPlanner(model),
                registry=stack.system.registry,
            )
            try:
                request = GoalRequest(request_id=new_id(), goal=GOAL)
                await runtime.goal(request)
                await runtime.planning
                planned = runtime.goal_status(request.request_id)
                print("规划结果：", planned["state"])
                if planned["task_id"]:
                    print("任务结果：", runtime.task_status(planned["task_id"])["state"])
                    snapshot = await stack.system.snapshot()
                    print("TCP 位置：", snapshot.data.robot.kinematics.tcp_pos)
                else:
                    print("说明：", planned.get("reason"))
            finally:
                await runtime.close()
    finally:
        await model.aclose()


if __name__ == "__main__":
    # 不依赖 IDE 的工作目录；显式环境变量优先于仓库 .env。
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False, encoding="utf-8-sig")
    asyncio.run(main())
