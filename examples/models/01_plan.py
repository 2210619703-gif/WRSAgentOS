"""在线调用所配置的模型，根据 WRS 节点的真实状态和能力生成计划，不执行运动。"""

import asyncio
from pathlib import Path

from dotenv import load_dotenv

from wrs_agent.planner import ModelPlanner, PlanRequest
from wrs_agent.planner.providers.llm import LLMClient, LLMConfig, LLMError
from wrs_agent.processes import LocalStack

CONFIG = Path(__file__).resolve().parents[2] / "configs/robot.toml"
GOAL = "将机械臂移动到命名姿态 B。"


async def main():
    # LLM_PROTOCOL、LLM_BASE_URL、LLM_MODEL、LLM_API_KEY 从环境读取；运行本文件会调用在线模型。
    try:
        model = LLMClient(LLMConfig.from_env(), live_model=True)
    except LLMError as exc:
        raise SystemExit(f"模型配置错误：{exc.error.message}") from None
    try:
        async with LocalStack(backend="wrs", bindings=CONFIG) as stack:
            snapshot = await stack.system.snapshot()
            skills = await stack.system.skills()
            planner = ModelPlanner(model)
            decision = await planner.plan(
                PlanRequest(
                    user_goal=GOAL,
                    world=snapshot.data.model_dump(),
                    skills=[skill.model_dump() for skill in skills],
                )
            )
            print("规划决定：", decision.kind)
            print("计划：", decision.plan)
            print("说明：", decision.text)
            config = model.config
            effort = config.reasoning_effort or "服务默认"
            print(f"模型：{config.model}（{config.protocol}），推理深度：{effort}")
            timing = planner.last_timing  # 观测用的单调耗时，不是超时或安全依据。
            print(
                f"耗时：等待首字节 {timing.model.wait:.3f} 秒，"
                f"读取 {timing.model.receive:.3f} 秒，解析 {timing.model.parse:.3f} 秒，"
                f"本地校验 {timing.validate:.3f} 秒，合计 {timing.total:.3f} 秒"
            )
            # 服务商回报的用量，不是本地测得的数字；Chat 与 Responses/Claude 的字段名不同。
            usage = planner.last_usage
            details = (
                usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
            )
            print(
                f"用量：输入 {usage.get('prompt_tokens', usage.get('input_tokens'))} tokens，"
                f"输出 {usage.get('completion_tokens', usage.get('output_tokens'))} tokens，"
                f"其中推理 {details.get('reasoning_tokens')} tokens"
            )
    except LLMError as exc:
        raise SystemExit(f"模型请求失败（{exc.code}）：{exc.error.message}") from None
    finally:
        await model.aclose()


if __name__ == "__main__":
    # 不依赖 IDE 的工作目录；显式环境变量优先于仓库 .env。
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False, encoding="utf-8-sig")
    asyncio.run(main())
