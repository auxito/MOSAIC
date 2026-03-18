"""
表单式简历输入 — 交互式收集简历信息。
v1 仅支持文本输入，架构预留文件解析扩展。
"""

from __future__ import annotations

from mosaic.resume.schema import (
    Education,
    Project,
    ResumeData,
    WorkExperience,
)


def create_sample_resume() -> ResumeData:
    """创建示例简历（用于演示和测试）"""
    return ResumeData(
        name="张三",
        target_position="高级后端工程师",
        email="zhangsan@example.com",
        phone="138-0000-0000",
        summary="5年Python后端开发经验，熟悉微服务架构和云原生技术。",
        education=[
            Education(
                school="北京大学",
                degree="硕士",
                major="计算机科学与技术",
                start_year="2016",
                end_year="2019",
                gpa="3.7/4.0",
            ),
            Education(
                school="武汉大学",
                degree="学士",
                major="软件工程",
                start_year="2012",
                end_year="2016",
            ),
        ],
        work_experience=[
            WorkExperience(
                company="字节跳动",
                title="高级后端工程师",
                start_date="2021-03",
                end_date="至今",
                description="负责推荐系统后端服务开发和优化",
                achievements=[
                    "主导推荐引擎重构，QPS 提升 3 倍",
                    "设计并实现 A/B 测试平台，支持 200+ 并行实验",
                    "带领 5 人小组完成微服务拆分",
                ],
                tech_stack=["Python", "Go", "Redis", "Kafka", "K8s"],
            ),
            WorkExperience(
                company="美团",
                title="后端工程师",
                start_date="2019-07",
                end_date="2021-02",
                description="外卖订单系统开发",
                achievements=[
                    "优化订单查询接口，P99 延迟从 200ms 降至 50ms",
                    "参与订单系统容灾方案设计",
                ],
                tech_stack=["Java", "Spring Boot", "MySQL", "RocketMQ"],
            ),
        ],
        projects=[
            Project(
                name="智能推荐引擎 v2",
                role="技术负责人",
                description="基于深度学习的个性化推荐系统",
                achievements=[
                    "CTR 提升 15%，用户时长增加 8%",
                    "设计实时特征工程管道，延迟 < 10ms",
                ],
                tech_stack=["PyTorch", "TensorFlow Serving", "Flink"],
            ),
        ],
        skills=[
            "Python", "Go", "Java",
            "Redis", "MySQL", "MongoDB",
            "Kafka", "Docker", "Kubernetes",
            "机器学习", "系统设计",
        ],
        certifications=["AWS Solutions Architect Associate"],
    )


def resume_from_text(text: str) -> ResumeData:
    """
    从纯文本创建简历（简单解析）。
    v1: 将整个文本作为 summary，后续可接 LLM 结构化。
    """
    return ResumeData(
        name="候选人",
        summary=text.strip(),
    )
