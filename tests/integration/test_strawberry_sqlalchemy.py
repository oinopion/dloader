from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Generator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any

import pytest
import strawberry
from sqlalchemy import ForeignKey, event, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from strawberry.types import Info

from dloader import DataLoader

AsyncSessionMaker = async_sessionmaker[AsyncSession]


@dataclass
class GraphQLContext:
    session: AsyncSession
    organization_loader: DataLoader[int, Organization | None]
    projects_by_org_loader: DataLoader[int, list[Project]]
    project_loader: DataLoader[int, Project | None]
    tasks_by_project_loader: DataLoader[int, list[Task]]
    tasks_by_user_loader: DataLoader[int, list[Task]]
    user_loader: DataLoader[int, User | None]


class Base(DeclarativeBase):
    pass


class Organization(Base):
    __tablename__ = "organizations"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]

    projects: Mapped[list[Project]] = relationship(back_populates="organization")


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    organization_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"))

    organization: Mapped[Organization] = relationship(back_populates="projects")
    tasks: Mapped[list[Task]] = relationship(back_populates="project")


class Task(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str]
    description: Mapped[str]
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id"))
    assignee_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))

    project: Mapped[Project] = relationship(back_populates="tasks")
    assignee: Mapped[User | None] = relationship(back_populates="assigned_tasks")


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    email: Mapped[str] = mapped_column(unique=True)

    assigned_tasks: Mapped[list[Task]] = relationship(back_populates="assignee")


async def load_organizations(session: AsyncSession, organization_ids: Sequence[int]) -> Sequence[Organization | None]:
    result = await session.execute(select(Organization).where(Organization.id.in_(organization_ids)))
    organizations = {org.id: org for org in result.scalars()}
    return [organizations.get(org_id) for org_id in organization_ids]


async def load_projects_by_org(session: AsyncSession, organization_ids: Sequence[int]) -> Sequence[list[Project]]:
    result = await session.execute(select(Project).where(Project.organization_id.in_(organization_ids)))
    projects_map: dict[int, list[Project]] = {org_id: [] for org_id in organization_ids}
    for project in result.scalars():
        projects_map[project.organization_id].append(project)
    return [projects_map[org_id] for org_id in organization_ids]


async def load_projects(session: AsyncSession, project_ids: Sequence[int]) -> Sequence[Project | None]:
    result = await session.execute(select(Project).where(Project.id.in_(project_ids)))
    projects = {proj.id: proj for proj in result.scalars()}
    return [projects.get(proj_id) for proj_id in project_ids]


async def load_tasks_by_project(session: AsyncSession, project_ids: Sequence[int]) -> Sequence[list[Task]]:
    result = await session.execute(select(Task).where(Task.project_id.in_(project_ids)))
    tasks_map: dict[int, list[Task]] = {proj_id: [] for proj_id in project_ids}
    for task in result.scalars():
        tasks_map[task.project_id].append(task)
    return [tasks_map[proj_id] for proj_id in project_ids]


async def load_tasks_by_user(session: AsyncSession, user_ids: Sequence[int]) -> Sequence[list[Task]]:
    result = await session.execute(select(Task).where(Task.assignee_id.in_(user_ids)))
    tasks_map: dict[int, list[Task]] = {user_id: [] for user_id in user_ids}
    for task in result.scalars():
        if task.assignee_id is not None:
            tasks_map[task.assignee_id].append(task)
    return [tasks_map[user_id] for user_id in user_ids]


async def load_users(session: AsyncSession, user_ids: Sequence[int]) -> Sequence[User | None]:
    result = await session.execute(select(User).where(User.id.in_(user_ids)))
    users = {user.id: user for user in result.scalars()}
    return [users.get(user_id) for user_id in user_ids]


async def load_users_with_error_on_666(session: AsyncSession, user_ids: Sequence[int]) -> Sequence[User | None]:
    if 666 in user_ids:
        raise RuntimeError("User 666 is forbidden!")

    result = await session.execute(select(User).where(User.id.in_(user_ids)))
    users = {user.id: user for user in result.scalars()}
    return [users.get(user_id) for user_id in user_ids]


@strawberry.type
class OrganizationType:
    id: int
    name: str

    @strawberry.field
    async def projects(self, info: Info[GraphQLContext, None]) -> list[ProjectType]:
        loader = info.context.projects_by_org_loader
        projects = await loader.load(self.id)
        return [ProjectType(id=p.id, name=p.name, organization_id=p.organization_id) for p in projects]


@strawberry.type
class ProjectType:
    id: int
    name: str
    organization_id: int

    @strawberry.field
    async def organization(self, info: Info[GraphQLContext, None]) -> OrganizationType | None:
        loader = info.context.organization_loader
        org = await loader.load(self.organization_id)
        return OrganizationType(id=org.id, name=org.name) if org else None

    @strawberry.field
    async def tasks(self, info: Info[GraphQLContext, None]) -> list[TaskType]:
        loader = info.context.tasks_by_project_loader
        tasks = await loader.load(self.id)
        return [
            TaskType(
                id=t.id,
                title=t.title,
                description=t.description,
                project_id=t.project_id,
                assignee_id=t.assignee_id,
            )
            for t in tasks
        ]


@strawberry.type
class TaskType:
    id: int
    title: str
    description: str
    project_id: int
    assignee_id: int | None

    @strawberry.field
    async def project(self, info: Info[GraphQLContext, None]) -> ProjectType | None:
        loader = info.context.project_loader
        proj = await loader.load(self.project_id)
        return ProjectType(id=proj.id, name=proj.name, organization_id=proj.organization_id) if proj else None

    @strawberry.field
    async def assignee(self, info: Info[GraphQLContext, None]) -> UserType | None:
        if self.assignee_id is None:
            return None
        loader = info.context.user_loader
        user = await loader.load(self.assignee_id)
        return UserType(id=user.id, name=user.name, email=user.email) if user else None


@strawberry.type
class UserType:
    id: int
    name: str
    email: str

    @strawberry.field
    async def assigned_tasks(self, info: Info[GraphQLContext, None]) -> list[TaskType]:
        loader = info.context.tasks_by_user_loader
        tasks = await loader.load(self.id)
        return [
            TaskType(
                id=t.id,
                title=t.title,
                description=t.description,
                project_id=t.project_id,
                assignee_id=t.assignee_id,
            )
            for t in tasks
        ]


@strawberry.type
class Query:
    @strawberry.field
    async def organizations(self, info: Info[GraphQLContext, None]) -> list[OrganizationType]:
        session = info.context.session
        result = await session.execute(select(Organization))
        return [OrganizationType(id=org.id, name=org.name) for org in result.scalars()]

    @strawberry.field
    async def organization(self, info: Info[GraphQLContext, None], id: int) -> OrganizationType | None:
        loader = info.context.organization_loader
        org = await loader.load(id)
        return OrganizationType(id=org.id, name=org.name) if org else None

    @strawberry.field
    async def users(self, info: Info[GraphQLContext, None]) -> list[UserType]:
        session = info.context.session
        result = await session.execute(select(User))
        return [UserType(id=user.id, name=user.name, email=user.email) for user in result.scalars()]

    @strawberry.field
    async def user(self, info: Info[GraphQLContext, None], id: int) -> UserType | None:
        loader = info.context.user_loader
        user = await loader.load(id)
        return UserType(id=user.id, name=user.name, email=user.email) if user else None


@asynccontextmanager
async def create_graphql_context(
    session_factory: AsyncSessionMaker,
    use_error_user_loader: bool = False,
) -> AsyncIterator[GraphQLContext]:
    async with session_factory() as session:
        organization_loader = DataLoader[int, Organization | None](load_fn=lambda ids: load_organizations(session, ids))
        projects_by_org_loader = DataLoader[int, list[Project]](load_fn=lambda ids: load_projects_by_org(session, ids))
        project_loader = DataLoader[int, Project | None](load_fn=lambda ids: load_projects(session, ids))
        tasks_by_project_loader = DataLoader[int, list[Task]](load_fn=lambda ids: load_tasks_by_project(session, ids))
        tasks_by_user_loader = DataLoader[int, list[Task]](load_fn=lambda ids: load_tasks_by_user(session, ids))

        if use_error_user_loader:
            user_loader = DataLoader[int, User | None](load_fn=lambda ids: load_users_with_error_on_666(session, ids))
        else:
            user_loader = DataLoader[int, User | None](load_fn=lambda ids: load_users(session, ids))

        async with AsyncExitStack() as stack:
            await stack.enter_async_context(organization_loader)
            await stack.enter_async_context(projects_by_org_loader)
            await stack.enter_async_context(project_loader)
            await stack.enter_async_context(tasks_by_project_loader)
            await stack.enter_async_context(user_loader)
            await stack.enter_async_context(tasks_by_user_loader)

            yield GraphQLContext(
                session=session,
                organization_loader=organization_loader,
                projects_by_org_loader=projects_by_org_loader,
                project_loader=project_loader,
                tasks_by_project_loader=tasks_by_project_loader,
                user_loader=user_loader,
                tasks_by_user_loader=tasks_by_user_loader,
            )


@dataclass
class SeedData:
    organizations: list[Organization]
    projects: list[Project]
    users: list[User]
    tasks: list[Task]
    error_user: User  # User 666 that triggers errors


@pytest.fixture
async def engine() -> AsyncGenerator[AsyncEngine, None]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
async def session_factory(engine: AsyncEngine) -> AsyncSessionMaker:
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture
async def seed_data(session_factory: AsyncSessionMaker) -> SeedData:
    async with session_factory() as session:
        orgs = [
            Organization(id=1, name="Tech Corp"),
            Organization(id=2, name="Design Studio"),
            Organization(id=3, name="Marketing Agency"),
        ]
        session.add_all(orgs)

        projects: list[Project] = []
        project_id = 1
        for org in orgs:
            for i in range(5):
                projects.append(
                    Project(
                        id=project_id,
                        name=f"{org.name} Project {i + 1}",
                        organization_id=org.id,
                    )
                )
                project_id += 1
        session.add_all(projects)

        regular_users = [User(id=i, name=f"User {i}", email=f"user{i}@example.com") for i in range(1, 11)]
        error_user = User(id=666, name="Forbidden User", email="forbidden@example.com")
        all_users = [*regular_users, error_user]
        session.add_all(all_users)

        tasks: list[Task] = []
        task_id = 1
        for project in projects:
            for i in range(10):
                assignee_id = (task_id % 10) + 1 if i % 2 == 0 else None
                tasks.append(
                    Task(
                        id=task_id,
                        title=f"Task {task_id}",
                        description=f"Description for task {task_id}",
                        project_id=project.id,
                        assignee_id=assignee_id,
                    )
                )
                task_id += 1
        session.add_all(tasks)

        await session.commit()

        return SeedData(
            organizations=orgs,
            projects=projects,
            users=all_users,
            tasks=tasks,
            error_user=error_user,
        )


class QueryTracker:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def track_query(
        self, conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, executemany: Any
    ) -> None:
        if statement.strip().upper().startswith("SELECT"):
            self.queries.append(statement)

    def reset(self) -> None:
        self.queries = []


@pytest.fixture
def query_tracker(engine: AsyncEngine) -> Generator[QueryTracker, None, None]:
    tracker = QueryTracker()
    event.listen(engine.sync_engine, "before_cursor_execute", tracker.track_query)
    yield tracker
    event.remove(engine.sync_engine, "before_cursor_execute", tracker.track_query)


async def test_n_plus_one_prevention(
    session_factory: AsyncSessionMaker,
    seed_data: SeedData,
    query_tracker: QueryTracker,
) -> None:
    schema = strawberry.Schema(query=Query)

    query = """
    query {
        organizations {
            id
            name
            projects {
                id
                name
                tasks {
                    id
                    title
                }
            }
        }
    }
    """

    query_tracker.reset()

    async with create_graphql_context(session_factory) as context:
        result = await schema.execute(query, context_value=context)

        assert result.errors is None
        assert result.data is not None

        org_data = result.data["organizations"]
        assert len(org_data) == len(seed_data.organizations)

        projects_per_org = len(seed_data.projects) // len(seed_data.organizations)
        tasks_per_project = len(seed_data.tasks) // len(seed_data.projects)

        for org in org_data:
            assert len(org["projects"]) == projects_per_org
            for project in org["projects"]:
                assert len(project["tasks"]) == tasks_per_project

        assert len(query_tracker.queries) == 3


async def test_deep_nesting_with_deduplication(
    session_factory: AsyncSessionMaker,
    seed_data: SeedData,
    query_tracker: QueryTracker,
) -> None:
    schema = strawberry.Schema(query=Query)

    query = """
    query {
        users {
            id
            name
            assignedTasks {
                id
                title
                project {
                    id
                    name
                    organization {
                        id
                        name
                        projects {
                            id
                            name
                        }
                    }
                }
            }
        }
    }
    """

    query_tracker.reset()

    async with create_graphql_context(session_factory) as context:
        result = await schema.execute(query, context_value=context)

        assert result.errors is None
        assert result.data is not None

        users_data = result.data["users"]
        assert len(users_data) == len(seed_data.users)

        assert len(query_tracker.queries) == 5


async def test_dataloader_cleanup_on_error(
    session_factory: AsyncSessionMaker,
    seed_data: SeedData,
) -> None:
    schema = strawberry.Schema(query=Query)

    # Use actual user IDs from seed data
    first_user = seed_data.users[0]
    second_user = seed_data.users[1]
    error_user = seed_data.error_user

    query = f"""
    query {{
        organizations {{
            id
            name
        }}
        user1: user(id: {first_user.id}) {{
            id
            name
        }}
        user2: user(id: {second_user.id}) {{
            id
            name
        }}
        errorUser: user(id: {error_user.id}) {{
            id
            name
        }}
    }}
    """

    async with create_graphql_context(session_factory, use_error_user_loader=True) as context:
        org_loader = context.organization_loader
        user_loader = context.user_loader

        result = await schema.execute(query, context_value=context)

        assert result.errors is not None
        assert any("User 666 is forbidden!" in str(e) for e in result.errors)

    # Verify DataLoaders were properly shut down after context exit
    assert len(org_loader._running_load_tasks) == 0  # type: ignore[attr-defined]
    assert len(user_loader._running_load_tasks) == 0  # type: ignore[attr-defined]
    assert org_loader._scheduled_load_task is None  # type: ignore[attr-defined]
    assert user_loader._scheduled_load_task is None  # type: ignore[attr-defined]
