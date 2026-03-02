"""Tests that repo soft-delete cascades to aweb agent soft-delete."""

from __future__ import annotations

from uuid import UUID

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from beadhub.api import create_app


@pytest.mark.asyncio
async def test_repo_delete_cascades_to_agents(db_infra, redis_client_async, init_workspace):
    """DELETE /v1/repos/{id} should soft-delete aweb agent records for all workspaces."""
    app = create_app(db_infra=db_infra, redis=redis_client_async, serve_frontend=False)
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            slug = "repo-cascade-test"
            repo_origin = "git@github.com:test/repo-cascade.git"

            # Create two workspaces in the same repo
            init1 = await init_workspace(
                client,
                project_slug=slug,
                repo_origin=repo_origin,
                alias="repo-agent-1",
            )
            init2 = await init_workspace(
                client,
                project_slug=slug,
                repo_origin=repo_origin,
                alias="repo-agent-2",
            )

            # workspace_id == agent_id (v1 mapping)
            agent_id_1 = init1["workspace_id"]
            agent_id_2 = init2["workspace_id"]
            repo_id = init1["repo_id"]
            assert init2["repo_id"] == repo_id, "Both should share the same repo"

            # Verify both agents are active
            aweb_db = db_infra.get_manager("aweb")
            for agent_id in [agent_id_1, agent_id_2]:
                agent = await aweb_db.fetch_one(
                    """
                    SELECT status, deleted_at
                    FROM {{tables.agents}}
                    WHERE agent_id = $1 AND deleted_at IS NULL
                    """,
                    UUID(agent_id),
                )
                assert agent is not None, f"Agent {agent_id} should be active"

            # Delete the repo
            delete_resp = await client.delete(f"/v1/repos/{repo_id}")
            assert delete_resp.status_code == 200, delete_resp.text

            # Both agents should now be soft-deleted
            for agent_id in [agent_id_1, agent_id_2]:
                agent_after = await aweb_db.fetch_one(
                    """
                    SELECT status, deleted_at
                    FROM {{tables.agents}}
                    WHERE agent_id = $1
                    """,
                    UUID(agent_id),
                )
                assert agent_after is not None, "Agent record should still exist"
                assert (
                    agent_after["deleted_at"] is not None
                ), f"Agent {agent_id} should be soft-deleted"
                assert agent_after["status"] == "deregistered"
